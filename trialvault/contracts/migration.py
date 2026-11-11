from ..validation import (
    name,
    object_value,
    list_value,
    canonical,
    digest,
    parse_json,
    reject,
    VaultError,
)
from ..contracts.evolution import compatibility


class SchemaMigration:
    def __init__(self, database, registry, ingestion):
        self.db = database
        self.registry = registry
        self.ingestion = ingestion
        with database.transaction() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS migrations(id TEXT PRIMARY KEY,request TEXT NOT NULL,
                report TEXT NOT NULL,state TEXT NOT NULL,result TEXT)""")

    def preview(
        self,
        migration_id,
        dataset,
        base_version,
        target_schema_version,
        renames=None,
        defaults=None,
        drop=None,
    ):
        name(migration_id)
        renames = renames or {}
        defaults = defaults or {}
        drop = drop or []
        object_value(renames)
        object_value(defaults)
        list_value(drop, 0, 256)
        if len(set(drop)) != len(drop):
            reject("migration", "duplicate dropped field")
        base = self.registry.version(dataset, base_version)
        schema_id = self.registry.dataset(dataset)["schema_id"]
        old_schema = self.registry.schema(schema_id, base["schema_version"])
        new_schema = self.registry.schema(schema_id, target_schema_version)
        if target_schema_version <= base["schema_version"]:
            reject("migration", "target schema must be newer")
        for old, new in renames.items():
            if old not in old_schema.by_name or new not in new_schema.by_name:
                reject("migration", "rename references unknown field")
            if old in drop:
                reject("migration", "cannot rename and drop same field")
        if len(set(renames.values())) != len(renames):
            reject("migration", "multiple fields rename to same target")
        if any(field not in old_schema.by_name for field in drop):
            reject("migration", "dropped field is unknown")
        if any(field not in new_schema.by_name for field in defaults):
            reject("migration", "default field is unknown")
        for field, value in defaults.items():
            new_schema.by_name[field].validate(value)
        request = {
            "dataset": dataset,
            "base_version": base_version,
            "target_schema_version": target_schema_version,
            "renames": renames,
            "defaults": defaults,
            "drop": drop,
        }
        valid = []
        errors = []
        keys = {}
        rows = self.ingestion.rows(dataset, base_version)
        for index, row in enumerate(rows, 1):
            candidate = {}
            try:
                for field, value in row.items():
                    if field in drop:
                        continue
                    destination = renames.get(field, field)
                    if destination in candidate:
                        reject(
                            "migration", "renamed field collides with retained field"
                        )
                    candidate[destination] = value
                for field, value in defaults.items():
                    candidate.setdefault(field, value)
                candidate = new_schema.validate(candidate)
                key = new_schema.row_key(candidate)
                if key in keys:
                    reject(
                        "migration", "new primary key collides", previous_row=keys[key]
                    )
                keys[key] = index
                valid.append(candidate)
            except VaultError as exc:
                errors.append(
                    {
                        "row": index,
                        "old_key": old_schema.row_key(row),
                        "error": exc.as_dict(),
                    }
                )
        valid.sort(key=new_schema.row_key)
        report = {
            "id": migration_id,
            "request": request,
            "base_digest": base["digest"],
            "compatibility": compatibility(old_schema, new_schema),
            "input_rows": len(rows),
            "valid_rows": len(valid),
            "error_count": len(errors),
            "errors": errors[:1000],
            "truncated": len(errors) > 1000,
            "ready": not errors,
            "result_digest": digest(valid),
            "rows": valid,
        }
        with self.db.transaction() as con:
            old = self.db.one(
                "SELECT request FROM migrations WHERE id=?", (migration_id,)
            )
            if old:
                if old["request"] != canonical(request):
                    reject("conflict", "migration id request differs")
                return self.get(migration_id)
            con.execute(
                "INSERT INTO migrations VALUES(?,?,?,'preview',NULL)",
                (migration_id, canonical(request), canonical(report)),
            )
            self.db.audit(
                "migration_preview",
                migration_id,
                {
                    "dataset": dataset,
                    "errors": len(errors),
                    "schema_version": target_schema_version,
                },
            )
        return self.get(migration_id)

    def get(self, migration_id, include_rows=False):
        row = self.db.one("SELECT * FROM migrations WHERE id=?", (migration_id,))
        if row is None:
            reject("missing", "migration not found")
        report = parse_json(row["report"])
        if not include_rows:
            del report["rows"]
        return {
            "id": migration_id,
            "state": row["state"],
            "report": report,
            "result": parse_json(row["result"]) if row["result"] else None,
        }

    def apply(self, migration_id):
        with self.db.transaction() as con:
            migration = self.get(migration_id, True)
            if migration["state"] == "applied":
                return self.get(migration_id)
            report = migration["report"]
            if not report["ready"]:
                reject("migration", "migration preview contains errors")
            request = report["request"]
            base = self.registry.version(request["dataset"], request["base_version"])
            if base["state"] != "active" or base["digest"] != report["base_digest"]:
                reject("conflict", "migration baseline changed")
            latest = self.registry.list_versions(request["dataset"])[-1]["version"]
            if latest != request["base_version"]:
                reject("conflict", "migration baseline is no longer dataset head")
            content = (
                "\n".join(canonical(row) for row in report["rows"])
                + ("\n" if report["rows"] else "")
            ).encode()
            result = self.ingestion.ingest(
                "migration-" + migration_id,
                request["dataset"],
                request["target_schema_version"],
                content,
                "jsonl",
                base["classification"],
                base["scopes"],
                [{"dataset": request["dataset"], "version": request["base_version"]}],
                {
                    "migration": migration_id,
                    "renames": request["renames"],
                    "defaults": request["defaults"],
                    "drop": request["drop"],
                },
                request["base_version"],
            )
            con.execute(
                "UPDATE migrations SET state='applied',result=? WHERE id=?",
                (canonical(result), migration_id),
            )
            self.db.audit("migration_applied", migration_id, result)
        return self.get(migration_id)

    def pending(self, dataset=None):
        results = []
        for row in self.db.rows(
            "SELECT id,request FROM migrations WHERE state='preview' ORDER BY id"
        ):
            if dataset is None or parse_json(row["request"])["dataset"] == dataset:
                results.append(self.get(row["id"]))
        return results
