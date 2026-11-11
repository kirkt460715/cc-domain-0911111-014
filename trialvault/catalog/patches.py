from ..validation import (
    name,
    object_value,
    list_value,
    canonical,
    digest,
    parse_json,
    reject,
)


class VersionPatches:
    def __init__(self, database, registry, ingestion):
        self.db = database
        self.registry = registry
        self.ingestion = ingestion
        with database.transaction() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS patch_plans(id TEXT PRIMARY KEY,request TEXT NOT NULL,
                preview TEXT NOT NULL,state TEXT NOT NULL,result TEXT)""")

    def plan(self, patch_id, dataset, base_version, changes):
        name(patch_id)
        base = self.registry.version(dataset, base_version)
        if base["state"] != "active":
            reject("state", "cannot patch a withdrawn version")
        schema = self.registry.schema(
            self.registry.dataset(dataset)["schema_id"], base["schema_version"]
        )
        records = {
            schema.row_key(row): row
            for row in self.ingestion.rows(dataset, base_version)
        }
        original = dict(records)
        touched = set()
        normalized = []
        for change in list_value(changes, 1, 100000):
            object_value(
                change,
                {"operation", "key", "row", "expected_digest"},
                {"operation", "key"},
            )
            operation = change["operation"]
            if operation not in {"insert", "replace", "delete"}:
                reject("patch", "unknown row patch operation")
            key_values = list_value(
                change["key"], len(schema.primary_key), len(schema.primary_key)
            )
            encoded = canonical(key_values)
            if encoded in touched:
                reject("patch", "row key touched multiple times")
            touched.add(encoded)
            current = records.get(encoded)
            if operation == "insert" and current is not None:
                reject("patch", "insert key already exists")
            if operation in {"replace", "delete"} and current is None:
                reject("patch", "patch key does not exist")
            expected = change.get("expected_digest")
            if expected is not None and (
                current is None or digest(current) != expected
            ):
                reject("conflict", "row digest precondition failed", key=key_values)
            if operation == "delete":
                if "row" in change:
                    reject("patch", "delete cannot include replacement row")
                del records[encoded]
                normalized.append(
                    {
                        "operation": operation,
                        "key": key_values,
                        "before_digest": digest(current),
                    }
                )
            else:
                if "row" not in change:
                    reject("patch", "insert and replace require row")
                row = schema.validate(change["row"])
                if schema.row_key(row) != encoded:
                    reject("patch", "replacement row changes key")
                records[encoded] = row
                normalized.append(
                    {
                        "operation": operation,
                        "key": key_values,
                        "before_digest": digest(current) if current else None,
                        "row": row,
                    }
                )
        ordered = [records[key] for key in sorted(records)]
        request = {"dataset": dataset, "base_version": base_version, "changes": changes}
        preview = {
            "id": patch_id,
            "dataset": dataset,
            "base_version": base_version,
            "base_digest": base["digest"],
            "schema_version": base["schema_version"],
            "classification": base["classification"],
            "scopes": base["scopes"],
            "changes": normalized,
            "before_rows": len(original),
            "after_rows": len(ordered),
            "result_rows": ordered,
            "result_digest": digest(ordered),
        }
        with self.db.transaction() as con:
            old = self.db.one("SELECT * FROM patch_plans WHERE id=?", (patch_id,))
            if old:
                if old["request"] != canonical(request):
                    reject("conflict", "patch plan identity differs")
                return self.get(patch_id)
            con.execute(
                "INSERT INTO patch_plans VALUES(?,?,?,'planned',NULL)",
                (patch_id, canonical(request), canonical(preview)),
            )
            self.db.audit(
                "patch_planned",
                patch_id,
                {
                    "dataset": dataset,
                    "base_version": base_version,
                    "changes": len(changes),
                },
            )
        return self.get(patch_id)

    def get(self, patch_id, include_rows=False):
        row = self.db.one("SELECT * FROM patch_plans WHERE id=?", (patch_id,))
        if row is None:
            reject("missing", "patch plan not found")
        preview = parse_json(row["preview"])
        if not include_rows:
            del preview["result_rows"]
        return {
            "id": patch_id,
            "state": row["state"],
            "preview": preview,
            "result": parse_json(row["result"]) if row["result"] else None,
        }

    def apply(self, patch_id):
        with self.db.transaction() as con:
            plan = self.get(patch_id, True)
            if plan["state"] == "applied":
                return self.get(patch_id)
            if plan["state"] != "planned":
                reject("state", "patch plan is cancelled")
            preview = plan["preview"]
            base = self.registry.version(preview["dataset"], preview["base_version"])
            if base["digest"] != preview["base_digest"] or base["state"] != "active":
                reject("conflict", "patch baseline changed or withdrawn")
            latest = self.registry.list_versions(preview["dataset"])[-1]["version"]
            if latest != preview["base_version"]:
                reject("conflict", "patch requires base version to remain dataset head")
            content = (
                "\n".join(canonical(row) for row in preview["result_rows"])
                + ("\n" if preview["result_rows"] else "")
            ).encode()
            result = self.ingestion.ingest(
                "patch-" + patch_id,
                preview["dataset"],
                preview["schema_version"],
                content,
                "jsonl",
                preview["classification"],
                preview["scopes"],
                [{"dataset": preview["dataset"], "version": preview["base_version"]}],
                {"patch": patch_id, "changes": preview["changes"]},
                preview["base_version"],
            )
            con.execute(
                "UPDATE patch_plans SET state='applied',result=? WHERE id=?",
                (canonical(result), patch_id),
            )
            self.db.audit("patch_applied", patch_id, result)
        return self.get(patch_id)

    def cancel(self, patch_id, reason):
        if not isinstance(reason, str) or not reason.strip():
            reject("input", "patch cancellation requires reason")
        with self.db.transaction() as con:
            plan = self.get(patch_id)
            if plan["state"] != "planned":
                reject("state", "patch cannot be cancelled after application")
            con.execute(
                "UPDATE patch_plans SET state='cancelled',result=? WHERE id=?",
                (canonical({"reason": reason}), patch_id),
            )
            self.db.audit("patch_cancelled", patch_id, {"reason": reason})
        return self.get(patch_id)
