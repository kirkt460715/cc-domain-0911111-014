from ..validation import (
    list_value,
    canonical,
    digest,
    parse_json,
    reject,
)
from ..contracts.profile import profile_rows, compare_rows


class DatasetChecks:
    def __init__(self, database, registry, ingestion, artifacts):
        self.db = database
        self.registry = registry
        self.ingestion = ingestion
        self.artifacts = artifacts

    def profile(self, dataset, version):
        record = self.registry.version(dataset, version)
        schema = self.registry.schema(
            self.registry.dataset(dataset)["schema_id"], record["schema_version"]
        )
        return {
            "dataset": dataset,
            "version": version,
            **profile_rows(schema, self.ingestion.rows(dataset, version)),
        }

    def compare(self, dataset, before, after, limit=1000):
        old = self.registry.version(dataset, before)
        new = self.registry.version(dataset, after)
        old_schema = self.registry.schema(
            self.registry.dataset(dataset)["schema_id"], old["schema_version"]
        )
        new_schema = self.registry.schema(
            self.registry.dataset(dataset)["schema_id"], new["schema_version"]
        )
        if old_schema.primary_key != new_schema.primary_key:
            reject("schema", "row comparison requires stable primary key")
        return compare_rows(
            new_schema,
            self.ingestion.rows(dataset, before),
            self.ingestion.rows(dataset, after),
            limit,
        )

    def foreign_key(
        self,
        child,
        child_version,
        parent,
        parent_version,
        child_fields,
        parent_fields,
        nullable=False,
    ):
        if not isinstance(nullable, bool):
            reject("input", "nullable must be boolean")
        list_value(child_fields, 1, 16)
        list_value(parent_fields, 1, 16)
        if len(child_fields) != len(parent_fields):
            reject("input", "foreign key arity differs")
        parent_keys = set()
        for row in self.ingestion.rows(parent, parent_version):
            if any(field not in row for field in parent_fields):
                reject("schema", "parent key field absent")
            values = [row[field] for field in parent_fields]
            if any(value is None for value in values):
                reject("schema", "parent key contains null")
            key = canonical(values)
            if key in parent_keys:
                reject("schema", "parent reference key is not unique")
            parent_keys.add(key)
        missing = []
        checked = 0
        for index, row in enumerate(self.ingestion.rows(child, child_version), 1):
            values = [row.get(field) for field in child_fields]
            if any(value is None for value in values):
                if nullable and all(value is None for value in values):
                    continue
                missing.append(
                    {
                        "row": index,
                        "key": values,
                        "reason": "partial_or_disallowed_null",
                    }
                )
                continue
            checked += 1
            if canonical(values) not in parent_keys:
                missing.append(
                    {"row": index, "key": values, "reason": "missing_parent"}
                )
        return {"valid": not missing, "checked": checked, "missing": missing}

    def inspect(self):
        sqlite = self.db.one("PRAGMA integrity_check")
        if list(sqlite.values()) != ["ok"]:
            reject("integrity", "SQLite check failed")
        audit = self.db.audit_check()
        version_count = 0
        for record in self.db.rows(
            "SELECT dataset,version FROM versions ORDER BY dataset,version"
        ):
            version = self.registry.version(record["dataset"], record["version"])
            if digest(version["payload"]) != version["digest"]:
                reject("integrity", "version metadata digest mismatch")
            dataset = self.registry.dataset(version["dataset"])
            schema = self.registry.schema(
                dataset["schema_id"], version["schema_version"]
            )
            rows = self.ingestion.rows(version["dataset"], version["version"])
            if len(rows) != version["row_count"]:
                reject("integrity", "version row count mismatch")
            keys = set()
            for row in rows:
                if schema.validate(row) != row:
                    reject("integrity", "stored row is not canonical")
                key = schema.row_key(row)
                if key in keys:
                    reject("integrity", "duplicate stored primary key")
                keys.add(key)
            actual = [
                {"dataset": row["parent"], "version": row["parent_version"]}
                for row in self.db.rows(
                    "SELECT parent,parent_version FROM lineage WHERE dataset=? AND version=? ORDER BY parent,parent_version",
                    (version["dataset"], version["version"]),
                )
            ]
            if actual != version["payload"]["parents"]:
                reject("integrity", "lineage metadata differs")
            version_count += 1
        experiment_runs = 0
        if self.db.one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='experiment_runs'"
        ):
            for row in self.db.rows(
                "SELECT experiment,result,digest,model_artifact FROM experiment_runs ORDER BY experiment"
            ):
                if digest(parse_json(row["result"])) != row["digest"]:
                    reject("integrity", "experiment run digest mismatch")
                self.artifacts.verify(row["model_artifact"])
                experiment_runs += 1
        return {
            "sqlite": "ok",
            "audit": audit,
            "versions": version_count,
            "experiment_runs": experiment_runs,
            "artifacts": len(self.artifacts.inventory()),
        }

    def unused_artifacts(self, dry_run=True):
        referenced = {
            row["artifact"] for row in self.db.rows("SELECT artifact FROM versions")
        }
        for row in self.db.rows("SELECT manifest FROM releases"):
            referenced.update(
                item["artifact"] for item in parse_json(row["manifest"])["versions"]
            )
        if self.db.one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='experiment_runs'"
        ):
            for row in self.db.rows("SELECT model_artifact FROM experiment_runs"):
                referenced.add(row["model_artifact"])
        with self.db.transaction():
            return self.artifacts.remove_unreferenced(referenced, dry_run)

    def withdrawal_impact(self, dataset, version):
        selected = {(dataset, version)} | {
            (item["dataset"], item["version"])
            for item in self.registry.descendants(dataset, version)
        }
        releases = []
        for row in self.db.rows("SELECT id,state,manifest FROM releases ORDER BY id"):
            references = {
                (item["dataset"], item["version"])
                for item in parse_json(row["manifest"])["versions"]
            }
            overlap = selected & references
            if overlap:
                releases.append(
                    {
                        "id": row["id"],
                        "state": row["state"],
                        "affected": [
                            {"dataset": item[0], "version": item[1]}
                            for item in sorted(overlap)
                        ],
                    }
                )
        return {
            "dataset": dataset,
            "version": version,
            "descendants": [
                {"dataset": item[0], "version": item[1]} for item in sorted(selected)
            ],
            "releases": releases,
        }
