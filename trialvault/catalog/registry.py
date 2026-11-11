from ..validation import (
    name,
    integer,
    object_value,
    list_value,
    canonical,
    digest,
    parse_json,
    reject,
)
from ..contracts.schema import Schema
from ..contracts.evolution import compatibility

CLASSIFICATIONS = {"public": 0, "internal": 1, "restricted": 2}


class Registry:
    def __init__(self, database, artifacts):
        self.db = database
        self.artifacts = artifacts

    def add_schema(self, data, allow_breaking=False):
        schema = Schema(data)
        if not isinstance(allow_breaking, bool):
            reject("input", "allow_breaking must be boolean")
        with self.db.transaction() as con:
            old = self.db.one(
                "SELECT digest FROM schemas WHERE id=? AND version=?",
                (schema.id, schema.version),
            )
            if old:
                if old["digest"] != schema.fingerprint:
                    reject("conflict", "schema version is immutable")
                return {"id": schema.id, "version": schema.version, "existing": True}
            latest = self.db.one(
                "SELECT version,payload FROM schemas WHERE id=? ORDER BY version DESC LIMIT 1",
                (schema.id,),
            )
            if latest:
                if schema.version != latest["version"] + 1:
                    reject("schema", "schema versions must be consecutive")
                change = compatibility(Schema(parse_json(latest["payload"])), schema)
                if not change["compatible"] and not allow_breaking:
                    reject(
                        "schema",
                        "breaking schema evolution requires explicit declaration",
                        change=change,
                    )
            elif schema.version != 1:
                reject("schema", "first schema version must be one")
            else:
                change = None
            con.execute(
                "INSERT INTO schemas VALUES(?,?,?,?)",
                (
                    schema.id,
                    schema.version,
                    canonical(schema.as_dict()),
                    schema.fingerprint,
                ),
            )
            self.db.audit(
                "schema_added", schema.id, {"version": schema.version, "change": change}
            )
        return {
            "id": schema.id,
            "version": schema.version,
            "existing": False,
            "change": change,
        }

    def schema(self, schema_id, version):
        row = self.db.one(
            "SELECT payload FROM schemas WHERE id=? AND version=?", (schema_id, version)
        )
        if row is None:
            reject("missing", "schema version not found")
        return Schema(parse_json(row["payload"]))

    def dataset(self, dataset_id):
        row = self.db.one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
        if row is None:
            reject("missing", "dataset not found")
        return row

    def register(self, dataset_id, title, schema_id, owner):
        name(dataset_id)
        name(schema_id)
        name(owner)
        if not isinstance(title, str) or not title.strip() or len(title) > 512:
            reject("input", "dataset title must be bounded text")
        self.schema(schema_id, 1)
        payload = {
            "id": dataset_id,
            "title": title.strip(),
            "schema_id": schema_id,
            "owner": owner,
        }
        with self.db.transaction() as con:
            old = self.db.one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
            if old:
                if old != payload:
                    reject("conflict", "dataset registration is immutable")
                return old
            con.execute(
                "INSERT INTO datasets VALUES(?,?,?,?)",
                (dataset_id, title.strip(), schema_id, owner),
            )
            self.db.audit("dataset_registered", dataset_id, payload)
        return payload

    def version(self, dataset_id, version):
        row = self.db.one(
            "SELECT * FROM versions WHERE dataset=? AND version=?",
            (dataset_id, version),
        )
        if row is None:
            reject("missing", "dataset version not found")
        row["scopes"] = parse_json(row["scopes"])
        row["payload"] = parse_json(row["payload"])
        return row

    def references(self, references):
        result = []
        seen = set()
        for reference in list_value(references, 0, 1000):
            object_value(reference, {"dataset", "version"}, {"dataset", "version"})
            dataset = name(reference["dataset"])
            version = integer(reference["version"], 1)
            if (dataset, version) in seen:
                reject("input", "duplicate dataset version reference")
            seen.add((dataset, version))
            result.append(self.version(dataset, version))
        return result

    def add_version(
        self,
        dataset_id,
        schema_version,
        artifact,
        row_count,
        classification,
        scopes,
        parents,
        metadata,
        expected_version,
    ):
        dataset = self.dataset(dataset_id)
        self.schema(dataset["schema_id"], schema_version)
        integer(row_count)
        if classification not in CLASSIFICATIONS:
            reject("input", "unknown data classification")
        scope_set = sorted({name(scope) for scope in list_value(scopes, 1, 100)})
        object_value(metadata)
        if len(canonical(metadata).encode()) > 16000:
            reject("input", "version metadata exceeds size limit")
        self.artifacts.verify(artifact)
        with self.db.transaction() as con:
            latest = (
                self.db.one(
                    "SELECT MAX(version) AS n FROM versions WHERE dataset=?",
                    (dataset_id,),
                )["n"]
                or 0
            )
            if latest != expected_version:
                reject("conflict", "dataset head changed", actual=latest)
            parent_versions = self.references(parents)
            for parent in parent_versions:
                if parent["state"] != "active":
                    reject("lineage", "cannot derive from withdrawn parent")
                if (
                    CLASSIFICATIONS[classification]
                    < CLASSIFICATIONS[parent["classification"]]
                ):
                    reject("scope", "derived version cannot lower classification")
                if not set(scope_set).issubset(parent["scopes"]):
                    reject("scope", "derived usage scope exceeds parent")
            version = latest + 1
            parent_refs = sorted(
                [
                    {"dataset": item["dataset"], "version": item["version"]}
                    for item in parent_versions
                ],
                key=lambda x: (x["dataset"], x["version"]),
            )
            payload = {
                "dataset": dataset_id,
                "version": version,
                "schema_version": schema_version,
                "schema_id": dataset["schema_id"],
                "artifact": artifact,
                "rows": row_count,
                "classification": classification,
                "scopes": scope_set,
                "parents": parent_refs,
                "metadata": metadata,
            }
            con.execute(
                "INSERT INTO versions VALUES(?,?,?,?,?,?,?,'active',?,?)",
                (
                    dataset_id,
                    version,
                    schema_version,
                    artifact,
                    row_count,
                    classification,
                    canonical(scope_set),
                    canonical(payload),
                    digest(payload),
                ),
            )
            for parent in parent_refs:
                con.execute(
                    "INSERT INTO lineage VALUES(?,?,?,?)",
                    (dataset_id, version, parent["dataset"], parent["version"]),
                )
            self.db.audit(
                "version_added",
                dataset_id,
                {"version": version, "digest": digest(payload)},
            )
        return self.version(dataset_id, version)

    def withdraw(self, dataset_id, version, reason):
        if not isinstance(reason, str) or not reason.strip():
            reject("input", "withdrawal requires reason")
        with self.db.transaction() as con:
            row = self.version(dataset_id, version)
            if row["state"] == "withdrawn":
                return {
                    "changed": False,
                    "affected": self.descendants(dataset_id, version),
                }
            con.execute(
                "UPDATE versions SET state='withdrawn' WHERE dataset=? AND version=?",
                (dataset_id, version),
            )
            affected = self.descendants(dataset_id, version)
            self.db.audit(
                "version_withdrawn",
                dataset_id,
                {"version": version, "reason": reason, "affected": affected},
            )
        return {"changed": True, "affected": affected}

    def closure(self, references):
        pending = [
            (item["dataset"], item["version"]) for item in self.references(references)
        ]
        visited = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            for row in self.db.rows(
                "SELECT parent,parent_version FROM lineage WHERE dataset=? AND version=?",
                current,
            ):
                pending.append((row["parent"], row["parent_version"]))
        return [self.version(dataset, version) for dataset, version in sorted(visited)]

    def descendants(self, dataset_id, version):
        self.version(dataset_id, version)
        visited = set()
        pending = [(dataset_id, version)]
        while pending:
            current = pending.pop()
            for row in self.db.rows(
                "SELECT dataset,version FROM lineage WHERE parent=? AND parent_version=?",
                current,
            ):
                child = (row["dataset"], row["version"])
                if child not in visited:
                    visited.add(child)
                    pending.append(child)
        return [
            {"dataset": dataset, "version": version}
            for dataset, version in sorted(visited)
        ]

    def list_versions(self, dataset_id):
        self.dataset(dataset_id)
        return [
            self.version(dataset_id, row["version"])
            for row in self.db.rows(
                "SELECT version FROM versions WHERE dataset=? ORDER BY version",
                (dataset_id,),
            )
        ]
