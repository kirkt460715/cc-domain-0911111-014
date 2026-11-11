import hashlib
from ..validation import name, integer, canonical, reject


class CatalogQueries:
    def __init__(self, database, registry, ingestion):
        self.db = database
        self.registry = registry
        self.ingestion = ingestion

    def search(
        self,
        text="",
        owner=None,
        classification=None,
        scope=None,
        include_withdrawn=False,
        after=None,
        limit=100,
    ):
        integer(limit, 1, 1000)
        if not isinstance(text, str) or len(text) > 512:
            reject("input", "catalog search text is invalid")
        if not isinstance(include_withdrawn, bool):
            reject("input", "include_withdrawn must be boolean")
        result = []
        for dataset in self.db.rows("SELECT * FROM datasets ORDER BY id"):
            if after is not None and dataset["id"] <= after:
                continue
            if owner is not None and dataset["owner"] != owner:
                continue
            if (
                text.casefold()
                not in (dataset["id"] + " " + dataset["title"]).casefold()
            ):
                continue
            versions = self.registry.list_versions(dataset["id"])
            eligible = []
            for version in versions:
                if not include_withdrawn and version["state"] != "active":
                    continue
                if (
                    classification is not None
                    and version["classification"] != classification
                ):
                    continue
                if scope is not None and scope not in version["scopes"]:
                    continue
                eligible.append(version)
            if not eligible:
                continue
            latest = eligible[-1]
            result.append(
                {
                    **dataset,
                    "version": latest["version"],
                    "rows": latest["row_count"],
                    "state": latest["state"],
                    "classification": latest["classification"],
                    "scopes": latest["scopes"],
                }
            )
            if len(result) > limit:
                break
        has_more = len(result) > limit
        result = result[:limit]
        return {"items": result, "next": result[-1]["id"] if has_more else None}

    def sample(self, dataset, version, count=10, seed="sample"):
        integer(count, 1, 10000)
        if not isinstance(seed, str) or len(seed) > 256:
            reject("input", "sample seed must be bounded text")
        record = self.registry.version(dataset, version)
        schema = self.registry.schema(
            self.registry.dataset(dataset)["schema_id"], record["schema_version"]
        )
        rows = self.ingestion.rows(dataset, version)
        ranked = sorted(
            rows,
            key=lambda row: (
                hashlib.sha256((seed + schema.row_key(row)).encode()).hexdigest(),
                schema.row_key(row),
            ),
        )
        return {
            "dataset": dataset,
            "version": version,
            "seed": seed,
            "total": len(rows),
            "rows": ranked[:count],
            "method": "deterministic key hash ordering",
        }

    def lookup(self, dataset, version, key_values):
        record = self.registry.version(dataset, version)
        schema = self.registry.schema(
            self.registry.dataset(dataset)["schema_id"], record["schema_version"]
        )
        if not isinstance(key_values, list) or len(key_values) != len(
            schema.primary_key
        ):
            reject("input", "lookup key arity differs")
        key = canonical(key_values)
        for row in self.ingestion.rows(dataset, version):
            if schema.row_key(row) == key:
                return {
                    "found": True,
                    "row": row,
                    "dataset": dataset,
                    "version": version,
                }
        return {"found": False, "row": None, "dataset": dataset, "version": version}

    def storage_usage(self):
        references = {}
        logical = 0
        for row in self.db.rows(
            "SELECT dataset,version,artifact,row_count FROM versions ORDER BY dataset,version"
        ):
            logical += row["row_count"]
            references.setdefault(row["artifact"], []).append(
                {"dataset": row["dataset"], "version": row["version"]}
            )
        artifacts = []
        for checksum, versions in sorted(references.items()):
            size = self.registry.artifacts.verify(checksum)["bytes"]
            artifacts.append(
                {"sha256": checksum, "bytes": size, "references": versions}
            )
        physical = sum(item["bytes"] for item in artifacts)
        logical_bytes = sum(
            item["bytes"] * len(item["references"]) for item in artifacts
        )
        return {
            "logical_rows": logical,
            "physical_bytes": physical,
            "logical_bytes": logical_bytes,
            "deduplicated_bytes": logical_bytes - physical,
            "artifacts": artifacts,
        }
