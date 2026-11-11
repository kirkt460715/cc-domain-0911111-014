from ..validation import name, integer, canonical, parse_json, digest, reject


class RetentionReview:
    def __init__(self, database, registry, planner):
        self.db = database
        self.registry = registry
        self.planner = planner
        with database.transaction() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS retention_reviews(id TEXT PRIMARY KEY,payload TEXT NOT NULL,digest TEXT NOT NULL)"
            )

    def review(self, review_id, keep_latest=1):
        name(review_id)
        integer(keep_latest, 1, 100000)
        protected = {}
        for row in self.db.rows("SELECT id,manifest,state FROM releases ORDER BY id"):
            if row["state"] == "revoked":
                continue
            manifest = parse_json(row["manifest"])
            for item in manifest["versions"]:
                protected.setdefault((item["dataset"], item["version"]), []).append(
                    "release:" + row["id"]
                )
        for row in self.db.rows(
            "SELECT parent,parent_version,dataset,version FROM lineage ORDER BY parent,parent_version"
        ):
            child = self.registry.version(row["dataset"], row["version"])
            if child["state"] == "active":
                protected.setdefault((row["parent"], row["parent_version"]), []).append(
                    "child:" + row["dataset"] + ":" + str(row["version"])
                )
        candidates = []
        retained = []
        for dataset in self.db.rows("SELECT id FROM datasets ORDER BY id"):
            versions = self.registry.list_versions(dataset["id"])
            latest = {item["version"] for item in versions[-keep_latest:]}
            for record in versions:
                reference = (record["dataset"], record["version"])
                reasons = protected.get(reference, [])[:]
                if record["version"] in latest:
                    reasons.append("latest_retained")
                item = {
                    "dataset": record["dataset"],
                    "version": record["version"],
                    "state": record["state"],
                    "artifact": record["artifact"],
                    "reasons": sorted(reasons),
                }
                if record["state"] == "withdrawn" and not reasons:
                    candidates.append(item)
                else:
                    retained.append(item)
        payload = {
            "id": review_id,
            "keep_latest": keep_latest,
            "candidates": candidates,
            "retained": retained,
            "action": "review_only_no_version_deletion",
            "version_fingerprint": digest(
                self.db.rows(
                    "SELECT dataset,version,state,digest FROM versions ORDER BY dataset,version"
                )
            ),
        }
        with self.db.transaction() as con:
            old = self.db.one(
                "SELECT digest FROM retention_reviews WHERE id=?", (review_id,)
            )
            if old:
                if old["digest"] != digest(payload):
                    reject("conflict", "retention review identity differs")
                return payload
            con.execute(
                "INSERT INTO retention_reviews VALUES(?,?,?)",
                (review_id, canonical(payload), digest(payload)),
            )
            self.db.audit(
                "retention_review",
                review_id,
                {"candidates": len(candidates), "retained": len(retained)},
            )
        return payload

    def verify(self, review_id):
        row = self.db.one("SELECT * FROM retention_reviews WHERE id=?", (review_id,))
        if row is None:
            reject("missing", "retention review not found")
        payload = parse_json(row["payload"])
        if digest(payload) != row["digest"]:
            reject("integrity", "retention review corrupted")
        current = digest(
            self.db.rows(
                "SELECT dataset,version,state,digest FROM versions ORDER BY dataset,version"
            )
        )
        return {
            "id": review_id,
            "current": current == payload["version_fingerprint"],
            "candidates": payload["candidates"],
        }
