from ..validation import name, canonical, digest, parse_json, reject
from ..catalog.registry import CLASSIFICATIONS


class ReleasePlanner:
    def __init__(self, database, registry, artifacts):
        self.db = database
        self.registry = registry
        self.artifacts = artifacts

    def plan(self, release_id, references, scope, clearance):
        name(release_id)
        name(scope)
        if clearance not in CLASSIFICATIONS:
            reject("input", "unknown release clearance")
        selected = self.registry.references(references)
        if not selected:
            reject("input", "release cannot be empty")
        request = {"references": references, "scope": scope, "clearance": clearance}
        with self.db.transaction() as con:
            old = self.db.one("SELECT * FROM releases WHERE id=?", (release_id,))
            if old:
                if old["request"] != canonical(request):
                    reject("conflict", "release identity differs")
                return self.get(release_id)
            closure = self.registry.closure(references)
            for record in closure:
                if record["state"] != "active":
                    reject(
                        "lineage",
                        "release includes withdrawn ancestor",
                        dataset=record["dataset"],
                        version=record["version"],
                    )
                if (
                    scope not in record["scopes"]
                    or CLASSIFICATIONS[record["classification"]]
                    > CLASSIFICATIONS[clearance]
                ):
                    reject(
                        "scope",
                        "release is not authorized for dependency",
                        dataset=record["dataset"],
                        version=record["version"],
                    )
                self.artifacts.verify(record["artifact"])
            schemas = {}
            for record in closure:
                dataset = self.registry.dataset(record["dataset"])
                schema = self.registry.schema(
                    dataset["schema_id"], record["schema_version"]
                )
                schemas[(schema.id, schema.version)] = schema.as_dict()
            manifest = {
                "schema": 1,
                "id": release_id,
                "scope": scope,
                "clearance": clearance,
                "selected": sorted(
                    [
                        {"dataset": item["dataset"], "version": item["version"]}
                        for item in selected
                    ],
                    key=lambda x: (x["dataset"], x["version"]),
                ),
                "versions": [record["payload"] for record in closure],
                "contracts": [schemas[key] for key in sorted(schemas)],
            }
            checksum = digest(manifest)
            con.execute(
                "INSERT INTO releases VALUES(?,?,?,?,'draft',1)",
                (release_id, canonical(request), canonical(manifest), checksum),
            )
            self.db.audit(
                "release_planned",
                release_id,
                {"digest": checksum, "versions": len(closure)},
            )
        return self.get(release_id)

    def get(self, release_id):
        row = self.db.one("SELECT * FROM releases WHERE id=?", (release_id,))
        if row is None:
            reject("missing", "release not found")
        manifest = parse_json(row["manifest"])
        if digest(manifest) != row["digest"]:
            reject("integrity", "release manifest digest mismatch")
        return {
            "id": release_id,
            "state": row["state"],
            "revision": row["revision"],
            "manifest": manifest,
            "digest": row["digest"],
        }

    def eligibility(self, release_id):
        release = self.get(release_id)
        problems = []
        for version in release["manifest"]["versions"]:
            current = self.registry.version(version["dataset"], version["version"])
            if current["state"] != "active":
                problems.append(
                    {
                        "dataset": version["dataset"],
                        "version": version["version"],
                        "code": "withdrawn",
                    }
                )
            if current["payload"] != version:
                problems.append(
                    {
                        "dataset": version["dataset"],
                        "version": version["version"],
                        "code": "version_changed",
                    }
                )
            try:
                self.artifacts.verify(version["artifact"])
            except Exception as exc:
                from ..validation import VaultError

                if not isinstance(exc, VaultError):
                    raise
                problems.append(
                    {
                        "dataset": version["dataset"],
                        "version": version["version"],
                        "code": exc.code,
                    }
                )
        return {"id": release_id, "eligible": not problems, "problems": problems}

    def approve(self, release_id, expected_revision, reviewer, reason):
        name(reviewer)
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1024:
            reject("input", "approval requires bounded reason")
        with self.db.transaction() as con:
            release = self.get(release_id)
            if release["revision"] != expected_revision or release["state"] != "draft":
                reject("conflict", "release state or revision changed")
            eligibility = self.eligibility(release_id)
            if not eligibility["eligible"]:
                reject(
                    "lineage",
                    "release dependencies are no longer eligible",
                    problems=eligibility["problems"],
                )
            owners = {
                self.registry.dataset(item["dataset"])["owner"]
                for item in release["manifest"]["versions"]
            }
            if reviewer in owners:
                reject(
                    "approval", "release reviewer must be independent of dataset owners"
                )
            con.execute(
                "INSERT INTO approvals VALUES(?,?,?,?)",
                (release_id, reviewer, release["digest"], reason),
            )
            con.execute(
                "UPDATE releases SET state='approved',revision=revision+1 WHERE id=?",
                (release_id,),
            )
            self.db.audit(
                "release_approved",
                release_id,
                {"reviewer": reviewer, "digest": release["digest"], "reason": reason},
            )
        return self.get(release_id)

    def revoke(self, release_id, reason):
        if not isinstance(reason, str) or not reason.strip():
            reject("input", "release revocation needs reason")
        with self.db.transaction() as con:
            release = self.get(release_id)
            if release["state"] == "revoked":
                return release
            con.execute(
                "UPDATE releases SET state='revoked',revision=revision+1 WHERE id=?",
                (release_id,),
            )
            self.db.audit("release_revoked", release_id, {"reason": reason})
        return self.get(release_id)

    def compare(self, left_id, right_id):
        left = self.get(left_id)["manifest"]
        right = self.get(right_id)["manifest"]
        before = {(item["dataset"], item["version"]): item for item in left["versions"]}
        after = {(item["dataset"], item["version"]): item for item in right["versions"]}
        return {
            "left": left_id,
            "right": right_id,
            "added": [
                {"dataset": key[0], "version": key[1]}
                for key in sorted(after.keys() - before.keys())
            ],
            "removed": [
                {"dataset": key[0], "version": key[1]}
                for key in sorted(before.keys() - after.keys())
            ],
            "scope_changed": left["scope"] != right["scope"],
            "clearance_changed": left["clearance"] != right["clearance"],
        }
