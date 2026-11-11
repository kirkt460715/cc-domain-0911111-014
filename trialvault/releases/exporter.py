import os
import hashlib
from pathlib import Path
from ..validation import name, canonical, digest, parse_json, reject, integer
from ..contracts.schema import Schema


class Exporter:
    def __init__(self, database, planner, artifacts):
        self.db = database
        self.planner = planner
        self.artifacts = artifacts

    def export(self, export_id, release_id, target, maximum_members=None):
        name(export_id)
        target = Path(target).absolute()
        if (
            not target.parent.is_dir()
            or target.is_symlink()
            or target.parent.is_symlink()
        ):
            reject("path", "export target requires safe existing parent")
        if maximum_members is not None:
            integer(maximum_members, 1, 10000)
        stage = target.parent / (".vault-export-" + export_id)
        with self.db.transaction() as con:
            release = self.planner.get(release_id)
            if release["state"] != "approved":
                reject("state", "export requires approved release")
            if not self.planner.eligibility(release_id)["eligible"]:
                reject(
                    "lineage", "export blocked by withdrawn or corrupted dependencies"
                )
            old = self.db.one("SELECT * FROM exports WHERE id=?", (export_id,))
            if old and (old["release"] != release_id or old["target"] != str(target)):
                reject("conflict", "export identity differs")
            if old and old["state"] == "complete":
                verified = verify_directory(target)
                if verified["digest"] != release["digest"]:
                    reject("integrity", "completed export differs from release")
                return {
                    "id": export_id,
                    "state": "complete",
                    "reused": True,
                    **verified,
                }
            if target.exists():
                if old:
                    verified = verify_directory(target)
                    if verified["digest"] == release["digest"]:
                        con.execute(
                            "UPDATE exports SET state='complete' WHERE id=?",
                            (export_id,),
                        )
                        return {
                            "id": export_id,
                            "state": "complete",
                            "reused": True,
                            **verified,
                        }
                reject("conflict", "export target already exists")
            if stage.is_symlink():
                reject("path", "export staging path cannot be a link")
            if not old:
                if stage.exists():
                    reject("conflict", "unowned staging path exists")
                stage.mkdir()
                con.execute(
                    "INSERT INTO exports VALUES(?,?,?,'writing',?)",
                    (
                        export_id,
                        release_id,
                        str(target),
                        canonical({"stage": str(stage), "digest": release["digest"]}),
                    ),
                )
                self.db.audit(
                    "export_started",
                    export_id,
                    {"release": release_id, "digest": release["digest"]},
                )
            elif not stage.is_dir():
                reject("recovery", "export staging directory missing")
        checksums = sorted(
            {item["artifact"] for item in release["manifest"]["versions"]}
        )
        written = 0
        for checksum in checksums:
            path = stage / (checksum + ".jsonl")
            if path.is_symlink():
                reject("path", "staging artifact cannot be a link")
            content = self.artifacts.read(checksum)
            if path.exists():
                if path.read_bytes() != content:
                    reject("integrity", "staged artifact differs from frozen content")
                continue
            if maximum_members is not None and written >= maximum_members:
                return {
                    "id": export_id,
                    "state": "writing",
                    "members_written": sum(
                        (stage / (value + ".jsonl")).exists() for value in checksums
                    ),
                }
            with path.open("xb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            written += 1
        manifest = {"digest": release["digest"], "manifest": release["manifest"]}
        manifest_path = stage / "manifest.json"
        expected = (canonical(manifest) + "\n").encode()
        if manifest_path.exists():
            if manifest_path.is_symlink() or manifest_path.read_bytes() != expected:
                reject("integrity", "staging manifest changed")
        else:
            with manifest_path.open("xb") as output:
                output.write(expected)
                output.flush()
                os.fsync(output.fileno())
        verified = verify_directory(stage)
        with self.db.transaction() as con:
            if (
                self.planner.get(release_id)["state"] != "approved"
                or not self.planner.eligibility(release_id)["eligible"]
            ):
                reject("state", "release changed before export commit")
            if target.exists():
                reject("conflict", "export destination appeared concurrently")
            os.rename(stage, target)
            parent_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
            con.execute("UPDATE exports SET state='complete' WHERE id=?", (export_id,))
            self.db.audit(
                "export_completed",
                export_id,
                {"release": release_id, "digest": release["digest"]},
            )
        return {"id": export_id, "state": "complete", "reused": False, **verified}

    def pending(self):
        return self.db.rows(
            "SELECT id,release,target,state FROM exports WHERE state!='complete' ORDER BY id"
        )


def verify_directory(path):
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        reject("path", "release must be a regular directory")
    manifest_path = root / "manifest.json"
    if (
        manifest_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.stat().st_size > 16000000
    ):
        reject("integrity", "manifest absent or invalid")
    envelope = parse_json(manifest_path.read_bytes())
    if not isinstance(envelope, dict) or set(envelope) != {"digest", "manifest"}:
        reject("input", "invalid release envelope")
    manifest = envelope["manifest"]
    if digest(manifest) != envelope["digest"] or manifest.get("schema") != 1:
        reject("integrity", "release manifest fingerprint or schema invalid")
    contracts = {
        (item["id"], item["version"]): Schema(item) for item in manifest["contracts"]
    }
    required = {"manifest.json"} | {
        item["artifact"] + ".jsonl" for item in manifest["versions"]
    }
    actual = {item.name for item in root.iterdir()}
    if actual != required:
        reject(
            "integrity",
            "release member set differs",
            missing=sorted(required - actual),
            extra=sorted(actual - required),
        )
    versions = {
        (item["dataset"], item["version"]): item for item in manifest["versions"]
    }
    if len(versions) != len(manifest["versions"]):
        reject("integrity", "duplicate dataset version in release")
    total = 0
    for item in manifest["versions"]:
        artifact = root / (item["artifact"] + ".jsonl")
        if (
            artifact.is_symlink()
            or not artifact.is_file()
            or artifact.stat().st_size > 64000000
        ):
            reject("path", "invalid release artifact")
        content = artifact.read_bytes()
        if hashlib.sha256(content).hexdigest() != item["artifact"]:
            reject("integrity", "release artifact hash differs")
        rows = [
            parse_json(line) for line in content.decode("utf-8").splitlines() if line
        ]
        if len(rows) != item["rows"]:
            reject("integrity", "release row count differs")
        schema_key = (item["schema_id"], item["schema_version"])
        if schema_key not in contracts:
            reject("integrity", "release contract is missing")
        contract = contracts[schema_key]
        row_keys = set()
        for row in rows:
            if contract.validate(row) != row:
                reject("integrity", "release row is not canonical")
            primary = contract.row_key(row)
            if primary in row_keys:
                reject("integrity", "release primary key duplicate")
            row_keys.add(primary)
        for parent in item["parents"]:
            if (parent["dataset"], parent["version"]) not in versions:
                reject("integrity", "release omits lineage ancestor")
        total += len(rows)
    for selected in manifest["selected"]:
        if (selected["dataset"], selected["version"]) not in versions:
            reject("integrity", "selected version missing")
    return {
        "release": manifest["id"],
        "digest": envelope["digest"],
        "versions": len(versions),
        "rows": total,
        "artifacts": len(required) - 1,
    }
