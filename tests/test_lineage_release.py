import unittest
import tempfile
import json
from pathlib import Path
from trialvault.service import Vault
from trialvault.validation import VaultError
from trialvault.releases.exporter import verify_directory
from support import register, ingest, release


class LineageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.vault = Vault(Path(self.temp.name) / "vault")
        self.addCleanup(self.vault.close)
        register(self.vault)
        register(self.vault, "derived")
        ingest(self.vault)

    def test_scope_and_classification_cannot_expand(self):
        parent = [{"dataset": "data", "version": 1}]
        with self.assertRaises(VaultError) as caught:
            ingest(
                self.vault, "child", "derived", classification="public", parents=parent
            )
        self.assertEqual(caught.exception.code, "scope")
        with self.assertRaises(VaultError):
            ingest(
                self.vault, "child", "derived", scopes=["commercial"], parents=parent
            )
        self.assertEqual(self.vault.registry.list_versions("derived"), [])
        result = ingest(
            self.vault, "child", "derived", classification="restricted", parents=parent
        )
        self.assertEqual(result["version"], 1)
        closure = self.vault.registry.closure([{"dataset": "derived", "version": 1}])
        self.assertEqual({item["dataset"] for item in closure}, {"data", "derived"})
        self.assertEqual(
            self.vault.registry.descendants("data", 1),
            [{"dataset": "derived", "version": 1}],
        )

    def test_withdrawal_blocks_derivation_and_publication(self):
        ingest(
            self.vault, "child", "derived", parents=[{"dataset": "data", "version": 1}]
        )
        self.vault.releases.plan(
            "planned", [{"dataset": "derived", "version": 1}], "research", "internal"
        )
        withdrawn = self.vault.registry.withdraw("data", 1, "Synthetic source error")
        self.assertTrue(withdrawn["changed"])
        self.assertEqual(withdrawn["affected"], [{"dataset": "derived", "version": 1}])
        self.assertFalse(self.vault.registry.withdraw("data", 1, "Repeated")["changed"])
        self.assertFalse(self.vault.releases.eligibility("planned")["eligible"])
        with self.assertRaises(VaultError):
            self.vault.releases.approve("planned", 1, "reviewer", "Cannot approve")
        with self.assertRaises(VaultError):
            self.vault.releases.plan(
                "new", [{"dataset": "derived", "version": 1}], "research", "internal"
            )
        impact = self.vault.checks.withdrawal_impact("data", 1)
        self.assertEqual(impact["releases"][0]["id"], "planned")
        self.assertEqual(len(impact["descendants"]), 2)

    def test_projection_preserves_lineage(self):
        result = self.vault.derivations.project(
            "project",
            "data",
            1,
            "derived",
            1,
            {"id": "id", "value": "value", "group": "group"},
            [{"field": "value", "operator": "gt", "value": "10"}],
            "internal",
            ["research"],
        )
        self.assertEqual(result["rows"], 1)
        self.assertEqual(self.vault.ingestion.rows("derived", 1)[0]["id"], "two")
        record = self.vault.registry.version("derived", 1)
        self.assertEqual(
            record["payload"]["parents"], [{"dataset": "data", "version": 1}]
        )
        self.assertEqual(record["payload"]["metadata"]["recipe"]["kind"], "projection")
        self.assertEqual(self.vault.checks.inspect()["versions"], 2)

    def test_projection_unknown_fields_fail(self):
        with self.assertRaises(VaultError):
            self.vault.derivations.project(
                "project",
                "data",
                1,
                "derived",
                1,
                {"id": "missing"},
                [],
                "internal",
                ["research"],
            )
        with self.assertRaises(VaultError):
            self.vault.derivations.project(
                "project",
                "data",
                1,
                "derived",
                1,
                {"id": "id"},
                [{"field": "missing", "operator": "eq", "value": "x"}],
                "internal",
                ["research"],
            )
        self.assertEqual(self.vault.registry.list_versions("derived"), [])

    def test_foreign_key_reports_missing_parent(self):
        ingest(
            self.vault,
            "child",
            "derived",
            rows=[
                {"id": "one", "value": "1", "group": "A"},
                {"id": "missing", "value": "2", "group": "A"},
            ],
        )
        checked = self.vault.checks.foreign_key("derived", 1, "data", 1, ["id"], ["id"])
        self.assertFalse(checked["valid"])
        self.assertEqual(checked["checked"], 2)
        self.assertEqual(checked["missing"][0]["key"], ["missing"])
        with self.assertRaises(VaultError):
            self.vault.checks.foreign_key(
                "derived", 1, "data", 1, ["id"], ["id", "group"]
            )

    def test_left_join_preserves_missing_reference_as_null(self):
        join_schema = {
            "id": "joined",
            "version": 1,
            "fields": [
                {"name": "id", "type": "text"},
                {"name": "value", "type": "decimal", "nullable": True},
            ],
            "primary_key": ["id"],
        }
        self.vault.registry.add_schema(join_schema)
        self.vault.registry.register("joined", "Joined", "joined", "owner")
        ingest(
            self.vault,
            "right",
            "derived",
            rows=[{"id": "one", "value": "15", "group": "A"}],
        )
        joined = self.vault.derivations.join(
            "join",
            "data",
            1,
            "derived",
            1,
            "joined",
            1,
            ["id"],
            ["id"],
            {
                "id": {"side": "left", "field": "id"},
                "value": {"side": "right", "field": "value"},
            },
            "left",
            "internal",
            ["research"],
        )
        self.assertEqual(joined["rows"], 2)
        rows = self.vault.ingestion.rows("joined", 1)
        self.assertEqual(rows[0], {"id": "one", "value": "15"})
        self.assertEqual(rows[1], {"id": "two", "value": None})
        self.assertEqual(
            len(self.vault.registry.closure([{"dataset": "joined", "version": 1}])), 3
        )

    def test_join_rejects_ambiguous_right_key(self):
        ingest(
            self.vault,
            "right",
            "derived",
            rows=[
                {"id": "x", "value": "1", "group": "A"},
                {"id": "y", "value": "2", "group": "A"},
            ],
        )
        with self.assertRaises(VaultError) as caught:
            self.vault.derivations.join(
                "join",
                "data",
                1,
                "derived",
                1,
                "derived",
                1,
                ["group"],
                ["group"],
                {"id": {"side": "left", "field": "id"}},
                "inner",
                "internal",
                ["research"],
                expected_version=1,
            )
        self.assertEqual(caught.exception.code, "recipe")
        self.assertEqual(len(self.vault.registry.list_versions("derived")), 1)

    def test_group_aggregate(self):
        aggregate_schema = {
            "id": "summary",
            "version": 1,
            "fields": [
                {"name": "group", "type": "text"},
                {"name": "total", "type": "decimal"},
                {"name": "count", "type": "integer"},
            ],
            "primary_key": ["group"],
        }
        self.vault.registry.add_schema(aggregate_schema)
        self.vault.registry.register("summary", "Summary", "summary", "owner")
        result = self.vault.derivations.aggregate(
            "aggregate",
            "data",
            1,
            "summary",
            1,
            ["group"],
            [
                {"output": "total", "operation": "sum", "field": "value"},
                {"output": "count", "operation": "count"},
            ],
            "internal",
            ["research"],
        )
        self.assertEqual(result["rows"], 2)
        self.assertEqual(
            self.vault.ingestion.rows("summary", 1),
            [
                {"group": "A", "total": "10", "count": 1},
                {"group": "B", "total": "20", "count": 1},
            ],
        )
        with self.assertRaises(VaultError):
            self.vault.derivations.aggregate(
                "bad",
                "data",
                1,
                "summary",
                1,
                ["group"],
                [{"output": "group", "operation": "count"}],
                "internal",
                ["research"],
                expected_version=1,
            )


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.vault = Vault(self.root / "vault")
        self.addCleanup(self.vault.close)
        register(self.vault)
        ingest(self.vault)

    def test_release_requires_independent_reviewer_and_scope(self):
        planned = self.vault.releases.plan(
            "release", [{"dataset": "data", "version": 1}], "research", "internal"
        )
        self.assertEqual(planned["state"], "draft")
        self.assertEqual(planned["revision"], 1)
        with self.assertRaises(VaultError):
            self.vault.releases.approve("release", 1, "owner", "Self approval")
        with self.assertRaises(VaultError):
            self.vault.releases.approve("release", 2, "reviewer", "Stale revision")
        approved = self.vault.releases.approve("release", 1, "reviewer", "Reviewed")
        self.assertEqual(approved["state"], "approved")
        self.assertEqual(approved["revision"], 2)
        with self.assertRaises(VaultError):
            self.vault.releases.plan(
                "public", [{"dataset": "data", "version": 1}], "research", "public"
            )
        with self.assertRaises(VaultError):
            self.vault.releases.plan(
                "scope", [{"dataset": "data", "version": 1}], "commercial", "internal"
            )

    def test_export_is_immutable_and_deterministic(self):
        release(self.vault)
        first = self.vault.exporter.export("export", "release", self.root / "release")
        self.assertEqual(first["state"], "complete")
        self.assertEqual(first["rows"], 2)
        again = self.vault.exporter.export("export", "release", self.root / "release")
        self.assertTrue(again["reused"])
        second = self.vault.exporter.export("another", "release", self.root / "copy")
        self.assertEqual(first["digest"], second["digest"])
        self.assertEqual(
            (self.root / "release/manifest.json").read_bytes(),
            (self.root / "copy/manifest.json").read_bytes(),
        )
        self.assertEqual(self.vault.exporter.pending(), [])
        with self.assertRaises(VaultError):
            self.vault.exporter.export("different", "release", self.root / "release")
        with self.assertRaises(VaultError):
            self.vault.exporter.export("export", "release", self.root / "different")

    def test_export_resumes_partial_members(self):
        register(self.vault, "second")
        ingest(
            self.vault,
            "second",
            "second",
            rows=[{"id": "different", "value": "5", "group": "A"}],
        )
        release(
            self.vault,
            references=[
                {"dataset": "data", "version": 1},
                {"dataset": "second", "version": 1},
            ],
        )
        first = self.vault.exporter.export(
            "export", "release", self.root / "release", maximum_members=1
        )
        self.assertEqual(first["state"], "writing")
        self.assertFalse((self.root / "release").exists())
        self.assertEqual(len(self.vault.exporter.pending()), 1)
        complete = self.vault.exporter.export(
            "export", "release", self.root / "release"
        )
        self.assertEqual(complete["state"], "complete")
        self.assertEqual(complete["artifacts"], 2)
        self.assertFalse((self.root / ".vault-export-export").exists())
        self.assertEqual(verify_directory(self.root / "release")["versions"], 2)

    def test_withdrawal_blocks_approved_export(self):
        release(self.vault)
        self.vault.registry.withdraw("data", 1, "Invalid source")
        with self.assertRaises(VaultError) as caught:
            self.vault.exporter.export("export", "release", self.root / "release")
        self.assertEqual(caught.exception.code, "lineage")
        self.assertFalse((self.root / "release").exists())
        self.assertEqual(self.vault.exporter.pending(), [])
        self.assertEqual(self.vault.releases.get("release")["state"], "approved")

    def test_manifest_and_artifact_tampering_are_detected(self):
        release(self.vault)
        self.vault.exporter.export("export", "release", self.root / "release")
        manifest = self.root / "release/manifest.json"
        data = json.loads(manifest.read_text())
        checksum = data["manifest"]["versions"][0]["artifact"]
        (self.root / "release" / (checksum + ".jsonl")).write_bytes(b"changed")
        with self.assertRaises(VaultError):
            verify_directory(self.root / "release")
        with self.assertRaises(VaultError):
            self.vault.exporter.export("export", "release", self.root / "release")
        manifest.write_text('{"digest":"wrong","manifest":{"schema":1}}')
        with self.assertRaises(VaultError):
            verify_directory(self.root / "release")

    def test_extra_file_and_symlink_rejected(self):
        release(self.vault)
        self.vault.exporter.export("export", "release", self.root / "release")
        extra = self.root / "release/extra"
        extra.write_text("not in manifest")
        with self.assertRaises(VaultError):
            verify_directory(self.root / "release")
        extra.unlink()
        link = self.root / "linked"
        link.symlink_to(self.root / "release", target_is_directory=True)
        with self.assertRaises(VaultError):
            verify_directory(link)
        self.assertEqual(verify_directory(self.root / "release")["rows"], 2)

    def test_release_revoke_and_diff(self):
        release(self.vault)
        ingest(self.vault, "second", expected=1)
        self.vault.releases.plan(
            "second", [{"dataset": "data", "version": 2}], "research", "internal"
        )
        compared = self.vault.releases.compare("release", "second")
        self.assertEqual(compared["added"], [{"dataset": "data", "version": 2}])
        self.assertEqual(compared["removed"], [{"dataset": "data", "version": 1}])
        revoked = self.vault.releases.revoke("release", "Distribution halted")
        self.assertEqual(revoked["state"], "revoked")
        self.assertEqual(self.vault.releases.revoke("release", "Repeated"), revoked)
        with self.assertRaises(VaultError):
            self.vault.exporter.export("export", "release", self.root / "release")
