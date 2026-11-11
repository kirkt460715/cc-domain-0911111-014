from support import schema
import unittest
import tempfile
import json
import threading
import io
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from trialvault.service import Vault
from trialvault.validation import VaultError, digest
from trialvault.app import main, dispatch
from support import register, ingest, release


class PatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.vault = Vault(Path(self.temp.name) / "vault")
        self.addCleanup(self.vault.close)
        register(self.vault)
        ingest(self.vault)

    def test_patch_creates_new_version_and_retains_original(self):
        original = self.vault.ingestion.rows("data", 1)
        changes = [
            {
                "operation": "replace",
                "key": ["one"],
                "row": {"id": "one", "value": "11", "group": "A"},
                "expected_digest": digest(original[0]),
            },
            {"operation": "delete", "key": ["two"]},
            {
                "operation": "insert",
                "key": ["three"],
                "row": {"id": "three", "value": "30", "group": "B"},
            },
        ]
        planned = self.vault.patches.plan("patch", "data", 1, changes)
        self.assertEqual(planned["state"], "planned")
        self.assertEqual(planned["preview"]["before_rows"], 2)
        self.assertEqual(planned["preview"]["after_rows"], 2)
        self.assertNotIn("result_rows", planned["preview"])
        applied = self.vault.patches.apply("patch")
        self.assertEqual(applied["state"], "applied")
        self.assertEqual(applied["result"]["version"], 2)
        self.assertEqual(self.vault.patches.apply("patch"), applied)
        self.assertEqual(self.vault.ingestion.rows("data", 1), original)
        self.assertEqual(self.vault.ingestion.rows("data", 2)[0]["value"], "11")
        self.assertEqual(
            self.vault.registry.version("data", 2)["payload"]["parents"],
            [{"dataset": "data", "version": 1}],
        )
        self.assertEqual(self.vault.checks.inspect()["versions"], 2)

    def test_patch_preconditions_are_strict(self):
        with self.assertRaises(VaultError):
            self.vault.patches.plan(
                "bad",
                "data",
                1,
                [{"operation": "delete", "key": ["one"], "expected_digest": "bad"}],
            )
        with self.assertRaises(VaultError):
            self.vault.patches.plan(
                "bad",
                "data",
                1,
                [
                    {
                        "operation": "insert",
                        "key": ["one"],
                        "row": {"id": "one", "value": "1", "group": "A"},
                    }
                ],
            )
        with self.assertRaises(VaultError):
            self.vault.patches.plan(
                "bad", "data", 1, [{"operation": "delete", "key": ["missing"]}]
            )
        with self.assertRaises(VaultError):
            self.vault.patches.plan(
                "bad",
                "data",
                1,
                [
                    {
                        "operation": "replace",
                        "key": ["one"],
                        "row": {"id": "other", "value": "1", "group": "A"},
                    }
                ],
            )
        with self.assertRaises(VaultError):
            self.vault.patches.plan(
                "bad",
                "data",
                1,
                [
                    {"operation": "delete", "key": ["one"]},
                    {"operation": "delete", "key": ["one"]},
                ],
            )
        self.assertEqual(len(self.vault.registry.list_versions("data")), 1)

    def test_patch_head_change_and_cancel(self):
        self.vault.patches.plan(
            "patch", "data", 1, [{"operation": "delete", "key": ["one"]}]
        )
        ingest(self.vault, "second", expected=1)
        with self.assertRaises(VaultError) as caught:
            self.vault.patches.apply("patch")
        self.assertEqual(caught.exception.code, "conflict")
        cancelled = self.vault.patches.cancel(
            "patch", "New version superseded the plan"
        )
        self.assertEqual(cancelled["state"], "cancelled")
        with self.assertRaises(VaultError):
            self.vault.patches.apply("patch")
        self.assertEqual(len(self.vault.registry.list_versions("data")), 2)

    def test_withdrawal_invalidates_pending_patch(self):
        self.vault.patches.plan(
            "patch", "data", 1, [{"operation": "delete", "key": ["one"]}]
        )
        self.vault.registry.withdraw("data", 1, "Source issue")
        with self.assertRaises(VaultError):
            self.vault.patches.apply("patch")
        self.assertEqual(self.vault.patches.get("patch")["state"], "planned")
        self.assertEqual(len(self.vault.registry.list_versions("data")), 1)


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.vault = Vault(Path(self.temp.name) / "vault")
        self.addCleanup(self.vault.close)
        register(self.vault)
        ingest(self.vault)
        self.rules = [
            {"id": "count", "kind": "row_count", "minimum": 2, "maximum": 2},
            {
                "id": "total",
                "kind": "sum",
                "field": "value",
                "expected": "30",
                "tolerance": "0",
            },
            {"id": "nonnull", "kind": "not_null", "fields": ["id", "value"]},
            {"id": "unique", "kind": "unique", "fields": ["group"]},
            {"id": "ordered", "kind": "monotonic", "fields": ["value"]},
        ]

    def test_quality_suite_and_release_attestation(self):
        defined = self.vault.quality.define("suite", "schema", 1, self.rules)
        self.assertEqual(len(defined["rules"]), 5)
        result = self.vault.quality.run("run", "suite", "data", 1)
        self.assertTrue(result["passed"])
        self.assertEqual(result["rules"][1]["observed"], "30")
        self.assertEqual(self.vault.quality.run("run", "suite", "data", 1), result)
        release(self.vault)
        attested = self.vault.quality.attest_release("release", ["run"])
        self.assertTrue(attested["passed"])
        self.assertEqual(attested["covered"], 1)
        self.assertEqual(attested["runs"], ["run"])
        self.assertEqual(len(attested["evidence_digest"]), 64)

    def test_failed_quality_evidence_cannot_attest(self):
        rules = [
            {
                "id": "total",
                "kind": "sum",
                "field": "value",
                "expected": "31",
                "tolerance": "0",
            }
        ]
        self.vault.quality.define("suite", "schema", 1, rules)
        result = self.vault.quality.run("run", "suite", "data", 1)
        self.assertFalse(result["passed"])
        self.assertEqual(result["rules"][0]["observed"], "30")
        release(self.vault)
        with self.assertRaises(VaultError) as caught:
            self.vault.quality.attest_release("release", ["run"])
        self.assertEqual(caught.exception.code, "quality")
        self.assertEqual(self.vault.releases.get("release")["state"], "approved")

    def test_suite_immutability_and_run_identity(self):
        self.vault.quality.define("suite", "schema", 1, self.rules)
        self.assertEqual(
            self.vault.quality.define("suite", "schema", 1, self.rules)["id"], "suite"
        )
        with self.assertRaises(VaultError):
            self.vault.quality.define("suite", "schema", 1, self.rules[:1])
        self.vault.quality.run("run", "suite", "data", 1)
        ingest(self.vault, "second", expected=1)
        with self.assertRaises(VaultError):
            self.vault.quality.run("run", "suite", "data", 2)
        self.assertTrue(self.vault.quality.result("run")["passed"])

    def test_withdrawal_makes_attestation_stale(self):
        self.vault.quality.define("suite", "schema", 1, self.rules)
        self.vault.quality.run("run", "suite", "data", 1)
        release(self.vault)
        self.vault.registry.withdraw("data", 1, "Correction required")
        with self.assertRaises(VaultError):
            self.vault.quality.attest_release("release", ["run"])
        self.assertTrue(self.vault.quality.result("run")["passed"])

    def test_unique_and_monotonic_violations(self):
        ingest(
            self.vault,
            "second",
            rows=[
                {"id": "a", "value": "20", "group": "A"},
                {"id": "b", "value": "10", "group": "A"},
            ],
            expected=1,
        )
        self.vault.quality.define("suite", "schema", 1, self.rules)
        result = self.vault.quality.run("run", "suite", "data", 2)
        self.assertFalse(result["passed"])
        reports = {item["id"]: item for item in result["rules"]}
        self.assertEqual(reports["unique"]["violation_count"], 1)
        self.assertEqual(reports["unique"]["violations"][0]["previous_row"], 1)
        self.assertEqual(reports["ordered"]["violation_count"], 1)
        self.assertEqual(reports["ordered"]["violations"][0]["reason"], "decreasing")
        self.assertTrue(reports["total"]["passed"])

    def test_invalid_rule_definition_fails_early(self):
        invalid = [
            {
                "id": "sum",
                "kind": "sum",
                "field": "id",
                "expected": "1",
                "tolerance": "0",
            },
            {"id": "unique", "kind": "unique", "fields": ["missing"]},
            {"id": "bad", "kind": "row_count", "minimum": 3, "maximum": 1},
            {"id": "bad", "kind": "unknown"},
        ]
        for rule in invalid:
            with self.subTest(rule=rule):
                with self.assertRaises(VaultError):
                    self.vault.quality.define("bad", "schema", 1, [rule])
        self.assertEqual(self.vault.database.rows("SELECT * FROM quality_suites"), [])


class RecoveryTests(unittest.TestCase):
    def test_reopen_retains_versions_and_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "vault"
            with Vault(root) as vault:
                register(vault)
                ingest(vault)
                original = release(vault)
            with Vault(root) as vault:
                self.assertEqual(vault.releases.get("release"), original)
                self.assertEqual(vault.ingestion.rows("data", 1)[0]["id"], "one")
                self.assertEqual(vault.checks.inspect()["versions"], 1)
                self.assertEqual(
                    vault.exporter.export(
                        "export", "release", Path(directory) / "release"
                    )["state"],
                    "complete",
                )

    def test_concurrent_same_ingestion_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "vault"
            with Vault(root) as vault:
                register(vault)
            vaults = [Vault(root), Vault(root)]
            barrier = threading.Barrier(2)
            results = []
            errors = []

            def worker(vault):
                try:
                    barrier.wait()
                    results.append(ingest(vault))
                except Exception as exc:
                    errors.append(exc)

            threads = [
                threading.Thread(target=worker, args=(vault,)) for vault in vaults
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(10)
            self.assertFalse(any(thread.is_alive() for thread in threads))
            self.assertEqual(errors, [])
            self.assertEqual(
                sorted(result["reused"] for result in results), [False, True]
            )
            self.assertEqual(len(vaults[0].registry.list_versions("data")), 1)
            for vault in vaults:
                vault.close()

    def test_export_recovers_rename_before_database_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            with Vault(Path(directory) / "vault") as vault:
                register(vault)
                ingest(vault)
                release(vault)
                target = Path(directory) / "release"
                vault.exporter.export("export", "release", target)
                with vault.database.transaction() as con:
                    con.execute("UPDATE exports SET state='writing' WHERE id='export'")
                recovered = vault.exporter.export("export", "release", target)
                self.assertEqual(recovered["state"], "complete")
                self.assertTrue(recovered["reused"])
                self.assertEqual(vault.exporter.pending(), [])

    def test_cli_demo_and_structured_error(self):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["demo"])
        self.assertEqual(code, 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["verification"]["rows"], 2)
        self.assertEqual(result["export"]["state"], "complete")
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / "bad.json"
            request.write_text('{"action":"unknown"}')
            error = io.StringIO()
            with redirect_stderr(error):
                code = main(
                    ["--root", str(Path(directory) / "vault"), "run", str(request)]
                )
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(error.getvalue())["error"]["code"], "input")

    def test_dispatch_and_audit_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            with Vault(Path(directory) / "vault") as vault:
                register(vault)
                ingest(vault)
                self.assertEqual(dispatch(vault, {"action": "check"})["versions"], 1)
                self.assertEqual(
                    dispatch(
                        vault, {"action": "profile", "dataset": "data", "version": 1}
                    )["rows"],
                    2,
                )
                with vault.database.transaction() as con:
                    con.execute("UPDATE audit SET payload='{}' WHERE sequence=1")
                with self.assertRaises(VaultError) as caught:
                    vault.checks.inspect()
                self.assertEqual(caught.exception.code, "integrity")


class MigrationRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.vault = Vault(Path(self.temp.name) / "vault")
        self.addCleanup(self.vault.close)
        register(self.vault)
        ingest(self.vault)

    def test_migration_adds_default_and_preserves_lineage(self):
        contract = schema(2)
        contract["fields"].append({"name": "reviewed", "type": "boolean"})
        self.vault.registry.add_schema(contract, True)
        preview = self.vault.migrations.preview(
            "migration", "data", 1, 2, defaults={"reviewed": False}
        )
        self.assertTrue(preview["report"]["ready"])
        self.assertEqual(preview["report"]["valid_rows"], 2)
        self.assertFalse(preview["report"]["compatibility"]["compatible"])
        self.assertEqual(len(self.vault.migrations.pending()), 1)
        applied = self.vault.migrations.apply("migration")
        self.assertEqual(applied["state"], "applied")
        self.assertEqual(applied["result"]["version"], 2)
        self.assertEqual(self.vault.registry.version("data", 2)["schema_version"], 2)
        self.assertFalse(self.vault.ingestion.rows("data", 2)[0]["reviewed"])
        self.assertNotIn("reviewed", self.vault.ingestion.rows("data", 1)[0])
        self.assertEqual(self.vault.migrations.pending(), [])
        self.assertEqual(self.vault.migrations.apply("migration"), applied)

    def test_migration_reports_row_failures_without_mutation(self):
        contract = schema(2)
        contract["fields"][1]["minimum"] = "15"
        self.vault.registry.add_schema(contract, True)
        preview = self.vault.migrations.preview("migration", "data", 1, 2)
        self.assertFalse(preview["report"]["ready"])
        self.assertEqual(preview["report"]["error_count"], 1)
        self.assertEqual(preview["report"]["errors"][0]["old_key"], '["one"]')
        with self.assertRaises(VaultError):
            self.vault.migrations.apply("migration")
        self.assertEqual(len(self.vault.registry.list_versions("data")), 1)
        self.assertEqual(self.vault.checks.inspect()["versions"], 1)

    def test_rename_and_drop_migration(self):
        contract = {
            "id": "schema",
            "version": 2,
            "fields": [
                {"name": "sample", "type": "text"},
                {"name": "value", "type": "decimal"},
            ],
            "primary_key": ["sample"],
        }
        self.vault.registry.add_schema(contract, True)
        preview = self.vault.migrations.preview(
            "migration", "data", 1, 2, renames={"id": "sample"}, drop=["group"]
        )
        self.assertTrue(preview["report"]["ready"])
        self.vault.migrations.apply("migration")
        self.assertEqual(
            self.vault.ingestion.rows("data", 2)[0], {"sample": "one", "value": "10"}
        )
        with self.assertRaises(VaultError):
            self.vault.checks.compare("data", 1, 2)
        self.assertEqual(self.vault.checks.inspect()["versions"], 2)

    def test_retention_respects_release_and_active_children(self):
        release(self.vault)
        ingest(self.vault, "second", expected=1)
        self.vault.registry.withdraw("data", 1, "Superseded")
        report = self.vault.retention.review("review")
        self.assertEqual(report["candidates"], [])
        old = [item for item in report["retained"] if item["version"] == 1][0]
        self.assertIn("release:release", old["reasons"])
        self.vault.releases.revoke("release", "No distribution")
        second = self.vault.retention.review("new-review")
        self.assertEqual(len(second["candidates"]), 1)
        self.assertEqual(second["candidates"][0]["version"], 1)
        self.assertEqual(second["action"], "review_only_no_version_deletion")
        self.assertTrue(self.vault.retention.verify("new-review")["current"])
        self.assertEqual(len(self.vault.registry.list_versions("data")), 2)

    def test_retention_review_detects_changed_version_state(self):
        self.vault.retention.review("review")
        self.assertTrue(self.vault.retention.verify("review")["current"])
        ingest(self.vault, "second", expected=1)
        self.assertFalse(self.vault.retention.verify("review")["current"])
        with self.assertRaises(VaultError):
            self.vault.retention.review("review")
        self.assertEqual(len(self.vault.registry.list_versions("data")), 2)


class QueryTests(unittest.TestCase):
    def test_catalog_sampling_lookup_and_deduplication(self):
        with tempfile.TemporaryDirectory() as directory:
            with Vault(Path(directory) / "vault") as vault:
                register(vault)
                ingest(vault)
                ingest(vault, "second", expected=1)
                search = vault.queries.search(owner="owner", scope="research")
                self.assertEqual(search["items"][0]["version"], 2)
                self.assertEqual(vault.queries.search(scope="commercial")["items"], [])
                sample = vault.queries.sample("data", 1, 1, "seed")
                self.assertEqual(sample, vault.queries.sample("data", 1, 1, "seed"))
                self.assertEqual(len(sample["rows"]), 1)
                self.assertTrue(vault.queries.lookup("data", 1, ["one"])["found"])
                self.assertFalse(vault.queries.lookup("data", 1, ["missing"])["found"])
                usage = vault.queries.storage_usage()
                self.assertEqual(len(usage["artifacts"]), 1)
                self.assertEqual(usage["logical_rows"], 4)
                self.assertEqual(usage["logical_bytes"], 2 * usage["physical_bytes"])
                self.assertEqual(usage["deduplicated_bytes"], usage["physical_bytes"])
