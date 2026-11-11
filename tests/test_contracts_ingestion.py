import tempfile
import unittest
from pathlib import Path
from trialvault.service import Vault
from trialvault.validation import VaultError
from trialvault.contracts.schema import Schema, Field
from trialvault.contracts.evolution import compatibility
from support import schema, register, ingest


class SchemaTests(unittest.TestCase):
    def test_field_validation_is_typed(self):
        integer = Field.parse(
            {"name": "n", "type": "integer", "minimum": "1", "maximum": "5"}
        )
        self.assertEqual(integer.validate(3), 3)
        for value in [True, "3", 0, 6, 1.2, None]:
            with self.subTest(value=value):
                with self.assertRaises(VaultError):
                    integer.validate(value)
        boolean = Field.parse({"name": "b", "type": "boolean"})
        self.assertTrue(boolean.validate(True))
        with self.assertRaises(VaultError):
            boolean.validate(1)

    def test_decimal_normalization_and_nullability(self):
        field = Field.parse({"name": "x", "type": "decimal", "nullable": True})
        self.assertEqual(field.validate("12.500"), "12.5")
        self.assertIsNone(field.validate(None))
        for value in ["NaN", "Infinity", "1e31", "0.0000000000000000001"]:
            with self.assertRaises(VaultError):
                field.validate(value)
        timestamp = Field.parse({"name": "at", "type": "timestamp"})
        self.assertEqual(
            timestamp.validate("2026-11-11T08:00:00+08:00"),
            "2026-11-11T00:00:00.000000Z",
        )
        with self.assertRaises(VaultError):
            timestamp.validate("2026-11-11T00:00:00")

    def test_primary_key_contract(self):
        contract = Schema(schema())
        normalized = contract.validate({"id": "one", "value": "1.00", "group": "A"})
        self.assertEqual(normalized["value"], "1")
        self.assertEqual(contract.row_key(normalized), '["one"]')
        self.assertEqual(Schema(contract.as_dict()).fingerprint, contract.fingerprint)
        with self.assertRaises(VaultError):
            Schema(schema(primary_key=["absent"]))
        with self.assertRaises(VaultError):
            Schema(schema(fields=[{"name": "id", "type": "text", "nullable": True}]))
        with self.assertRaises(VaultError):
            contract.validate({"id": "one", "value": "1", "group": "A", "extra": 1})

    def test_optional_fields_and_enum(self):
        contract = Schema(
            {
                "id": "s",
                "version": 1,
                "fields": [
                    {"name": "id", "type": "text"},
                    {
                        "name": "optional",
                        "type": "text",
                        "required": False,
                        "maximum_length": 2,
                    },
                ],
                "primary_key": ["id"],
            }
        )
        self.assertEqual(contract.validate({"id": "one"}), {"id": "one"})
        with self.assertRaises(VaultError):
            contract.validate({"id": "one", "optional": "long"})
        with self.assertRaises(VaultError):
            Schema(schema()).validate({"id": "one", "value": "1", "group": "C"})
        with self.assertRaises(VaultError):
            Field.parse({"name": "enum", "type": "enum", "choices": ["A", "A"]})
        with self.assertRaises(VaultError):
            Field.parse({"name": "text", "type": "text", "minimum": "0"})

    def test_evolution_distinguishes_safe_and_breaking(self):
        old = schema()
        added = schema(
            2,
            fields=old["fields"]
            + [{"name": "note", "type": "text", "required": False}],
        )
        safe = compatibility(old, added)
        self.assertTrue(safe["compatible"])
        self.assertEqual(safe["added_fields"], ["note"])
        required = schema(2, fields=old["fields"] + [{"name": "note", "type": "text"}])
        self.assertFalse(compatibility(old, required)["compatible"])
        narrowed = schema(2)
        narrowed["fields"][1]["minimum"] = "5"
        report = compatibility(old, narrowed)
        self.assertIn("minimum_narrowed", [item["code"] for item in report["breaking"]])
        with self.assertRaises(VaultError):
            compatibility(old, old)

    def test_enum_and_null_evolution(self):
        old = {
            "id": "s",
            "version": 1,
            "fields": [
                {"name": "id", "type": "text"},
                {
                    "name": "kind",
                    "type": "enum",
                    "choices": ["A", "B"],
                    "nullable": True,
                },
            ],
            "primary_key": ["id"],
        }
        new = {
            **old,
            "version": 2,
            "fields": [
                old["fields"][0],
                {"name": "kind", "type": "enum", "choices": ["A"], "nullable": False},
            ],
        }
        report = compatibility(old, new)
        codes = {item["code"] for item in report["breaking"]}
        self.assertIn("enum_narrowed", codes)
        self.assertIn("nullability_narrowed", codes)
        expanded = {
            **old,
            "version": 2,
            "fields": [
                old["fields"][0],
                {
                    "name": "kind",
                    "type": "enum",
                    "choices": ["A", "B", "C"],
                    "nullable": True,
                },
            ],
        }
        self.assertTrue(compatibility(old, expanded)["compatible"])


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.vault = Vault(Path(self.temp.name) / "vault")
        self.addCleanup(self.vault.close)
        register(self.vault)

    def test_ingestion_normalizes_and_deduplicates_artifact(self):
        first = ingest(self.vault)
        self.assertEqual(first["rows"], 2)
        second = ingest(self.vault)
        self.assertTrue(second["reused"])
        self.assertEqual(first["artifact"], second["artifact"])
        rows = self.vault.ingestion.rows("data", 1)
        self.assertEqual([row["id"] for row in rows], ["one", "two"])
        self.assertEqual(len(self.vault.artifacts.inventory()), 1)
        self.assertEqual(self.vault.checks.inspect()["versions"], 1)
        self.assertEqual(self.vault.registry.version("data", 1)["state"], "active")

    def test_request_identity_and_optimistic_head(self):
        ingest(self.vault)
        with self.assertRaises(VaultError):
            ingest(self.vault, rows=[{"id": "one", "value": "99", "group": "A"}])
        with self.assertRaises(VaultError) as caught:
            ingest(self.vault, request="next", expected=0)
        self.assertEqual(caught.exception.code, "conflict")
        second = ingest(self.vault, request="next", expected=1)
        self.assertEqual(second["version"], 2)
        self.assertEqual(len(self.vault.registry.list_versions("data")), 2)
        self.assertEqual(len(self.vault.artifacts.inventory()), 1)

    def test_invalid_rows_leave_no_catalog_version(self):
        invalid = [
            {"id": "one", "value": "10", "group": "A"},
            {"id": "one", "value": "11", "group": "A"},
        ]
        with self.assertRaises(VaultError) as caught:
            ingest(self.vault, rows=invalid)
        self.assertEqual(caught.exception.code, "row")
        self.assertEqual(self.vault.registry.list_versions("data"), [])
        self.assertEqual(self.vault.artifacts.inventory(), [])
        with self.assertRaises(VaultError):
            self.vault.ingestion.ingest(
                "req", "data", 1, b"{bad}", "jsonl", "internal", ["research"]
            )
        self.assertEqual(self.vault.registry.list_versions("data"), [])

    def test_csv_converts_only_declared_types(self):
        contract = {
            "id": "csv-schema",
            "version": 1,
            "fields": [
                {"name": "id", "type": "integer"},
                {"name": "flag", "type": "boolean"},
                {"name": "value", "type": "decimal", "nullable": True},
            ],
            "primary_key": ["id"],
        }
        self.vault.registry.add_schema(contract)
        self.vault.registry.register("csv", "CSV dataset", "csv-schema", "owner")
        result = self.vault.ingestion.ingest(
            "csv",
            "csv",
            1,
            b"id,flag,value\n1,true,1.20\n2,false,\n",
            "csv",
            "public",
            ["research"],
        )
        self.assertEqual(result["rows"], 2)
        rows = self.vault.ingestion.rows("csv", 1)
        self.assertEqual(rows[0], {"id": 1, "flag": True, "value": "1.2"})
        self.assertIsNone(rows[1]["value"])
        with self.assertRaises(VaultError):
            self.vault.ingestion.ingest(
                "bad",
                "csv",
                1,
                b"id,flag,value\n3,yes,1\n",
                "csv",
                "public",
                ["research"],
                expected_version=1,
            )
        with self.assertRaises(VaultError):
            self.vault.ingestion.parse(
                b"id,id,flag\n", "csv", self.vault.registry.schema("csv-schema", 1)
            )

    def test_schema_versions_require_explicit_breaking(self):
        changed = schema(
            2, fields=schema()["fields"] + [{"name": "extra", "type": "text"}]
        )
        with self.assertRaises(VaultError):
            self.vault.registry.add_schema(changed)
        accepted = self.vault.registry.add_schema(changed, allow_breaking=True)
        self.assertFalse(accepted["change"]["compatible"])
        self.assertTrue(self.vault.registry.add_schema(changed)["existing"])
        with self.assertRaises(VaultError):
            self.vault.registry.add_schema(schema(4))
        with self.assertRaises(VaultError):
            self.vault.registry.add_schema(schema(2))

    def test_artifact_tampering_and_paths(self):
        record = ingest(self.vault)
        path = self.vault.artifacts.path(record["artifact"])
        path.write_bytes(b"tampered")
        with self.assertRaises(VaultError) as caught:
            self.vault.ingestion.rows("data", 1)
        self.assertEqual(caught.exception.code, "integrity")
        with self.assertRaises(VaultError):
            self.vault.artifacts.path("../bad")
        with self.assertRaises(VaultError):
            self.vault.checks.inspect()

    def test_safe_orphan_cleanup(self):
        record = ingest(self.vault)
        orphan = self.vault.artifacts.put(b"orphan")
        preview = self.vault.checks.unused_artifacts()
        self.assertTrue(preview["dry_run"])
        self.assertEqual(
            [item["sha256"] for item in preview["candidates"]], [orphan["sha256"]]
        )
        self.assertEqual(len(self.vault.artifacts.inventory()), 2)
        deleted = self.vault.checks.unused_artifacts(False)
        self.assertFalse(deleted["dry_run"])
        self.assertEqual(len(self.vault.artifacts.inventory()), 1)
        self.assertTrue(self.vault.artifacts.verify(record["artifact"])["verified"])
        stage = self.vault.artifacts.staging / "write-abandoned"
        stage.write_bytes(b"partial")
        self.assertEqual(self.vault.artifacts.clean_staging(), ["write-abandoned"])

    def test_profiles_and_version_diff(self):
        ingest(self.vault)
        profile = self.vault.checks.profile("data", 1)
        self.assertEqual(profile["fields"]["value"]["mean"], "15")
        self.assertEqual(profile["fields"]["group"]["frequencies"], {"A": 1, "B": 1})
        ingest(
            self.vault,
            "second",
            rows=[
                {"id": "one", "value": "11", "group": "A"},
                {"id": "three", "value": "30", "group": "B"},
            ],
            expected=1,
        )
        compared = self.vault.checks.compare("data", 1, 2)
        self.assertEqual(compared["added_count"], 1)
        self.assertEqual(compared["removed_count"], 1)
        self.assertEqual(compared["changed_count"], 1)
        self.assertEqual(compared["added"][0]["id"], "three")
        self.assertEqual(compared["removed"][0]["id"], "two")
        self.assertEqual(compared["changed"][0]["fields"][0]["field"], "value")
        self.assertFalse(compared["truncated"])
