import math
import tempfile
import unittest
from pathlib import Path
from trialvault.service import Vault
from trialvault.validation import VaultError, canonical
from trialvault.experiments.definition import parse_definition, plan_split
from trialvault.experiments.preprocessing import fit_preprocessing, transform_row
from trialvault.experiments.metrics import (
    confusion_at,
    best_threshold,
    pr_data,
    sweep,
)
from trialvault.contracts.schema import Schema


def specimen_schema():
    return {
        "id": "specimens",
        "version": 1,
        "fields": [
            {"name": "id", "type": "text"},
            {"name": "batch", "type": "text", "nullable": True},
            {"name": "cast_at", "type": "timestamp", "nullable": True},
            {"name": "thickness", "type": "decimal", "nullable": True},
            {"name": "pressure", "type": "decimal"},
            {
                "name": "alloy",
                "type": "enum",
                "choices": ["A", "B", "C"],
                "nullable": True,
            },
            {"name": "leak", "type": "decimal", "nullable": True},
            {"name": "failed", "type": "boolean", "nullable": True},
        ],
        "primary_key": ["id"],
    }


def batch_rows(batch, month, leak=None, alloys=("A", "A"), failures_from=50):
    rows = []
    for index, pressure in enumerate([10, 20, 30, 40, 60, 70, 80, 90]):
        rows.append(
            {
                "id": "%s-%d" % (batch, index),
                "batch": batch,
                "cast_at": "2026-%02d-%02dT00:00:00Z" % (month, index + 1),
                "thickness": None if index == 0 else "%d.5" % (pressure // 10),
                "pressure": str(pressure),
                "alloy": alloys[index % len(alloys)],
                "leak": leak,
                "failed": pressure >= failures_from,
            }
        )
    return rows


def content(rows):
    return ("\n".join(canonical(row) for row in rows) + "\n").encode()


DEFINITION = {
    "dataset": "specimens",
    "version": 1,
    "features": ["thickness", "pressure", "alloy", "leak"],
    "label": "failed",
    "batch": "batch",
    "time": "cast_at",
    "costs": {"false_positive": "1", "false_negative": "5"},
}


class ExperimentVault(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.vault = Vault(Path(self.temp.name) / "vault")
        self.addCleanup(self.vault.close)
        self.vault.registry.add_schema(specimen_schema())
        self.vault.registry.register(
            "specimens", "Synthetic specimens", "specimens", "lab"
        )

    def ingest(self, rows, request="ingest", expected=0):
        return self.vault.ingestion.ingest(
            request,
            "specimens",
            1,
            content(rows),
            "jsonl",
            "internal",
            ["research"],
            expected_version=expected,
        )

    def standard_rows(self):
        rows = []
        rows += batch_rows("b1", 1, alloys=("A", "B"))
        rows += batch_rows("b2", 2)
        rows += batch_rows("b3", 3, alloys=("A", "C"))
        rows += batch_rows("b4", 4, alloys=("A", "B"))
        rows += batch_rows("b5", 5, leak="0.5")
        return rows

    def define(self, experiment_id="exp", **changes):
        definition = {**DEFINITION, **changes}
        return self.vault.experiments.define(experiment_id, definition)


class DefinitionTests(ExperimentVault):
    def test_definition_binds_version_features_label_and_batch(self):
        self.ingest(self.standard_rows())
        defined = self.define()
        self.assertFalse(defined["existing"])
        definition = defined["definition"]
        self.assertEqual(definition["dataset"], "specimens")
        self.assertEqual(definition["version"], 1)
        self.assertEqual(definition["features"], DEFINITION["features"])
        self.assertEqual(definition["label"], "failed")
        self.assertEqual(definition["batch"], "batch")
        self.assertEqual(definition["costs"], {"false_positive": "1", "false_negative": "5"})
        split = definition["split"]
        self.assertEqual(split["holdout_batches"], ["b5"])
        self.assertEqual(split["train_batches"], ["b1", "b2", "b3", "b4"])
        roles = {item["id"]: item["role"] for item in split["batches"]}
        self.assertEqual(roles["b5"], "holdout")
        self.assertEqual(roles["b1"], "train")
        again = self.define()
        self.assertTrue(again["existing"])
        with self.assertRaises(VaultError) as caught:
            self.define(folds=3)
        self.assertEqual(caught.exception.code, "conflict")

    def test_definition_rejects_bad_references(self):
        self.ingest(self.standard_rows())
        for changes in [
            {"features": ["missing"]},
            {"features": ["pressure", "failed"]},
            {"features": ["pressure", "batch"]},
            {"features": ["cast_at"]},
            {"label": "pressure"},
            {"batch": "failed"},
            {"time": "pressure"},
            {"costs": {"false_positive": "0", "false_negative": "5"}},
            {"costs": {"false_positive": "1"}},
            {"positive": False},
            {"holdout_batches": 4},
        ]:
            with self.subTest(changes=changes):
                with self.assertRaises(VaultError):
                    self.define(**changes)
        self.assertEqual(self.vault.experiments.list(), {"experiments": []})

    def test_definition_requires_labeled_rows_and_batch_identity(self):
        rows = self.standard_rows()
        for row in rows:
            if row["id"] == "b1-0":
                row["failed"] = None
        self.ingest(rows)
        defined = self.define()
        self.assertEqual(defined["definition"]["unlabeled_rows"], 1)

    def test_null_batch_or_time_is_rejected(self):
        rows = self.standard_rows()
        rows[0]["batch"] = None
        self.ingest(rows)
        with self.assertRaises(VaultError) as caught:
            self.define()
        self.assertEqual(caught.exception.code, "experiment")

    def test_overlapping_holdout_time_is_rejected(self):
        rows = []
        rows += batch_rows("b1", 1)
        rows += batch_rows("b2", 2)
        rows += batch_rows("b3", 3)
        rows += batch_rows("b4", 4)
        rows += batch_rows("b5", 4)
        self.ingest(rows)
        with self.assertRaises(VaultError) as caught:
            self.define()
        self.assertEqual(caught.exception.code, "experiment")
        self.assertIn("strictly later", str(caught.exception))

    def test_withdrawn_version_cannot_define(self):
        self.ingest(self.standard_rows())
        self.vault.registry.withdraw("specimens", 1, "bad source")
        with self.assertRaises(VaultError) as caught:
            self.define()
        self.assertEqual(caught.exception.code, "lineage")


class SplitPlanTests(unittest.TestCase):
    def rows(self, spec):
        rows = []
        for batch, month in spec:
            rows.append(
                {"batch": batch, "cast_at": "2026-%02d-01T00:00:00Z" % month}
            )
        return rows

    def test_batches_stay_whole_and_holdout_is_latest(self):
        rows = self.rows([("b1", 1), ("b2", 2), ("b3", 3), ("b4", 4), ("b5", 5)])
        split = plan_split(rows, "batch", "cast_at", 1, None)
        self.assertEqual(split["holdout_batches"], ["b5"])
        self.assertEqual(len(split["folds"]), 4)
        covered = []
        for fold in split["folds"]:
            self.assertFalse(
                set(fold["validation_batches"]) & set(fold["train_batches"])
            )
            covered.extend(fold["validation_batches"])
        self.assertEqual(sorted(covered), ["b1", "b2", "b3", "b4"])
        split = plan_split(rows, "batch", "cast_at", 2, 2)
        self.assertEqual(split["holdout_batches"], ["b4", "b5"])
        self.assertEqual(len(split["folds"]), 2)

    def test_equal_boundary_times_rejected(self):
        rows = [
            {"batch": "b1", "cast_at": "2026-01-01T00:00:00Z"},
            {"batch": "b2", "cast_at": "2026-02-01T00:00:00Z"},
            {"batch": "b3", "cast_at": "2026-03-01T00:00:00Z"},
            {"batch": "b4", "cast_at": "2026-03-01T00:00:00Z"},
        ]
        with self.assertRaises(VaultError) as caught:
            plan_split(rows, "batch", "cast_at", 1, None)
        self.assertEqual(caught.exception.code, "experiment")

    def test_too_few_batches_rejected(self):
        rows = self.rows([("b1", 1), ("b2", 2)])
        with self.assertRaises(VaultError):
            plan_split(rows, "batch", "cast_at", 1, None)


class PreprocessingTests(unittest.TestCase):
    def specs(self):
        return [
            {"name": "num", "kind": "numeric"},
            {"name": "flag", "kind": "boolean"},
            {"name": "cat", "kind": "categorical"},
            {"name": "empty", "kind": "numeric"},
            {"name": "flat", "kind": "numeric"},
        ]

    def rows(self):
        return [
            {"num": "1", "flag": True, "cat": "x", "empty": None, "flat": "7"},
            {"num": "3", "flag": False, "cat": "y", "empty": None, "flat": "7"},
            {"num": None, "flag": None, "cat": None, "empty": None, "flat": "7"},
        ]

    def test_fit_drops_all_empty_and_constant_columns(self):
        fitted = fit_preprocessing(self.specs(), self.rows())
        reasons = {item["feature"]: item["reason"] for item in fitted["dropped"]}
        self.assertEqual(reasons, {"empty": "all_missing", "flat": "constant"})
        numeric = [c for c in fitted["columns"] if c["name"] == "num"][0]
        self.assertEqual(numeric["fill"], 2.0)
        categorical = [c for c in fitted["columns"] if c["name"] == "cat"][0]
        self.assertEqual(categorical["categories"], ["x", "y"])
        self.assertIn("cat=<missing>", fitted["encoded_features"])

    def test_transform_marks_imputation_and_unknown_category(self):
        fitted = fit_preprocessing(self.specs(), self.rows())
        vector, notes = transform_row(
            fitted, {"num": None, "flag": True, "cat": "unseen", "empty": "9", "flat": "8"}
        )
        codes = [note["code"] for note in notes]
        self.assertIn("imputed", codes)
        self.assertIn("unknown_category", codes)
        unknown = [n for n in notes if n["code"] == "unknown_category"][0]
        self.assertEqual(unknown["value"], "unseen")
        self.assertEqual(len(vector), len(fitted["encoded_features"]))
        # unseen category encodes as an all-zero block
        self.assertEqual(vector[-3:], [0.0, 0.0, 0.0])

    def test_high_cardinality_column_is_dropped(self):
        specs = [{"name": "cat", "kind": "categorical"}]
        rows = [{"cat": "v%d" % index} for index in range(150)]
        fitted = fit_preprocessing(specs, rows)
        self.assertEqual(fitted["dropped"][0]["reason"], "cardinality_exceeded")


class MetricsTests(unittest.TestCase):
    def test_sweep_boundaries(self):
        points = sweep([0.2, 0.8], [0, 1])
        self.assertEqual(points[0]["confusion"], {"tp": 1, "fp": 1, "tn": 0, "fn": 0})
        self.assertEqual(points[-1]["confusion"], {"tp": 0, "fp": 0, "tn": 1, "fn": 1})
        self.assertEqual(len(points), 3)

    def test_cost_drives_threshold_choice(self):
        scores = [0.2, 0.8]
        labels = [1, 0]
        balanced = best_threshold(scores, labels, 1, 1)
        self.assertEqual(balanced["expected_cost"], 1)
        self.assertGreater(balanced["threshold"], 0.8)
        sensitive = best_threshold(scores, labels, 1, 10)
        self.assertEqual(sensitive["threshold"], 0.2)
        self.assertEqual(sensitive["confusion"]["fn"], 0)

    def test_confusion_and_pr_nulls(self):
        confusion = confusion_at([0.9, 0.1], [1, 0], 0.5)
        self.assertEqual(confusion, {"tp": 1, "fp": 0, "tn": 1, "fn": 0})
        points = pr_data([0.5], [0])
        self.assertIsNone(points[0]["recall"])
        self.assertEqual(points[0]["precision"], 0.0)
        self.assertIsNone(points[-1]["precision"])


class ParseDefinitionTests(unittest.TestCase):
    def test_tuning_is_normalized_and_bounded(self):
        schema = Schema(specimen_schema())
        parsed = parse_definition(
            {
                **DEFINITION,
                "tuning": {"logistic_C": ["1", "0.5", "1"], "tree_max_depth": [3, 3]},
            },
            schema,
        )
        self.assertEqual(parsed["tuning"]["logistic_C"], ["0.5", "1"])
        self.assertEqual(parsed["tuning"]["tree_max_depth"], [3])
        with self.assertRaises(VaultError):
            parse_definition(
                {**DEFINITION, "tuning": {"logistic_C": ["0"]}}, schema
            )
        with self.assertRaises(VaultError):
            parse_definition(
                {**DEFINITION, "tuning": {"unknown": [1]}}, schema
            )


class RunTests(ExperimentVault):
    def setUp(self):
        super().setUp()
        self.ingest(self.standard_rows())
        self.define()

    def test_run_selects_by_cost_and_reports_holdout_once(self):
        result = self.vault.experiments.run("exp")
        self.assertFalse(result["reused"])
        self.assertEqual(result["status"], "complete")
        kinds = {candidate["model"] for candidate in result["candidates"]}
        self.assertEqual(kinds, {"logistic", "tree", "baseline"})
        best = min(candidate["cv_expected_cost"] for candidate in result["candidates"])
        self.assertEqual(result["selection"]["cv_expected_cost"], best)
        holdout = result["holdout"]
        self.assertEqual(holdout["batches"], ["b5"])
        self.assertEqual(holdout["threshold"], result["selection"]["threshold"])
        confusion = holdout["confusion"]
        self.assertEqual(
            confusion["tp"] + confusion["fp"] + confusion["tn"] + confusion["fn"],
            holdout["rows"],
        )
        self.assertEqual(holdout["rows"], 8)
        self.assertTrue(holdout["consumed"])
        self.assertEqual([item["batch"] for item in holdout["per_batch"]], ["b5"])
        self.assertEqual(holdout["per_batch"][0]["confusion"], confusion)
        self.assertGreater(len(holdout["pr_data"]), 1)
        for point in holdout["pr_data"]:
            self.assertIn("threshold", point)
            self.assertIn("precision", point)
            self.assertIn("recall", point)
        again = self.vault.experiments.run("exp")
        self.assertTrue(again["reused"])
        self.assertEqual(again["model_artifact"], result["model_artifact"])

    def test_split_integrity_in_run_result(self):
        result = self.vault.experiments.run("exp")
        train = set(result["batches"][i]["id"] for i in range(4))
        for fold in result["folds"]:
            self.assertFalse(set(fold["validation_batches"]) & set(fold["train_batches"]))
            self.assertEqual(
                set(fold["validation_batches"]) | set(fold["train_batches"]), train
            )
        times = {item["id"]: item["time"] for item in result["batches"]}
        holdout_time = min(times[b] for b in result["holdout"]["batches"])
        train_time = max(times[b] for b in train)
        self.assertGreater(holdout_time, train_time)

    def test_preprocessing_is_fit_inside_training_folds(self):
        result = self.vault.experiments.run("exp")
        chosen = [
            c
            for c in result["candidates"]
            if c["model"] == result["selection"]["model"]
            and c["params"] == result["selection"]["params"]
        ][0]
        # alloy "C" only exists in batch b3; the fold validating b3 never saw it
        fold_b3 = [
            f for f in chosen["folds"] if f["validation_batches"] == ["b3"]
        ][0]
        self.assertGreaterEqual(fold_b3["transform_notes"]["unknown_categories"], 1)
        # leak only has values in the holdout batch: every training fold drops it
        for fold in chosen["folds"]:
            reasons = {
                item["feature"]: item["reason"]
                for item in fold["preprocessing"]["dropped"]
            }
            self.assertEqual(reasons.get("leak"), "all_missing")
        final_reasons = {
            item["feature"]: item["reason"]
            for item in result["preprocessing_final"]["dropped"]
        }
        self.assertEqual(final_reasons["leak"], "all_missing")
        # thickness fill comes from training batches only (b1..b4), never from b5
        thickness = [
            c
            for c in result["preprocessing_final"]["columns"]
            if c["name"] == "thickness"
        ][0]
        train_values = []
        for batch in ["b1", "b2", "b3", "b4"]:
            for row in batch_rows(batch, 1):
                if row["thickness"] is not None:
                    train_values.append(float(row["thickness"]))
        train_values.sort()
        middle = len(train_values) // 2
        expected = (
            train_values[middle]
            if len(train_values) % 2
            else (train_values[middle - 1] + train_values[middle]) / 2
        )
        self.assertEqual(thickness["fill"], expected)

    def test_holdout_is_consumed_exactly_once(self):
        self.vault.experiments.run("exp")
        self.define("second")
        with self.assertRaises(VaultError) as caught:
            self.vault.experiments.run("second")
        self.assertEqual(caught.exception.code, "conflict")
        self.assertEqual(
            caught.exception.details["consumed"],
            [{"batch": "b5", "experiment": "exp"}],
        )
        # a new data version frees a different holdout
        rows = self.standard_rows() + batch_rows("b6", 6)
        self.ingest(rows, request="more", expected=1)
        self.define("third", version=2)
        result = self.vault.experiments.run("third")
        self.assertEqual(result["holdout"]["batches"], ["b6"])

    def test_run_requires_active_input(self):
        self.vault.registry.withdraw("specimens", 1, "defect found")
        with self.assertRaises(VaultError) as caught:
            self.vault.experiments.run("exp")
        self.assertEqual(caught.exception.code, "lineage")


class DegenerateTests(ExperimentVault):
    def test_fold_without_positives_is_explained(self):
        rows = []
        rows += batch_rows("b1", 1, failures_from=100)
        rows += batch_rows("b2", 2, failures_from=100)
        rows += batch_rows("b3", 3)
        rows += batch_rows("b4", 4)
        rows += batch_rows("b5", 5)
        self.ingest(rows)
        self.define()
        result = self.vault.experiments.run("exp")
        chosen = [c for c in result["candidates"] if c["model"] == "logistic"][0]
        fold0 = [f for f in chosen["folds"] if f["fold"] == 0][0]
        self.assertEqual(fold0["positives"], 0)
        codes = [note["code"] for note in fold0["notes"]]
        self.assertIn("validation_no_positives", codes)
        self.assertTrue(
            any("no positive" in text for text in result["conclusions"])
        )

    def test_single_class_data_is_explained_not_skipped(self):
        rows = []
        for month, batch in enumerate(["b1", "b2", "b3", "b4", "b5"], 1):
            rows += batch_rows(batch, month, failures_from=100)
        self.ingest(rows)
        self.define()
        result = self.vault.experiments.run("exp")
        self.assertEqual(result["status"], "single_class")
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["selection"]["model"], "baseline")
        self.assertEqual(
            result["holdout"]["confusion"], {"tp": 0, "fp": 0, "tn": 8, "fn": 0}
        )
        self.assertIsNone(result["holdout"]["recall"])
        self.assertTrue(
            any("single class" in text for text in result["conclusions"])
        )
        self.assertTrue(
            any("constant" in text for text in result["conclusions"])
        )


class PredictTests(ExperimentVault):
    def setUp(self):
        super().setUp()
        self.ingest(self.standard_rows())
        self.define()
        self.result = self.vault.experiments.run("exp")

    def test_schema_validation_of_scoring_rows(self):
        base = {"thickness": "5", "pressure": "55", "alloy": "A", "leak": None}
        for row in [
            {k: v for k, v in base.items() if k != "pressure"},
            {**base, "unknown": 1},
            {**base, "pressure": "not-a-number"},
            {**base, "alloy": 3},
        ]:
            with self.subTest(row=row):
                with self.assertRaises(VaultError) as caught:
                    self.vault.experiments.predict("exp", rows=[row])
                self.assertEqual(caught.exception.code, "schema")
        with self.assertRaises(VaultError) as caught:
            self.vault.experiments.predict("exp")
        self.assertEqual(caught.exception.code, "input")

    def test_prediction_is_label_free_and_explained(self):
        out = self.vault.experiments.predict(
            "exp",
            rows=[
                {"thickness": "9.5", "pressure": "95", "alloy": "Z", "leak": "1"},
                {"thickness": None, "pressure": "5", "alloy": "A", "leak": None},
            ],
        )
        self.assertFalse(out["label_used"])
        self.assertEqual(out["schema_check"]["status"], "ok")
        self.assertEqual(out["total"], 2)
        first, second = out["predictions"]
        self.assertIn("unknown_category", [n["code"] for n in first["notes"]])
        self.assertIn("imputed", [n["code"] for n in second["notes"]])
        self.assertTrue(first["predicted"])
        self.assertFalse(second["predicted"])
        self.assertEqual(
            first["explanation"]["type"],
            {"logistic": "linear_contributions", "tree": "decision_path", "baseline": "constant"}[
                self.result["selection"]["model"]
            ],
        )
        if self.result["selection"]["model"] == "logistic":
            explanation = first["explanation"]
            logit = explanation["intercept"] + sum(
                item["contribution"] for item in explanation["contributions"]
            )
            sigmoid = 1 / (1 + math.exp(-logit))
            self.assertAlmostEqual(sigmoid, first["score"], places=9)

    def test_prediction_ignores_label_even_when_present(self):
        out = self.vault.experiments.predict(
            "exp",
            rows=[
                {
                    "thickness": "1",
                    "pressure": "10",
                    "alloy": "A",
                    "leak": None,
                    "failed": True,
                }
            ],
        )
        self.assertFalse(out["label_used"])
        self.assertFalse(out["predictions"][0]["predicted"])

    def test_predict_on_vault_dataset_version(self):
        scoring_schema = specimen_schema()
        scoring_schema["id"] = "scoring"
        scoring_schema["fields"] = [
            field if field["name"] != "failed" else {**field, "required": False}
            for field in scoring_schema["fields"]
        ]
        self.vault.registry.add_schema(scoring_schema)
        self.vault.registry.register("scoring", "Unlabeled scoring", "scoring", "lab")
        rows = []
        for row in batch_rows("nb", 6):
            del row["failed"]
            rows.append(row)
        self.vault.ingestion.ingest(
            "score-ingest",
            "scoring",
            1,
            content(rows),
            "jsonl",
            "internal",
            ["research"],
        )
        out = self.vault.experiments.predict("exp", dataset="scoring", version=1)
        self.assertEqual(out["input"], {"dataset": "scoring", "version": 1})
        self.assertEqual(out["total"], 8)
        self.assertFalse(out["label_used"])

    def test_predict_rejects_incompatible_schema(self):
        other = {
            "id": "other",
            "version": 1,
            "fields": [
                {"name": "id", "type": "text"},
                {"name": "pressure", "type": "decimal"},
            ],
            "primary_key": ["id"],
        }
        self.vault.registry.add_schema(other)
        self.vault.registry.register("other", "Incompatible", "other", "lab")
        self.vault.ingestion.ingest(
            "other-ingest",
            "other",
            1,
            content([{"id": "x", "pressure": "10"}]),
            "jsonl",
            "internal",
            ["research"],
        )
        with self.assertRaises(VaultError) as caught:
            self.vault.experiments.predict("exp", dataset="other", version=1)
        self.assertEqual(caught.exception.code, "schema")

    def test_predict_requires_run_and_valid_product(self):
        self.define("fresh", version=1)
        with self.assertRaises(VaultError) as caught:
            self.vault.experiments.predict("fresh", rows=[])
        self.assertEqual(caught.exception.code, "state")
        self.vault.registry.withdraw("specimens", 1, "contaminated")
        with self.assertRaises(VaultError) as caught:
            self.vault.experiments.predict(
                "exp",
                rows=[{"thickness": "1", "pressure": "10", "alloy": "A", "leak": None}],
            )
        self.assertEqual(caught.exception.code, "lineage")


class ValidityTests(ExperimentVault):
    def setUp(self):
        super().setUp()
        self.ingest(self.standard_rows())
        self.define()
        self.vault.experiments.run("exp")

    def test_withdrawal_invalidates_product(self):
        validity = self.vault.experiments.validity("exp")
        self.assertEqual(validity["state"], "valid")
        self.vault.registry.withdraw("specimens", 1, "source recalled")
        validity = self.vault.experiments.validity("exp")
        self.assertEqual(validity["state"], "invalid")
        self.assertEqual(validity["problems"][0]["code"], "input_withdrawn")
        self.assertEqual(self.vault.experiments.get("exp")["validity"]["state"], "invalid")

    def test_newer_version_marks_product_superseded(self):
        rows = self.standard_rows() + batch_rows("b6", 6)
        self.ingest(rows, request="more", expected=1)
        validity = self.vault.experiments.validity("exp")
        self.assertEqual(validity["state"], "valid")
        self.assertTrue(validity["superseded"])
        self.assertEqual(validity["notes"][0]["latest_version"], 2)
        listed = self.vault.experiments.list()["experiments"][0]
        self.assertTrue(listed["superseded"])

    def test_model_artifact_is_protected_and_checked(self):
        self.assertEqual(self.vault.checks.unused_artifacts()["candidates"], [])
        inspected = self.vault.checks.inspect()
        self.assertEqual(inspected["experiment_runs"], 1)
        run = self.vault.database.one("SELECT * FROM experiment_runs WHERE experiment='exp'")
        artifact = self.vault.artifacts.path(run["model_artifact"])
        artifact.write_bytes(b"tampered")
        with self.assertRaises(VaultError) as caught:
            self.vault.checks.inspect()
        self.assertEqual(caught.exception.code, "integrity")
        validity = self.vault.experiments.validity("exp")
        self.assertEqual(validity["state"], "invalid")
        self.assertEqual(validity["problems"][0]["code"], "model_artifact_unavailable")


if __name__ == "__main__":
    unittest.main()
