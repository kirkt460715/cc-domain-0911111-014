import tempfile
from pathlib import Path
from .service import Vault
from .validation import canonical
from .releases.exporter import verify_directory


def demonstrate():
    with tempfile.TemporaryDirectory() as directory:
        with Vault(Path(directory) / "vault") as vault:
            schema = {
                "id": "measurements",
                "version": 1,
                "fields": [
                    {"name": "id", "type": "text"},
                    {"name": "pressure", "type": "decimal", "minimum": "0"},
                ],
                "primary_key": ["id"],
            }
            vault.registry.add_schema(schema)
            vault.registry.register(
                "trial", "Synthetic trial measurements", "measurements", "operator"
            )
            content = (
                canonical({"id": "sample-A", "pressure": "12.5"})
                + "\n"
                + canonical({"id": "sample-B", "pressure": "13"})
                + "\n"
            ).encode()
            ingested = vault.ingestion.ingest(
                "request", "trial", 1, content, "jsonl", "internal", ["research"]
            )
            vault.releases.plan(
                "release", [{"dataset": "trial", "version": 1}], "research", "internal"
            )
            vault.releases.approve(
                "release", 1, "independent-reviewer", "Synthetic dataset verified"
            )
            exported = vault.exporter.export(
                "export", "release", Path(directory) / "release"
            )
            verified = verify_directory(Path(directory) / "release")
            assert ingested["rows"] == 2
            assert verified["rows"] == 2
            assert exported["state"] == "complete"
            experiment = demonstrate_experiment(vault)
            return {
                "ingestion": ingested,
                "export": exported,
                "verification": verified,
                "experiment": experiment,
                "integrity": vault.checks.inspect(),
            }


def demonstrate_experiment(vault):
    schema = {
        "id": "specimens",
        "version": 1,
        "fields": [
            {"name": "id", "type": "text"},
            {"name": "batch", "type": "text"},
            {"name": "cast_at", "type": "timestamp"},
            {"name": "thickness", "type": "decimal", "nullable": True},
            {"name": "pressure", "type": "decimal"},
            {"name": "alloy", "type": "enum", "choices": ["A", "B"]},
            {"name": "failed", "type": "boolean"},
        ],
        "primary_key": ["id"],
    }
    vault.registry.add_schema(schema)
    vault.registry.register("specimens", "Synthetic specimen trials", "specimens", "lab")
    rows = []
    for month, batch in enumerate(["batch-1", "batch-2", "batch-3", "batch-4"], 1):
        for index, pressure in enumerate([10, 25, 40, 60, 75, 90]):
            rows.append(
                {
                    "id": "%s-%d" % (batch, index),
                    "batch": batch,
                    "cast_at": "2026-%02d-%02dT00:00:00Z" % (month, index + 1),
                    "thickness": None if index == 0 else "%d.2" % (pressure // 10),
                    "pressure": str(pressure),
                    "alloy": "A" if index % 2 == 0 else "B",
                    "failed": pressure >= 50,
                }
            )
    content = ("\n".join(canonical(row) for row in rows) + "\n").encode()
    vault.ingestion.ingest(
        "specimen-ingest", "specimens", 1, content, "jsonl", "internal", ["research"]
    )
    vault.experiments.define(
        "specimen-experiment",
        {
            "dataset": "specimens",
            "version": 1,
            "features": ["thickness", "pressure", "alloy"],
            "label": "failed",
            "batch": "batch",
            "time": "cast_at",
            "costs": {"false_positive": "1", "false_negative": "5"},
        },
    )
    run = vault.experiments.run("specimen-experiment")
    prediction = vault.experiments.predict(
        "specimen-experiment",
        rows=[
            {"thickness": "9.2", "pressure": "95", "alloy": "A"},
            {"thickness": None, "pressure": "15", "alloy": "B"},
        ],
    )
    assert run["status"] == "complete"
    assert prediction["label_used"] is False
    return {
        "status": run["status"],
        "holdout_batches": run["holdout"]["batches"],
        "selection": {
            "model": run["selection"]["model"],
            "cv_expected_cost": run["selection"]["cv_expected_cost"],
            "threshold": run["selection"]["threshold"],
        },
        "holdout_confusion": run["holdout"]["confusion"],
        "per_batch": [
            {"batch": item["batch"], "confusion": item["confusion"]}
            for item in run["holdout"]["per_batch"]
        ],
        "prediction": {
            "total": prediction["total"],
            "label_used": prediction["label_used"],
            "first": prediction["predictions"][0]["predicted"],
            "explanation": prediction["predictions"][0]["explanation"]["type"],
        },
        "validity": vault.experiments.validity("specimen-experiment")["state"],
    }
