import pickle
from ..validation import (
    VaultError,
    name,
    object_value,
    list_value,
    integer,
    decimal,
    decimal_text,
    canonical,
    digest,
    parse_json,
    reject,
)
from .definition import parse_definition, plan_split, FEATURE_KINDS, MODEL_ORDER
from .preprocessing import (
    fit_preprocessing,
    transform_matrix,
    transform_row,
    summarize,
)
from .metrics import (
    confusion_at,
    best_threshold,
    pr_data,
    expected_cost,
    class_notes,
)


def _models():
    try:
        from . import models
    except ImportError:
        reject("dependency", "analysis dependencies are not installed")
    return models


class Experiments:
    def __init__(self, database, registry, ingestion, artifacts):
        self.db = database
        self.registry = registry
        self.ingestion = ingestion
        self.artifacts = artifacts
        with database.transaction() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS experiments(id TEXT PRIMARY KEY,definition TEXT NOT NULL,digest TEXT NOT NULL)"
            )
            con.execute(
                "CREATE TABLE IF NOT EXISTS experiment_runs(experiment TEXT PRIMARY KEY,result TEXT NOT NULL,digest TEXT NOT NULL,model_artifact TEXT NOT NULL)"
            )
            con.execute(
                "CREATE TABLE IF NOT EXISTS holdout_consumptions(dataset TEXT NOT NULL,version INTEGER NOT NULL,batch TEXT NOT NULL,experiment TEXT NOT NULL,PRIMARY KEY(dataset,version,batch))"
            )

    def define(self, experiment_id, definition):
        name(experiment_id)
        object_value(definition, None, {"dataset", "version"})
        record = self.registry.version(definition["dataset"], definition["version"])
        if record["state"] != "active":
            reject("lineage", "cannot define an experiment on a withdrawn version")
        dataset = self.registry.dataset(record["dataset"])
        schema = self.registry.schema(dataset["schema_id"], record["schema_version"])
        parsed = parse_definition(definition, schema)
        rows = self.ingestion.rows(parsed["dataset"], parsed["version"])
        labeled, unlabeled = self._labeled_rows(schema, parsed, rows)
        if not labeled:
            reject("experiment", "no labeled rows available for the experiment")
        split = plan_split(
            labeled,
            parsed["batch"],
            parsed["time"],
            parsed["holdout_batches"],
            parsed["folds"],
        )
        parsed["folds"] = len(split["folds"])
        normalized = {**parsed, "split": split, "unlabeled_rows": unlabeled}
        checksum = digest(normalized)
        with self.db.transaction() as con:
            old = self.db.one(
                "SELECT digest FROM experiments WHERE id=?", (experiment_id,)
            )
            if old:
                if old["digest"] != checksum:
                    reject("conflict", "experiment definition is immutable")
                return {
                    "id": experiment_id,
                    "existing": True,
                    "definition": normalized,
                    "digest": checksum,
                }
            con.execute(
                "INSERT INTO experiments VALUES(?,?,?)",
                (experiment_id, canonical(normalized), checksum),
            )
            self.db.audit(
                "experiment_defined",
                experiment_id,
                {
                    "digest": checksum,
                    "dataset": parsed["dataset"],
                    "version": parsed["version"],
                },
            )
        return {
            "id": experiment_id,
            "existing": False,
            "definition": normalized,
            "digest": checksum,
        }

    def get(self, experiment_id):
        row = self.db.one("SELECT * FROM experiments WHERE id=?", (experiment_id,))
        if row is None:
            reject("missing", "experiment not found")
        definition = parse_json(row["definition"])
        if digest(definition) != row["digest"]:
            reject("integrity", "experiment definition digest mismatch")
        run = self.db.one(
            "SELECT * FROM experiment_runs WHERE experiment=?", (experiment_id,)
        )
        result = None
        if run:
            result = parse_json(run["result"])
            if digest(result) != run["digest"]:
                reject("integrity", "experiment run digest mismatch")
        return {
            "id": experiment_id,
            "definition": definition,
            "digest": row["digest"],
            "run": result,
            "validity": self.validity(experiment_id),
        }

    def list(self):
        result = []
        for row in self.db.rows("SELECT id FROM experiments ORDER BY id"):
            run = self.db.one(
                "SELECT experiment FROM experiment_runs WHERE experiment=?",
                (row["id"],),
            )
            validity = self.validity(row["id"])
            result.append(
                {
                    "id": row["id"],
                    "has_run": run is not None,
                    "state": validity["state"],
                    "superseded": validity["superseded"],
                }
            )
        return {"experiments": result}

    def validity(self, experiment_id):
        row = self.db.one(
            "SELECT definition FROM experiments WHERE id=?", (experiment_id,)
        )
        if row is None:
            reject("missing", "experiment not found")
        definition = parse_json(row["definition"])
        problems = []
        record = self.registry.version(definition["dataset"], definition["version"])
        if record["state"] != "active":
            problems.append(
                {
                    "code": "input_withdrawn",
                    "dataset": definition["dataset"],
                    "version": definition["version"],
                    "detail": "the training input version is withdrawn",
                }
            )
        run = self.db.one(
            "SELECT model_artifact FROM experiment_runs WHERE experiment=?",
            (experiment_id,),
        )
        if run:
            try:
                self.artifacts.verify(run["model_artifact"])
            except VaultError as exc:
                problems.append(
                    {"code": "model_artifact_unavailable", "detail": exc.code}
                )
        notes = []
        versions = self.registry.list_versions(definition["dataset"])
        latest = versions[-1]["version"] if versions else definition["version"]
        superseded = latest > definition["version"]
        if superseded:
            notes.append(
                {
                    "code": "superseded",
                    "detail": "a newer version of the input dataset exists",
                    "latest_version": latest,
                }
            )
        return {
            "id": experiment_id,
            "state": "invalid" if problems else "valid",
            "problems": problems,
            "notes": notes,
            "superseded": superseded,
        }

    def run(self, experiment_id):
        name(experiment_id)
        row = self.db.one("SELECT * FROM experiments WHERE id=?", (experiment_id,))
        if row is None:
            reject("missing", "experiment not found")
        definition = parse_json(row["definition"])
        if digest(definition) != row["digest"]:
            reject("integrity", "experiment definition digest mismatch")
        with self.db.transaction() as con:
            old = self.db.one(
                "SELECT * FROM experiment_runs WHERE experiment=?", (experiment_id,)
            )
            if old:
                result = parse_json(old["result"])
                if digest(result) != old["digest"]:
                    reject("integrity", "experiment run digest mismatch")
                return {**result, "reused": True}
            record = self.registry.version(
                definition["dataset"], definition["version"]
            )
            if record["state"] != "active":
                reject("lineage", "experiment input version is withdrawn")
            consumed = []
            for batch in definition["split"]["holdout_batches"]:
                owner = self.db.one(
                    "SELECT experiment FROM holdout_consumptions WHERE dataset=? AND version=? AND batch=?",
                    (
                        definition["dataset"],
                        definition["version"],
                        canonical(batch),
                    ),
                )
                if owner:
                    consumed.append(
                        {"batch": batch, "experiment": owner["experiment"]}
                    )
            if consumed:
                reject(
                    "conflict",
                    "holdout batches were already consumed by another experiment",
                    consumed=consumed,
                )
            models = _models()
            schema = self.registry.schema(
                self.registry.dataset(definition["dataset"])["schema_id"],
                record["schema_version"],
            )
            rows = self.ingestion.rows(definition["dataset"], definition["version"])
            labeled, unlabeled = self._labeled_rows(schema, definition, rows)
            result, payload = self._execute(
                models, experiment_id, definition, labeled, unlabeled
            )
            artifact = self.artifacts.put(pickle.dumps(payload))
            result["model_artifact"] = artifact["sha256"]
            for batch in definition["split"]["holdout_batches"]:
                con.execute(
                    "INSERT INTO holdout_consumptions VALUES(?,?,?,?)",
                    (
                        definition["dataset"],
                        definition["version"],
                        canonical(batch),
                        experiment_id,
                    ),
                )
            con.execute(
                "INSERT INTO experiment_runs VALUES(?,?,?,?)",
                (experiment_id, canonical(result), digest(result), artifact["sha256"]),
            )
            self.db.audit(
                "experiment_run",
                experiment_id,
                {
                    "model": result["selection"]["model"],
                    "status": result["status"],
                    "result_digest": digest(result),
                },
            )
        return {**result, "reused": False}

    def predict(self, experiment_id, dataset=None, version=None, rows=None, limit=10000):
        name(experiment_id)
        limit = integer(limit, 1, 10000)
        row = self.db.one("SELECT * FROM experiments WHERE id=?", (experiment_id,))
        if row is None:
            reject("missing", "experiment not found")
        definition = parse_json(row["definition"])
        run = self.db.one(
            "SELECT * FROM experiment_runs WHERE experiment=?", (experiment_id,)
        )
        if run is None:
            reject("state", "experiment has not been run")
        result = parse_json(run["result"])
        if digest(result) != run["digest"]:
            reject("integrity", "experiment run digest mismatch")
        validity = self.validity(experiment_id)
        if validity["state"] != "valid":
            reject(
                "lineage",
                "experiment product is no longer valid",
                problems=validity["problems"],
            )
        artifact = pickle.loads(self.artifacts.read(run["model_artifact"]))
        if artifact["definition_digest"] != row["digest"]:
            reject("integrity", "model artifact does not match the experiment")
        models = _models()
        if (dataset is None) != (version is None):
            reject("input", "dataset and version must be given together")
        if rows is not None and dataset is not None:
            reject("input", "give either a dataset version or explicit rows")
        specs = artifact["feature_specs"]
        feature_names = set(artifact["features"])
        if dataset is not None:
            name(dataset)
            version = integer(version, 1)
            record = self.registry.version(dataset, version)
            if record["state"] != "active":
                reject("lineage", "scoring input version is withdrawn")
            scoring_schema = self.registry.schema(
                self.registry.dataset(dataset)["schema_id"],
                record["schema_version"],
            )
            for spec in specs:
                field = scoring_schema.by_name.get(spec["name"])
                if field is None:
                    reject(
                        "schema",
                        "scoring data is missing a feature field",
                        field=spec["name"],
                    )
                if FEATURE_KINDS.get(field.type) != spec["kind"]:
                    reject(
                        "schema",
                        "scoring feature type is incompatible",
                        field=spec["name"],
                        expected=spec["kind"],
                        actual=field.type,
                    )
            input_rows = self.ingestion.rows(dataset, version)
            keys = [scoring_schema.row_key(item) for item in input_rows]
            source = {"dataset": dataset, "version": version}
        elif rows is not None:
            list_value(rows, 1, 100000)
            input_rows = []
            for index, item in enumerate(rows, 1):
                object_value(item)
                unknown = set(item) - feature_names - {artifact["label"]}
                if unknown:
                    reject(
                        "schema",
                        "row has fields outside the experiment schema",
                        row=index,
                        fields=sorted(unknown),
                    )
                missing = [
                    feature
                    for feature in artifact["features"]
                    if feature not in item
                ]
                if missing:
                    reject(
                        "schema",
                        "row is missing feature fields",
                        row=index,
                        fields=missing,
                    )
                normalized = {}
                for spec in specs:
                    value = item[spec["name"]]
                    if value is None:
                        normalized[spec["name"]] = None
                    elif spec["kind"] == "numeric":
                        try:
                            normalized[spec["name"]] = decimal_text(decimal(value))
                        except VaultError:
                            reject(
                                "schema",
                                "numeric feature must be exact text or integer",
                                row=index,
                                field=spec["name"],
                            )
                    elif spec["kind"] == "boolean":
                        if not isinstance(value, bool):
                            reject(
                                "schema",
                                "boolean feature must be true or false",
                                row=index,
                                field=spec["name"],
                            )
                        normalized[spec["name"]] = value
                    else:
                        if not isinstance(value, str):
                            reject(
                                "schema",
                                "categorical feature must be text",
                                row=index,
                                field=spec["name"],
                            )
                        normalized[spec["name"]] = value
                input_rows.append(normalized)
            keys = list(range(1, len(input_rows) + 1))
            source = {"rows": len(input_rows)}
        else:
            reject("input", "prediction requires a dataset version or explicit rows")
        vectors = []
        notes_per_row = []
        for input_row in input_rows:
            vector, notes = transform_row(artifact["preprocessing"], input_row)
            vectors.append(vector)
            notes_per_row.append(notes)
        scores = models.predict_scores(artifact["model"], vectors) if vectors else []
        entries = []
        for key, score, notes in zip(keys, scores, notes_per_row):
            entries.append(
                {
                    "key": key,
                    "score": score,
                    "predicted": score >= artifact["threshold"],
                    "notes": notes,
                }
            )
        full_digest = digest(entries)
        returned = entries[:limit]
        for entry, vector in zip(returned, vectors[:limit]):
            entry["explanation"] = models.explain_prediction(
                artifact["model"],
                vector,
                artifact["preprocessing"]["encoded_features"],
            )
        return {
            "experiment": experiment_id,
            "model_artifact": run["model_artifact"],
            "threshold": artifact["threshold"],
            "input": source,
            "schema_check": {"status": "ok", "features_checked": artifact["features"]},
            "label_used": False,
            "label_detail": "the label field is never read; explanations derive only from features and stored parameters",
            "validity": validity,
            "total": len(entries),
            "predictions": returned,
            "truncated": len(entries) > limit,
            "predictions_digest": full_digest,
        }

    def _labeled_rows(self, schema, definition, rows):
        labeled = []
        unlabeled = 0
        for row in rows:
            if row.get(definition["batch"]) is None:
                reject(
                    "experiment",
                    "row is missing batch identity",
                    key=schema.row_key(row),
                )
            if row.get(definition["time"]) is None:
                reject(
                    "experiment",
                    "row is missing batch time",
                    key=schema.row_key(row),
                )
            if row.get(definition["label"]) is None:
                unlabeled += 1
                continue
            labeled.append(row)
        return labeled, unlabeled

    def _execute(self, models, experiment_id, definition, labeled, unlabeled):
        specs = definition["feature_specs"]
        label = definition["label"]
        positive = definition["positive"]
        split = definition["split"]
        fp_cost = float(decimal(definition["costs"]["false_positive"]))
        fn_cost = float(decimal(definition["costs"]["false_negative"]))
        by_batch = {}
        for row in labeled:
            by_batch.setdefault(row[definition["batch"]], []).append(row)

        def labels_of(rows):
            return [1 if row[label] == positive else 0 for row in rows]

        train_rows = [
            row for batch in split["train_batches"] for row in by_batch[batch]
        ]
        holdout_rows = []
        holdout_batch_ids = []
        for batch in split["holdout_batches"]:
            for row in by_batch[batch]:
                holdout_rows.append(row)
                holdout_batch_ids.append(batch)
        base = {
            "experiment": experiment_id,
            "definition_digest": digest(definition),
            "data": {
                "dataset": definition["dataset"],
                "version": definition["version"],
                "labeled_rows": len(labeled),
                "unlabeled_rows": unlabeled,
            },
            "batches": split["batches"],
            "folds": [
                {
                    "index": fold["index"],
                    "validation_batches": fold["validation_batches"],
                    "train_batches": fold["train_batches"],
                }
                for fold in split["folds"]
            ],
            "costs": definition["costs"],
        }
        if len({row[label] for row in labeled}) < 2:
            return self._execute_single_class(
                models, base, definition, train_rows, holdout_rows,
                holdout_batch_ids, labels_of, fp_cost, fn_cost,
            )
        candidates = []
        for candidate in models.candidate_grid(definition["tuning"]):
            pooled_scores = []
            pooled_labels = []
            fold_reports = []
            for fold in split["folds"]:
                fold_train = [
                    row
                    for batch in fold["train_batches"]
                    for row in by_batch[batch]
                ]
                fold_validation = [
                    row
                    for batch in fold["validation_batches"]
                    for row in by_batch[batch]
                ]
                fitted = fit_preprocessing(specs, fold_train)
                train_vectors, _ = transform_matrix(fitted, fold_train)
                validation_vectors, validation_notes = transform_matrix(
                    fitted, fold_validation
                )
                y_train = labels_of(fold_train)
                y_validation = labels_of(fold_validation)
                where = "fold %d" % fold["index"]
                notes = class_notes(y_train, y_validation, where)
                if len(set(y_train)) < 2:
                    model = models.constant_model(y_train)
                else:
                    model = models.fit_model(
                        candidate["model"],
                        candidate["params"],
                        train_vectors,
                        y_train,
                    )
                scores = models.predict_scores(model, validation_vectors)
                pooled_scores.extend(scores)
                pooled_labels.extend(y_validation)
                fold_reports.append(
                    {
                        "fold": fold["index"],
                        "validation_batches": fold["validation_batches"],
                        "rows": len(fold_validation),
                        "positives": sum(y_validation),
                        "negatives": len(y_validation) - sum(y_validation),
                        "notes": notes,
                        "transform_notes": {
                            "imputed_values": sum(
                                1
                                for row_notes in validation_notes
                                for note in row_notes
                                if note["code"] == "imputed"
                            ),
                            "unknown_categories": sum(
                                1
                                for row_notes in validation_notes
                                for note in row_notes
                                if note["code"] == "unknown_category"
                            ),
                        },
                        "preprocessing": summarize(fitted),
                        "scores": scores,
                        "labels": y_validation,
                    }
                )
            best = best_threshold(pooled_scores, pooled_labels, fp_cost, fn_cost)
            for report in fold_reports:
                scores = report.pop("scores")
                labels = report.pop("labels")
                report["confusion"] = confusion_at(scores, labels, best["threshold"])
            candidates.append(
                {
                    "model": candidate["model"],
                    "params": candidate["params"],
                    "cv_expected_cost": best["expected_cost"],
                    "threshold": best["threshold"],
                    "cv_confusion": best["confusion"],
                    "folds": fold_reports,
                }
            )
        order = {kind: index for index, kind in enumerate(MODEL_ORDER)}
        chosen = min(
            candidates,
            key=lambda item: (
                item["cv_expected_cost"],
                order[item["model"]],
                canonical(item["params"]),
            ),
        )
        selection = {
            "model": chosen["model"],
            "params": chosen["params"],
            "cv_expected_cost": chosen["cv_expected_cost"],
            "threshold": chosen["threshold"],
            "rule": "minimum pooled cross-validation expected cost; ties resolved "
            "by model order logistic, tree, baseline, then parameter identity",
        }
        conclusions = []
        fitted_final = fit_preprocessing(specs, train_rows)
        final_vectors, _ = transform_matrix(fitted_final, train_rows)
        y_final = labels_of(train_rows)
        if len(set(y_final)) < 2:
            final_model = models.constant_model(y_final)
            conclusions.append(
                "final training data has a single class; the selected model was "
                "replaced by a constant predictor"
            )
        else:
            final_model = models.fit_model(
                selection["model"], selection["params"], final_vectors, y_final
            )
        for report in chosen["folds"]:
            for note in report["notes"]:
                conclusions.append(
                    "fold %d: %s" % (report["fold"], note["detail"])
                )
        holdout = self._evaluate_holdout(
            models,
            final_model,
            fitted_final,
            split,
            holdout_rows,
            holdout_batch_ids,
            labels_of,
            selection["threshold"],
            fp_cost,
            fn_cost,
        )
        conclusions.extend(holdout.pop("conclusions"))
        conclusions.append(
            "holdout batches %s were evaluated exactly once at the threshold "
            "chosen on cross-validation" % split["holdout_batches"]
        )
        result = {
            **base,
            "status": "complete",
            "candidates": candidates,
            "selection": selection,
            "preprocessing_final": fitted_final,
            "holdout": holdout,
            "conclusions": conclusions,
        }
        payload = {
            "format": 1,
            "definition_digest": digest(definition),
            "features": definition["features"],
            "feature_specs": specs,
            "label": label,
            "positive": positive,
            "preprocessing": fitted_final,
            "model": final_model,
            "threshold": selection["threshold"],
        }
        return result, payload

    def _execute_single_class(
        self,
        models,
        base,
        definition,
        train_rows,
        holdout_rows,
        holdout_batch_ids,
        labels_of,
        fp_cost,
        fn_cost,
    ):
        specs = definition["feature_specs"]
        split = definition["split"]
        fitted_final = fit_preprocessing(specs, train_rows)
        final_model = models.constant_model(labels_of(train_rows))
        train_vectors, _ = transform_matrix(fitted_final, train_rows)
        train_scores = models.predict_scores(final_model, train_vectors)
        best = best_threshold(train_scores, labels_of(train_rows), fp_cost, fn_cost)
        conclusions = [
            "all labeled rows belong to a single class; no classifier comparison "
            "is meaningful",
            "the only defensible predictor is the constant majority predictor",
            "holdout metrics are reported for completeness but cannot demonstrate "
            "generalization to a class that was never observed",
        ]
        holdout = self._evaluate_holdout(
            models,
            final_model,
            fitted_final,
            split,
            holdout_rows,
            holdout_batch_ids,
            labels_of,
            best["threshold"],
            fp_cost,
            fn_cost,
        )
        conclusions.extend(holdout.pop("conclusions"))
        result = {
            **base,
            "status": "single_class",
            "candidates": [],
            "selection": {
                "model": "baseline",
                "params": {},
                "threshold": best["threshold"],
                "cv_expected_cost": None,
                "reason": "a single observed class makes model comparison impossible",
            },
            "preprocessing_final": fitted_final,
            "holdout": holdout,
            "conclusions": conclusions,
        }
        payload = {
            "format": 1,
            "definition_digest": base["definition_digest"],
            "features": definition["features"],
            "feature_specs": specs,
            "label": definition["label"],
            "positive": definition["positive"],
            "preprocessing": fitted_final,
            "model": final_model,
            "threshold": best["threshold"],
        }
        return result, payload

    def _evaluate_holdout(
        self,
        models,
        final_model,
        fitted_final,
        split,
        holdout_rows,
        holdout_batch_ids,
        labels_of,
        threshold,
        fp_cost,
        fn_cost,
    ):
        vectors, _ = transform_matrix(fitted_final, holdout_rows)
        y_holdout = labels_of(holdout_rows)
        scores = models.predict_scores(final_model, vectors)
        confusion = confusion_at(scores, y_holdout, threshold)
        tp, fp, fn = confusion["tp"], confusion["fp"], confusion["fn"]
        conclusions = []
        for note in class_notes(None, y_holdout, "holdout"):
            conclusions.append("holdout: %s" % note["detail"])
        per_batch = []
        offset = 0
        for batch in split["holdout_batches"]:
            count = sum(1 for identity in holdout_batch_ids if identity == batch)
            batch_scores = scores[offset : offset + count]
            batch_labels = y_holdout[offset : offset + count]
            batch_confusion = confusion_at(batch_scores, batch_labels, threshold)
            per_batch.append(
                {
                    "batch": batch,
                    "rows": count,
                    "positives": sum(batch_labels),
                    "negatives": count - sum(batch_labels),
                    "confusion": batch_confusion,
                    "expected_cost": expected_cost(
                        batch_confusion, fp_cost, fn_cost
                    ),
                    "notes": class_notes(
                        None, batch_labels, "holdout batch %s" % batch
                    ),
                }
            )
            offset += count
        return {
            "batches": split["holdout_batches"],
            "rows": len(holdout_rows),
            "positives": sum(y_holdout),
            "negatives": len(y_holdout) - sum(y_holdout),
            "threshold": threshold,
            "confusion": confusion,
            "expected_cost": expected_cost(confusion, fp_cost, fn_cost),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "pr_data": pr_data(scores, y_holdout),
            "per_batch": per_batch,
            "consumed": True,
            "conclusions": conclusions,
        }
