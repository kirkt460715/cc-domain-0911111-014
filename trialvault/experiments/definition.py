from ..validation import (
    name,
    object_value,
    list_value,
    integer,
    decimal,
    decimal_text,
    reject,
)

# Field types usable as experiment features and how they are encoded.
FEATURE_KINDS = {
    "integer": "numeric",
    "decimal": "numeric",
    "boolean": "boolean",
    "enum": "categorical",
    "text": "categorical",
}

MODEL_ORDER = ["logistic", "tree", "baseline"]

DEFAULT_TUNING = {
    "logistic_C": ["0.1", "1", "10"],
    "tree_max_depth": [2, 4],
    "tree_min_samples_leaf": [1, 5],
}

DEFINITION_FIELDS = {
    "dataset",
    "version",
    "features",
    "label",
    "positive",
    "batch",
    "time",
    "holdout_batches",
    "folds",
    "costs",
    "tuning",
}


def parse_tuning(data):
    if data is None:
        return {key: list(value) for key, value in DEFAULT_TUNING.items()}
    object_value(data, {"logistic_C", "tree_max_depth", "tree_min_samples_leaf"})
    result = {key: list(value) for key, value in DEFAULT_TUNING.items()}
    if "logistic_C" in data:
        values = sorted({decimal(item) for item in list_value(data["logistic_C"], 1, 6)})
        if any(value <= 0 for value in values):
            reject("experiment", "logistic regularization must be positive")
        result["logistic_C"] = [decimal_text(value) for value in values]
    if "tree_max_depth" in data:
        result["tree_max_depth"] = sorted(
            {integer(item, 1, 32) for item in list_value(data["tree_max_depth"], 1, 6)}
        )
    if "tree_min_samples_leaf" in data:
        result["tree_min_samples_leaf"] = sorted(
            {
                integer(item, 1, 1000)
                for item in list_value(data["tree_min_samples_leaf"], 1, 6)
            }
        )
    return result


def parse_definition(data, schema):
    object_value(
        data,
        DEFINITION_FIELDS,
        {"dataset", "version", "features", "label", "batch", "time", "costs"},
    )
    dataset = name(data["dataset"])
    version = integer(data["version"], 1)
    features = list_value(data["features"], 1, 64)
    if len(set(features)) != len(features):
        reject("experiment", "duplicate feature field")
    label = name(data["label"])
    batch = name(data["batch"])
    time = name(data["time"])
    reserved = {label, batch, time}
    if len(reserved) != 3:
        reject("experiment", "label, batch and time fields must differ")
    if reserved & set(features):
        reject("experiment", "label, batch and time fields cannot be features")
    specs = []
    for feature in features:
        name(feature)
        field = schema.by_name.get(feature)
        if field is None:
            reject("experiment", "feature field absent", field=feature)
        kind = FEATURE_KINDS.get(field.type)
        if kind is None:
            reject(
                "experiment",
                "unsupported feature field type",
                field=feature,
                type=field.type,
            )
        specs.append({"name": feature, "kind": kind})
    label_field = schema.by_name.get(label)
    if label_field is None:
        reject("experiment", "label field absent")
    if label_field.type == "boolean":
        positive = data.get("positive", True)
        if positive is not True:
            reject("experiment", "boolean label positive class must be true")
    elif label_field.type == "enum":
        if len(label_field.choices) != 2:
            reject("experiment", "enum label needs exactly two choices")
        if "positive" not in data:
            reject("experiment", "enum label requires an explicit positive class")
        positive = data["positive"]
        if positive not in label_field.choices:
            reject("experiment", "positive class is not a label choice")
    else:
        reject("experiment", "label field must be boolean or two-valued enum")
    batch_field = schema.by_name.get(batch)
    if batch_field is None or batch_field.type not in {"text", "enum", "integer"}:
        reject("experiment", "batch field must be text, enum or integer")
    time_field = schema.by_name.get(time)
    if time_field is None or time_field.type != "timestamp":
        reject("experiment", "time field must be a timestamp")
    costs = object_value(
        data["costs"],
        {"false_positive", "false_negative"},
        {"false_positive", "false_negative"},
    )
    parsed_costs = {}
    for key in ("false_positive", "false_negative"):
        value = decimal(costs[key])
        if value <= 0:
            reject("experiment", "misclassification costs must be positive")
        parsed_costs[key] = decimal_text(value)
    holdout_batches = integer(data.get("holdout_batches", 1), 1, 1000)
    folds = data.get("folds")
    if folds is not None:
        folds = integer(folds, 2, 10)
    return {
        "dataset": dataset,
        "version": version,
        "features": list(features),
        "feature_specs": specs,
        "label": label,
        "positive": positive,
        "batch": batch,
        "time": time,
        "costs": parsed_costs,
        "holdout_batches": holdout_batches,
        "folds": folds,
        "tuning": parse_tuning(data.get("tuning")),
    }


def plan_split(labeled_rows, batch_field, time_field, holdout_batches, requested_folds):
    batches = {}
    for row in labeled_rows:
        identity = row[batch_field]
        moment = row[time_field]
        entry = batches.setdefault(identity, {"id": identity, "time": moment, "rows": 0})
        entry["rows"] += 1
        if moment < entry["time"]:
            entry["time"] = moment
    ordered = sorted(batches.values(), key=lambda item: (item["time"], item["id"]))
    if len(ordered) < holdout_batches + 2:
        reject(
            "experiment",
            "not enough batches for a holdout plus two training folds",
            batches=len(ordered),
        )
    holdout = ordered[len(ordered) - holdout_batches :]
    train = ordered[: len(ordered) - holdout_batches]
    if holdout[0]["time"] <= train[-1]["time"]:
        reject(
            "experiment",
            "holdout batches must be strictly later than every training batch",
            latest_training={"batch": train[-1]["id"], "time": train[-1]["time"]},
            earliest_holdout={"batch": holdout[0]["id"], "time": holdout[0]["time"]},
        )
    fold_count = min(requested_folds or 5, len(train))
    folds = []
    total = len(train)
    for index in range(fold_count):
        block = train[index * total // fold_count : (index + 1) * total // fold_count]
        validation = {item["id"] for item in block}
        folds.append(
            {
                "index": index,
                "validation_batches": [item["id"] for item in block],
                "train_batches": [
                    item["id"] for item in train if item["id"] not in validation
                ],
            }
        )
    fold_of = {}
    for fold in folds:
        for identity in fold["validation_batches"]:
            fold_of[identity] = fold["index"]
    holdout_ids = {item["id"] for item in holdout}
    return {
        "batches": [
            {
                "id": item["id"],
                "time": item["time"],
                "rows": item["rows"],
                "role": "holdout" if item["id"] in holdout_ids else "train",
                "fold": fold_of.get(item["id"]),
            }
            for item in ordered
        ],
        "train_batches": [item["id"] for item in train],
        "holdout_batches": [item["id"] for item in holdout],
        "folds": folds,
    }
