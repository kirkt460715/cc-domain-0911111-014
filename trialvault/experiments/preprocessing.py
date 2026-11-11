from statistics import median
from ..validation import decimal

# Categorical features with more distinct values are dropped, not encoded.
MAX_CATEGORIES = 100
MISSING = "<missing>"


def fit_preprocessing(specs, rows):
    """Learn fill values, categories and feature screening from training rows only.

    Every learned value comes exclusively from the rows passed in, so callers
    must pass exactly one training fold (or the final training batch set).
    Columns with no observed value or no variation in the training fold are
    dropped with an explicit, recorded reason.
    """
    columns = []
    dropped = []
    for spec in specs:
        feature = spec["name"]
        kind = spec["kind"]
        values = [row.get(feature) for row in rows]
        observed = [value for value in values if value is not None]
        if not observed:
            dropped.append(
                {
                    "feature": feature,
                    "reason": "all_missing",
                    "detail": "no observed value in the training fold; column dropped",
                }
            )
            continue
        if kind == "numeric":
            numbers = [float(decimal(value)) for value in observed]
            center = median(numbers)
            mean = sum(numbers) / len(numbers)
            variance = sum((number - mean) ** 2 for number in numbers) / len(numbers)
            std = variance**0.5
            if std == 0:
                dropped.append(
                    {
                        "feature": feature,
                        "reason": "constant",
                        "detail": "no variation in the training fold; column dropped",
                    }
                )
                continue
            columns.append(
                {
                    "name": feature,
                    "kind": "numeric",
                    "fill": center,
                    "mean": mean,
                    "std": std,
                }
            )
        elif kind == "boolean":
            if len(set(observed)) == 1:
                dropped.append(
                    {
                        "feature": feature,
                        "reason": "constant",
                        "detail": "no variation in the training fold; column dropped",
                    }
                )
                continue
            fill = sum(1 for value in observed if value) * 2 > len(observed)
            columns.append({"name": feature, "kind": "boolean", "fill": fill})
        else:
            categories = sorted({value for value in observed})
            if len(categories) > MAX_CATEGORIES:
                dropped.append(
                    {
                        "feature": feature,
                        "reason": "cardinality_exceeded",
                        "categories": len(categories),
                        "detail": "more than %d distinct categories in the training fold; column dropped"
                        % MAX_CATEGORIES,
                    }
                )
                continue
            if len(categories) == 1 and len(observed) == len(values):
                dropped.append(
                    {
                        "feature": feature,
                        "reason": "constant",
                        "detail": "no variation in the training fold; column dropped",
                    }
                )
                continue
            columns.append(
                {"name": feature, "kind": "categorical", "categories": categories}
            )
    encoded = []
    for column in columns:
        if column["kind"] == "categorical":
            encoded.extend(
                "%s=%s" % (column["name"], category)
                for category in column["categories"] + [MISSING]
            )
        else:
            encoded.append(column["name"])
    return {"columns": columns, "dropped": dropped, "encoded_features": encoded}


def transform_row(fitted, row):
    """Encode one row with a fitted preprocessing state.

    Missing values use the training-fold fill value; unseen categories become
    an all-zero block. Both situations produce explicit notes.
    """
    vector = []
    notes = []
    for column in fitted["columns"]:
        feature = column["name"]
        value = row.get(feature)
        if column["kind"] == "numeric":
            if value is None:
                notes.append(
                    {
                        "feature": feature,
                        "code": "imputed",
                        "detail": "filled with the training-fold median %s"
                        % column["fill"],
                    }
                )
                number = column["fill"]
            else:
                number = float(decimal(value))
            vector.append((number - column["mean"]) / column["std"])
        elif column["kind"] == "boolean":
            if value is None:
                notes.append(
                    {
                        "feature": feature,
                        "code": "imputed",
                        "detail": "filled with the training-fold majority %s"
                        % column["fill"],
                    }
                )
                value = column["fill"]
            vector.append(1.0 if value else 0.0)
        else:
            block = [0.0] * (len(column["categories"]) + 1)
            if value is None:
                block[-1] = 1.0
            elif value in column["categories"]:
                block[column["categories"].index(value)] = 1.0
            else:
                notes.append(
                    {
                        "feature": feature,
                        "code": "unknown_category",
                        "value": value,
                        "detail": "category never seen in training; encoded as an all-zero block",
                    }
                )
            vector.extend(block)
    return vector, notes


def transform_matrix(fitted, rows):
    vectors = []
    all_notes = []
    for row in rows:
        vector, notes = transform_row(fitted, row)
        vectors.append(vector)
        all_notes.append(notes)
    return vectors, all_notes


def summarize(fitted):
    """Compact, reviewable description of a fitted preprocessing state."""
    return {
        "kept": len(fitted["columns"]),
        "dropped": fitted["dropped"],
        "fills": {
            column["name"]: column["fill"]
            for column in fitted["columns"]
            if column["kind"] in {"numeric", "boolean"}
        },
        "categories": {
            column["name"]: column["categories"]
            for column in fitted["columns"]
            if column["kind"] == "categorical"
        },
    }
