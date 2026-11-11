import numpy as np
from ..validation import decimal

MODEL_ORDER = ["logistic", "tree", "baseline"]


def candidate_grid(tuning):
    grid = [{"model": "logistic", "params": {"C": c}} for c in tuning["logistic_C"]]
    for depth in tuning["tree_max_depth"]:
        for leaf in tuning["tree_min_samples_leaf"]:
            grid.append(
                {
                    "model": "tree",
                    "params": {"max_depth": depth, "min_samples_leaf": leaf},
                }
            )
    grid.append({"model": "baseline", "params": {}})
    return grid


def fit_model(kind, params, matrix, labels):
    X = np.asarray(matrix, dtype=float)
    y = np.asarray(labels, dtype=int)
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            C=float(decimal(params["C"])),
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
        )
        model.fit(X, y)
        return {"kind": kind, "params": params, "object": model}
    if kind == "tree":
        from sklearn.tree import DecisionTreeClassifier

        model = DecisionTreeClassifier(
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=0,
        )
        model.fit(X, y)
        return {"kind": kind, "params": params, "object": model}
    if kind == "baseline":
        return {"kind": kind, "params": {}, "object": _constant(y)}
    raise ValueError("unknown model kind")


def _constant(y):
    prevalence = float(y.mean()) if len(y) else 0.0
    return {"prevalence": prevalence, "majority": int(prevalence >= 0.5)}


def constant_model(labels):
    """Fallback for a training fold that contains a single class."""
    y = np.asarray(labels, dtype=int)
    return {"kind": "baseline", "params": {}, "object": _constant(y), "fallback": True}


def predict_scores(model, matrix):
    X = np.asarray(matrix, dtype=float)
    if model["kind"] == "baseline":
        return [model["object"]["prevalence"]] * len(X)
    classes = list(model["object"].classes_)
    positive = classes.index(1)
    return [float(row[positive]) for row in model["object"].predict_proba(X)]


def explain_prediction(model, vector, encoded_features):
    """Label-free, recomputable explanation of one prediction."""
    kind = model["kind"]
    if kind == "logistic":
        coefficients = model["object"].coef_[0]
        intercept = float(model["object"].intercept_[0])
        contributions = [
            {
                "feature": feature,
                "transformed_value": float(value),
                "coefficient": float(coefficient),
                "contribution": float(coefficient * value),
            }
            for feature, value, coefficient in zip(
                encoded_features, vector, coefficients
            )
        ]
        logit = intercept + sum(item["contribution"] for item in contributions)
        return {
            "type": "linear_contributions",
            "intercept": intercept,
            "logit": logit,
            "contributions": contributions,
            "verification": "score equals sigmoid(intercept + sum(contributions)); no label is read",
        }
    if kind == "tree":
        tree = model["object"].tree_
        node = 0
        path = []
        while tree.children_left[node] != tree.children_right[node]:
            feature = tree.feature[node]
            threshold = float(tree.threshold[node])
            value = float(vector[feature])
            direction = "<=" if value <= threshold else ">"
            path.append(
                {
                    "feature": encoded_features[feature],
                    "threshold": threshold,
                    "transformed_value": value,
                    "direction": direction,
                }
            )
            node = (
                tree.children_left[node]
                if value <= threshold
                else tree.children_right[node]
            )
        counts = tree.value[node][0]
        total = float(counts.sum())
        positives = float(counts[list(model["object"].classes_).index(1)])
        return {
            "type": "decision_path",
            "path": path,
            "leaf": int(node),
            "leaf_samples": int(tree.n_node_samples[node]),
            "leaf_positive_fraction": positives / total if total else None,
            "verification": "follow the path thresholds on transformed values; score equals the leaf positive fraction",
        }
    return {
        "type": "constant",
        "prevalence": model["object"]["prevalence"],
        "verification": "majority-class baseline always returns the training prevalence",
    }
