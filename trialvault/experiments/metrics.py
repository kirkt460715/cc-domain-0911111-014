import math

TOLERANCE = 1e-9


def confusion_at(scores, labels, threshold):
    tp = fp = tn = fn = 0
    for score, label in zip(scores, labels):
        if score >= threshold:
            if label:
                tp += 1
            else:
                fp += 1
        elif label:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def expected_cost(confusion, false_positive_cost, false_negative_cost):
    return (
        confusion["fp"] * false_positive_cost + confusion["fn"] * false_negative_cost
    )


def sweep(scores, labels):
    """Every distinct decision boundary for the given scores.

    The decision rule is "positive when score >= threshold". The first point
    predicts everything positive, the last predicts everything negative.
    """
    pairs = sorted(zip(scores, labels), key=lambda pair: (pair[0], pair[1]))
    total_positives = sum(1 for _, label in pairs if label)
    total_negatives = len(pairs) - total_positives
    if not pairs:
        return []
    points = [
        {
            "threshold": pairs[0][0],
            "confusion": {
                "tp": total_positives,
                "fp": total_negatives,
                "tn": 0,
                "fn": 0,
            },
        }
    ]
    fn = tn = 0
    index = 0
    while index < len(pairs):
        follow = index
        while follow < len(pairs) and pairs[follow][0] == pairs[index][0]:
            if pairs[follow][1]:
                fn += 1
            else:
                tn += 1
            follow += 1
        score = pairs[index][0]
        if follow < len(pairs):
            threshold = (score + pairs[follow][0]) / 2
        else:
            threshold = math.nextafter(score, 2.0)
        points.append(
            {
                "threshold": threshold,
                "confusion": {
                    "tp": total_positives - fn,
                    "fp": total_negatives - tn,
                    "tn": tn,
                    "fn": fn,
                },
            }
        )
        index = follow
    return points


def best_threshold(scores, labels, false_positive_cost, false_negative_cost):
    """Lowest expected misclassification cost; ties take the larger threshold."""
    best = None
    for point in sweep(scores, labels):
        cost = expected_cost(
            point["confusion"], false_positive_cost, false_negative_cost
        )
        if (
            best is None
            or cost < best["expected_cost"] - TOLERANCE
            or (
                abs(cost - best["expected_cost"]) <= TOLERANCE
                and point["threshold"] > best["threshold"]
            )
        ):
            best = {
                "threshold": point["threshold"],
                "expected_cost": cost,
                "confusion": point["confusion"],
            }
    return best


def pr_data(scores, labels, cap=1000):
    """Precision/recall at every decision boundary.

    Precision is null when nothing is predicted positive; recall is null when
    the data contains no positives. Nulls are explicit, never skipped.
    """
    points = []
    for point in sweep(scores, labels):
        confusion = point["confusion"]
        tp, fp, fn = confusion["tp"], confusion["fp"], confusion["fn"]
        points.append(
            {
                "threshold": point["threshold"],
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                **confusion,
            }
        )
    if len(points) > cap:
        indices = sorted(
            {round(position * (len(points) - 1) / (cap - 1)) for position in range(cap)}
        )
        points = [points[index] for index in indices]
    return points


def class_notes(train_labels, validation_labels, where):
    """Explicit statements for degenerate class situations."""
    notes = []
    if train_labels is not None and len(set(train_labels)) < 2:
        notes.append(
            {
                "code": "train_single_class",
                "detail": "%s training data has a single class; no discriminative "
                "model can be fitted, so a constant predictor is used" % where,
            }
        )
    if validation_labels is not None:
        if not any(validation_labels):
            notes.append(
                {
                    "code": "validation_no_positives",
                    "detail": "%s has no positive examples; recall is undefined "
                    "there and only false-positive behaviour is measurable" % where,
                }
            )
        if all(validation_labels):
            notes.append(
                {
                    "code": "validation_no_negatives",
                    "detail": "%s has no negative examples; the false-positive "
                    "rate is not measurable there" % where,
                }
            )
    return notes
