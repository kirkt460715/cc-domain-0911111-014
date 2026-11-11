from collections import Counter
from ..validation import decimal, decimal_text, canonical, integer


def profile_rows(schema, rows):
    result = {}
    for field in schema.fields:
        present = [row[field.name] for row in rows if field.name in row]
        nonnull = [value for value in present if value is not None]
        info = {
            "type": field.type,
            "rows": len(rows),
            "missing": len(rows) - len(present),
            "nulls": len(present) - len(nonnull),
            "distinct": len({canonical(value) for value in nonnull}),
        }
        if field.type in {"integer", "decimal"} and nonnull:
            numbers = [decimal(value) for value in nonnull]
            info.update(
                minimum=decimal_text(min(numbers)),
                maximum=decimal_text(max(numbers)),
                mean=decimal_text(sum(numbers) / len(numbers)),
            )
        elif field.type in {"text", "enum"} and nonnull:
            info.update(
                minimum_length=min(len(value) for value in nonnull),
                maximum_length=max(len(value) for value in nonnull),
            )
            if field.type == "enum":
                info["frequencies"] = dict(sorted(Counter(nonnull).items()))
        elif field.type == "boolean":
            info["true"] = sum(value is True for value in nonnull)
            info["false"] = sum(value is False for value in nonnull)
        elif field.type == "timestamp" and nonnull:
            info.update(earliest=min(nonnull), latest=max(nonnull))
        result[field.name] = info
    return {"rows": len(rows), "fields": result}


def compare_rows(schema, before, after, limit=1000):
    integer(limit, 1, 100000)
    old = {schema.row_key(row): row for row in before}
    new = {schema.row_key(row): row for row in after}
    added = sorted(new.keys() - old.keys())
    removed = sorted(old.keys() - new.keys())
    changed = []
    for key in sorted(old.keys() & new.keys()):
        fields = []
        for field in sorted(old[key].keys() | new[key].keys()):
            if (field in old[key]) != (field in new[key]) or old[key].get(field) != new[
                key
            ].get(field):
                fields.append(
                    {
                        "field": field,
                        "before_present": field in old[key],
                        "after_present": field in new[key],
                        "before": old[key].get(field),
                        "after": new[key].get(field),
                    }
                )
        if fields:
            changed.append({"key": key, "fields": fields})
    return {
        "added_count": len(added),
        "removed_count": len(removed),
        "changed_count": len(changed),
        "added": [new[key] for key in added[:limit]],
        "removed": [old[key] for key in removed[:limit]],
        "changed": changed[:limit],
        "truncated": any(
            count > limit for count in [len(added), len(removed), len(changed)]
        ),
    }
