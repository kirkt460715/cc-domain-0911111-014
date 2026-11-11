from .schema import Schema
from ..validation import reject


def compatibility(old, new):
    if not isinstance(old, Schema):
        old = Schema(old)
    if not isinstance(new, Schema):
        new = Schema(new)
    if old.id != new.id or new.version <= old.version:
        reject("schema", "evolution requires same schema and increasing version")
    breaking = []
    additions = []
    if old.primary_key != new.primary_key:
        breaking.append({"code": "primary_key_changed"})
    if old.allow_extra and not new.allow_extra:
        breaking.append({"code": "extras_no_longer_allowed"})
    for field_name, previous in old.by_name.items():
        current = new.by_name.get(field_name)
        if current is None:
            if not new.allow_extra:
                breaking.append({"code": "field_removed", "field": field_name})
            continue
        if previous.type != current.type:
            breaking.append({"code": "type_changed", "field": field_name})
        if previous.nullable and not current.nullable:
            breaking.append({"code": "nullability_narrowed", "field": field_name})
        if not previous.required and current.required:
            breaking.append({"code": "field_now_required", "field": field_name})
        if current.minimum is not None and (
            previous.minimum is None or current.minimum > previous.minimum
        ):
            breaking.append({"code": "minimum_narrowed", "field": field_name})
        if current.maximum is not None and (
            previous.maximum is None or current.maximum < previous.maximum
        ):
            breaking.append({"code": "maximum_narrowed", "field": field_name})
        if current.maximum_length < previous.maximum_length:
            breaking.append({"code": "length_narrowed", "field": field_name})
        if previous.type == "enum" and not set(previous.choices).issubset(
            current.choices
        ):
            breaking.append({"code": "enum_narrowed", "field": field_name})
    for field_name, current in new.by_name.items():
        if field_name not in old.by_name:
            additions.append(field_name)
            if current.required:
                breaking.append({"code": "required_field_added", "field": field_name})
    return {
        "schema": old.id,
        "from": old.version,
        "to": new.version,
        "compatible": not breaking,
        "breaking": breaking,
        "added_fields": additions,
    }
