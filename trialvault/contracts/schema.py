from dataclasses import dataclass
from ..validation import (
    name,
    object_value,
    list_value,
    integer,
    decimal,
    decimal_text,
    moment,
    canonical,
    digest,
    reject,
)

TYPES = {"text", "integer", "decimal", "boolean", "timestamp", "enum"}


@dataclass(frozen=True)
class Field:
    name: str
    type: str
    nullable: bool
    required: bool
    minimum: object
    maximum: object
    choices: tuple
    maximum_length: int

    @classmethod
    def parse(cls, data):
        allowed = {
            "name",
            "type",
            "nullable",
            "required",
            "minimum",
            "maximum",
            "choices",
            "maximum_length",
        }
        object_value(data, allowed, {"name", "type"})
        field_name = name(data["name"])
        field_type = data["type"]
        if field_type not in TYPES:
            reject("schema", "unsupported field type", type=field_type)
        nullable = data.get("nullable", False)
        required = data.get("required", True)
        if not isinstance(nullable, bool) or not isinstance(required, bool):
            reject("schema", "required and nullable must be boolean")
        low = data.get("minimum")
        high = data.get("maximum")
        if (low is not None or high is not None) and field_type not in {
            "integer",
            "decimal",
        }:
            reject("schema", "numeric bounds require numeric field")
        low = decimal(low) if low is not None else None
        high = decimal(high) if high is not None else None
        if low is not None and high is not None and low > high:
            reject("schema", "numeric bounds reversed")
        choices = data.get("choices", [])
        list_value(choices, 0, 1000)
        if any(
            not isinstance(value, str) or not value or len(value) > 1024
            for value in choices
        ):
            reject("schema", "enum choices must be bounded strings")
        if len(set(choices)) != len(choices):
            reject("schema", "duplicate enum choice")
        if (field_type == "enum") != bool(choices):
            reject("schema", "only enum fields require choices")
        maximum_length = integer(data.get("maximum_length", 4096), 1, 1000000)
        return cls(
            field_name,
            field_type,
            nullable,
            required,
            low,
            high,
            tuple(choices),
            maximum_length,
        )

    def validate(self, value):
        if value is None:
            if not self.nullable:
                reject("row", "null is not allowed", field=self.name)
            return None
        if self.type in {"text", "enum"}:
            if (
                not isinstance(value, str)
                or len(value) > self.maximum_length
                or "\x00" in value
            ):
                reject("row", "invalid bounded text", field=self.name)
            if self.type == "enum" and value not in self.choices:
                reject("row", "value not in enumeration", field=self.name)
            return value
        if self.type == "boolean":
            if not isinstance(value, bool):
                reject("row", "expected boolean", field=self.name)
            return value
        if self.type == "timestamp":
            return moment(value)
        result = (
            integer(value, -(2**53) + 1, 2**53 - 1)
            if self.type == "integer"
            else decimal(value)
        )
        if self.minimum is not None and result < self.minimum:
            reject("row", "value below minimum", field=self.name)
        if self.maximum is not None and result > self.maximum:
            reject("row", "value above maximum", field=self.name)
        return result if self.type == "integer" else decimal_text(result)

    def as_dict(self):
        return {
            "name": self.name,
            "type": self.type,
            "nullable": self.nullable,
            "required": self.required,
            "minimum": decimal_text(self.minimum) if self.minimum is not None else None,
            "maximum": decimal_text(self.maximum) if self.maximum is not None else None,
            "choices": list(self.choices),
            "maximum_length": self.maximum_length,
        }


class Schema:
    def __init__(self, data):
        object_value(
            data,
            {"id", "version", "fields", "primary_key", "allow_extra"},
            {"id", "version", "fields", "primary_key"},
        )
        self.id = name(data["id"])
        self.version = integer(data["version"], 1, 100000)
        self.fields = [
            Field.parse(value) for value in list_value(data["fields"], 1, 256)
        ]
        self.by_name = {field.name: field for field in self.fields}
        if len(self.by_name) != len(self.fields):
            reject("schema", "duplicate field names")
        self.primary_key = list_value(data["primary_key"], 1, 16)
        if len(set(self.primary_key)) != len(self.primary_key):
            reject("schema", "duplicate primary key field")
        for field in self.primary_key:
            if field not in self.by_name:
                reject("schema", "primary key field absent")
            definition = self.by_name[field]
            if definition.nullable or not definition.required:
                reject("schema", "primary key fields must be required and nonnullable")
        self.allow_extra = data.get("allow_extra", False)
        if not isinstance(self.allow_extra, bool):
            reject("schema", "allow_extra must be boolean")

    def validate(self, row):
        object_value(row)
        extra = row.keys() - self.by_name.keys()
        if extra and not self.allow_extra:
            reject("row", "extra fields prohibited", fields=sorted(extra))
        result = {}
        for field in self.fields:
            if field.name not in row:
                if field.required:
                    reject("row", "required field missing", field=field.name)
                continue
            result[field.name] = field.validate(row[field.name])
        if self.allow_extra:
            for key in extra:
                result[key] = row[key]
            canonical(result)
        return result

    def row_key(self, row):
        return canonical([row[field] for field in self.primary_key])

    def as_dict(self):
        return {
            "id": self.id,
            "version": self.version,
            "fields": [field.as_dict() for field in self.fields],
            "primary_key": self.primary_key,
            "allow_extra": self.allow_extra,
        }

    @property
    def fingerprint(self):
        return digest(self.as_dict())
