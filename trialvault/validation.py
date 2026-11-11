import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone


class VaultError(Exception):
    def __init__(self, code, message, **details):
        super().__init__(message)
        self.code = code
        self.details = details

    def as_dict(self):
        return {"code": self.code, "message": str(self), "details": self.details}


def reject(code, message, **details):
    raise VaultError(code, message, **details)


def name(value):
    if not isinstance(value, str) or not re.fullmatch(
        "[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", value
    ):
        reject("input", "invalid bounded identifier")
    return value


def object_value(value, allowed=None, required=()):
    if not isinstance(value, dict):
        reject("input", "expected object")
    if set(required) - value.keys():
        reject(
            "input",
            "missing object fields",
            fields=sorted(set(required) - value.keys()),
        )
    if allowed is not None and value.keys() - set(allowed):
        reject(
            "input", "unknown object fields", fields=sorted(value.keys() - set(allowed))
        )
    return value


def list_value(value, minimum=0, maximum=10000):
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        reject("input", "list length outside allowed range")
    return value


def integer(value, minimum=0, maximum=2**53 - 1):
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        reject("input", "expected bounded integer")
    return value


def decimal(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        reject("input", "decimal must be exact text or integer")
    try:
        result = Decimal(value)
    except InvalidOperation:
        reject("input", "invalid decimal")
    if (
        not result.is_finite()
        or abs(result) > Decimal("1e30")
        or result.as_tuple().exponent < -18
    ):
        reject("input", "unsupported decimal precision or magnitude")
    return result


def decimal_text(value):
    result = format(value, "f")
    return result.rstrip("0").rstrip(".") if "." in result else result


def moment(value):
    if not isinstance(value, str):
        reject("input", "timestamp must be text")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        reject("input", "invalid timestamp")
    if result.tzinfo is None:
        reject("input", "timestamp must include timezone")
    return (
        result.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def parse_json(content):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                reject("input", "duplicate JSON key", key=key)
            result[key] = value
        return result

    def invalid(value):
        reject("input", "nonfinite JSON token", value=value)

    try:
        return json.loads(content, object_pairs_hook=pairs, parse_constant=invalid)
    except (ValueError, UnicodeError) as exc:
        reject("input", "malformed JSON", reason=str(exc))
