from ..validation import (
    name,
    object_value,
    list_value,
    canonical,
    decimal,
    decimal_text,
    reject,
)


class Derivations:
    def __init__(self, registry, ingestion):
        self.registry = registry
        self.ingestion = ingestion

    def project(
        self,
        request_id,
        source,
        source_version,
        target,
        schema_version,
        columns,
        filters,
        classification,
        scopes,
        expected_version=0,
    ):
        object_value(columns)
        if not columns or len(columns) > 256:
            reject("recipe", "projection needs bounded column map")
        for destination, origin in columns.items():
            name(destination)
            name(origin)
        predicates = []
        for predicate in list_value(filters, 0, 100):
            object_value(
                predicate,
                {"field", "operator", "value"},
                {"field", "operator", "value"},
            )
            if predicate["operator"] not in {
                "eq",
                "ne",
                "in",
                "is_null",
                "gt",
                "ge",
                "lt",
                "le",
            }:
                reject("recipe", "unknown filter operator")
            name(predicate["field"])
            predicates.append(predicate)
        source_record = self.registry.version(source, source_version)
        source_dataset = self.registry.dataset(source)
        contract = self.registry.schema(
            source_dataset["schema_id"], source_record["schema_version"]
        )
        for origin in columns.values():
            if origin not in contract.by_name:
                reject("recipe", "projection source field absent", field=origin)
        for predicate in predicates:
            if predicate["field"] not in contract.by_name:
                reject("recipe", "filter field absent")
        selected = []
        for row in self.ingestion.rows(source, source_version):
            if all(matches(row, predicate, contract) for predicate in predicates):
                selected.append(
                    {
                        destination: row[origin]
                        for destination, origin in columns.items()
                        if origin in row
                    }
                )
        recipe = {"kind": "projection", "columns": columns, "filters": predicates}
        content = (
            "\n".join(canonical(row) for row in selected) + ("\n" if selected else "")
        ).encode()
        return self.ingestion.ingest(
            request_id,
            target,
            schema_version,
            content,
            "jsonl",
            classification,
            scopes,
            [{"dataset": source, "version": source_version}],
            {"recipe": recipe},
            expected_version,
        )

    def join(
        self,
        request_id,
        left,
        left_version,
        right,
        right_version,
        target,
        schema_version,
        left_on,
        right_on,
        columns,
        mode,
        classification,
        scopes,
        expected_version=0,
    ):
        if mode not in {"inner", "left"}:
            reject("recipe", "join mode must be inner or left")
        list_value(left_on, 1, 16)
        list_value(right_on, 1, 16)
        if len(left_on) != len(right_on):
            reject("recipe", "join key arity differs")
        left_data = self.registry.version(left, left_version)
        right_data = self.registry.version(right, right_version)
        left_schema = self.registry.schema(
            self.registry.dataset(left)["schema_id"], left_data["schema_version"]
        )
        right_schema = self.registry.schema(
            self.registry.dataset(right)["schema_id"], right_data["schema_version"]
        )
        for a, b in zip(left_on, right_on):
            if a not in left_schema.by_name or b not in right_schema.by_name:
                reject("recipe", "join key field absent")
            if left_schema.by_name[a].type != right_schema.by_name[b].type:
                reject("recipe", "join key types differ")
        object_value(columns)
        for output, selection in columns.items():
            name(output)
            object_value(selection, {"side", "field"}, {"side", "field"})
            if selection["side"] not in {"left", "right"}:
                reject("recipe", "unknown join side")
            schema = left_schema if selection["side"] == "left" else right_schema
            if selection["field"] not in schema.by_name:
                reject("recipe", "join selected field absent")
        indexed = {}
        for row in self.ingestion.rows(right, right_version):
            key_values = [row.get(field) for field in right_on]
            if any(value is None for value in key_values):
                continue
            key = canonical(key_values)
            if key in indexed:
                reject("recipe", "right join key is not unique")
            indexed[key] = row
        result = []
        for left_row in self.ingestion.rows(left, left_version):
            key_values = [left_row.get(field) for field in left_on]
            right_row = (
                indexed.get(canonical(key_values))
                if all(value is not None for value in key_values)
                else None
            )
            if right_row is None and mode == "inner":
                continue
            row = {}
            for output, selection in columns.items():
                source = left_row if selection["side"] == "left" else right_row
                row[output] = (
                    source.get(selection["field"]) if source is not None else None
                )
            result.append(row)
        content = (
            "\n".join(canonical(row) for row in result) + ("\n" if result else "")
        ).encode()
        recipe = {
            "kind": "join",
            "left_on": left_on,
            "right_on": right_on,
            "columns": columns,
            "mode": mode,
        }
        parents = [
            {"dataset": left, "version": left_version},
            {"dataset": right, "version": right_version},
        ]
        return self.ingestion.ingest(
            request_id,
            target,
            schema_version,
            content,
            "jsonl",
            classification,
            scopes,
            parents,
            {"recipe": recipe},
            expected_version,
        )

    def aggregate(
        self,
        request_id,
        source,
        source_version,
        target,
        schema_version,
        group_by,
        measures,
        classification,
        scopes,
        expected_version=0,
    ):
        list_value(group_by, 1, 16)
        if len(set(group_by)) != len(group_by):
            reject("recipe", "duplicate grouping column")
        source_record = self.registry.version(source, source_version)
        schema = self.registry.schema(
            self.registry.dataset(source)["schema_id"], source_record["schema_version"]
        )
        for field in group_by:
            if field not in schema.by_name:
                reject("recipe", "grouping field missing")
        parsed = []
        for item in list_value(measures, 1, 64):
            object_value(
                item, {"output", "operation", "field"}, {"output", "operation"}
            )
            name(item["output"])
            if item["operation"] not in {"count", "sum", "minimum", "maximum", "mean"}:
                reject("recipe", "unsupported aggregation")
            if item["operation"] != "count":
                field = schema.by_name.get(item.get("field"))
                if field is None or field.type not in {"integer", "decimal"}:
                    reject("recipe", "numeric aggregation needs numeric field")
            parsed.append(item)
        outputs = [item["output"] for item in parsed]
        if len(set(outputs)) != len(outputs) or set(outputs) & set(group_by):
            reject("recipe", "aggregate output names collide")
        groups = {}
        for row in self.ingestion.rows(source, source_version):
            values = [row.get(field) for field in group_by]
            groups.setdefault(canonical(values), (values, []))[1].append(row)
        result = []
        for key, (values, rows) in sorted(groups.items()):
            row = dict(zip(group_by, values))
            for measure in parsed:
                operation = measure["operation"]
                if operation == "count":
                    value = len(rows)
                else:
                    numbers = [
                        decimal(item[measure["field"]])
                        for item in rows
                        if item.get(measure["field"]) is not None
                    ]
                    if not numbers:
                        value = None
                    elif operation == "sum":
                        value = decimal_text(sum(numbers))
                    elif operation == "minimum":
                        value = decimal_text(min(numbers))
                    elif operation == "maximum":
                        value = decimal_text(max(numbers))
                    else:
                        value = decimal_text(sum(numbers) / len(numbers))
                row[measure["output"]] = value
            result.append(row)
        content = (
            "\n".join(canonical(row) for row in result) + ("\n" if result else "")
        ).encode()
        recipe = {"kind": "aggregate", "group_by": group_by, "measures": measures}
        return self.ingestion.ingest(
            request_id,
            target,
            schema_version,
            content,
            "jsonl",
            classification,
            scopes,
            [{"dataset": source, "version": source_version}],
            {"recipe": recipe},
            expected_version,
        )


def matches(row, predicate, schema):
    field = predicate["field"]
    operation = predicate["operator"]
    expected = predicate["value"]
    value = row.get(field)
    if operation == "is_null":
        if not isinstance(expected, bool):
            reject("recipe", "is_null value must be boolean")
        return (value is None) == expected
    if operation == "in":
        list_value(expected, 0, 10000)
        return value in expected
    if operation == "eq":
        return value == expected
    if operation == "ne":
        return value != expected
    if value is None:
        return False
    if schema.by_name[field].type in {"integer", "decimal"}:
        value, expected = decimal(value), decimal(expected)
    elif not isinstance(expected, str) or not isinstance(value, str):
        reject("recipe", "ordered filter requires comparable scalar")
    return {
        "gt": lambda: value > expected,
        "ge": lambda: value >= expected,
        "lt": lambda: value < expected,
        "le": lambda: value <= expected,
    }[operation]()
