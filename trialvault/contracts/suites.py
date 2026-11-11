from decimal import Decimal
from ..validation import (
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


class QualitySuites:
    def __init__(self, database, registry, ingestion, checks):
        self.db = database
        self.registry = registry
        self.ingestion = ingestion
        self.checks = checks
        with database.transaction() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS quality_suites(id TEXT PRIMARY KEY,payload TEXT NOT NULL,digest TEXT NOT NULL)"
            )
            con.execute(
                "CREATE TABLE IF NOT EXISTS quality_runs(id TEXT PRIMARY KEY,request TEXT NOT NULL,result TEXT NOT NULL)"
            )

    def define(self, suite_id, schema_id, schema_version, rules):
        name(suite_id)
        schema = self.registry.schema(schema_id, schema_version)
        parsed = []
        names = set()
        for rule in list_value(rules, 1, 100):
            object_value(rule, required={"id", "kind"})
            rule_id = name(rule["id"])
            if rule_id in names:
                reject("quality", "duplicate quality rule id")
            names.add(rule_id)
            kind = rule["kind"]
            if kind == "row_count":
                object_value(
                    rule,
                    {"id", "kind", "minimum", "maximum"},
                    {"id", "kind", "minimum", "maximum"},
                )
                low = integer(rule["minimum"])
                high = integer(rule["maximum"], low)
                parsed.append(
                    {"id": rule_id, "kind": kind, "minimum": low, "maximum": high}
                )
            elif kind in {"unique", "not_null", "monotonic"}:
                object_value(rule, {"id", "kind", "fields"}, {"id", "kind", "fields"})
                fields = list_value(rule["fields"], 1, 16)
                if len(set(fields)) != len(fields) or any(
                    field not in schema.by_name for field in fields
                ):
                    reject("quality", "invalid quality rule fields")
                parsed.append({"id": rule_id, "kind": kind, "fields": fields})
            elif kind == "sum":
                object_value(
                    rule,
                    {"id", "kind", "field", "expected", "tolerance"},
                    {"id", "kind", "field", "expected", "tolerance"},
                )
                field = schema.by_name.get(rule["field"])
                if field is None or field.type not in {"integer", "decimal"}:
                    reject("quality", "sum rule requires numeric field")
                expected = decimal(rule["expected"])
                tolerance = decimal(rule["tolerance"])
                if tolerance < 0:
                    reject("quality", "sum tolerance cannot be negative")
                parsed.append(
                    {
                        "id": rule_id,
                        "kind": kind,
                        "field": field.name,
                        "expected": decimal_text(expected),
                        "tolerance": decimal_text(tolerance),
                    }
                )
            elif kind == "relation":
                object_value(
                    rule,
                    {"id", "kind", "left", "operator", "right"},
                    {"id", "kind", "left", "operator", "right"},
                )
                if (
                    rule["left"] not in schema.by_name
                    or rule["right"] not in schema.by_name
                ):
                    reject("quality", "relation field absent")
                if (
                    schema.by_name[rule["left"]].type
                    != schema.by_name[rule["right"]].type
                ):
                    reject("quality", "relation fields must have same type")
                if rule["operator"] not in {"eq", "le", "lt", "ge", "gt"}:
                    reject("quality", "unsupported relation operator")
                parsed.append(dict(rule))
            else:
                reject("quality", "unknown quality rule kind")
        payload = {
            "id": suite_id,
            "schema": schema_id,
            "schema_version": schema_version,
            "rules": parsed,
        }
        with self.db.transaction() as con:
            old = self.db.one(
                "SELECT digest FROM quality_suites WHERE id=?", (suite_id,)
            )
            if old:
                if old["digest"] != digest(payload):
                    reject("conflict", "quality suite is immutable")
                return payload
            con.execute(
                "INSERT INTO quality_suites VALUES(?,?,?)",
                (suite_id, canonical(payload), digest(payload)),
            )
            self.db.audit("quality_suite", suite_id, payload)
        return payload

    def suite(self, suite_id):
        row = self.db.one("SELECT payload FROM quality_suites WHERE id=?", (suite_id,))
        if row is None:
            reject("missing", "quality suite not found")
        return parse_json(row["payload"])

    def run(self, run_id, suite_id, dataset, version):
        name(run_id)
        suite = self.suite(suite_id)
        record = self.registry.version(dataset, version)
        registration = self.registry.dataset(dataset)
        if (
            registration["schema_id"] != suite["schema"]
            or record["schema_version"] != suite["schema_version"]
        ):
            reject("quality", "quality suite schema differs from dataset version")
        request = {
            "suite": suite_id,
            "dataset": dataset,
            "version": version,
            "version_digest": record["digest"],
        }
        with self.db.transaction() as con:
            old = self.db.one("SELECT * FROM quality_runs WHERE id=?", (run_id,))
            if old:
                if old["request"] != canonical(request):
                    reject("conflict", "quality run identity differs")
                return parse_json(old["result"])
            schema = self.registry.schema(suite["schema"], suite["schema_version"])
            rows = self.ingestion.rows(dataset, version)
            results = [evaluate_rule(rule, schema, rows) for rule in suite["rules"]]
            result = {
                "id": run_id,
                **request,
                "passed": all(item["passed"] for item in results),
                "rules": results,
            }
            con.execute(
                "INSERT INTO quality_runs VALUES(?,?,?)",
                (run_id, canonical(request), canonical(result)),
            )
            self.db.audit(
                "quality_run",
                run_id,
                {"passed": result["passed"], "dataset": dataset, "version": version},
            )
        return result

    def result(self, run_id):
        row = self.db.one("SELECT result FROM quality_runs WHERE id=?", (run_id,))
        if row is None:
            reject("missing", "quality run not found")
        return parse_json(row["result"])

    def attest_release(self, release_id, run_ids):
        release = self.db.one(
            "SELECT manifest,state FROM releases WHERE id=?", (release_id,)
        )
        if release is None:
            reject("missing", "release not found")
        manifest = parse_json(release["manifest"])
        needed = {(item["dataset"], item["version"]) for item in manifest["selected"]}
        covered = set()
        results = []
        for run_id in list_value(run_ids, 1, 1000):
            result = self.result(run_id)
            reference = (result["dataset"], result["version"])
            if reference not in needed:
                reject("quality", "quality run does not cover selected release version")
            if not result["passed"]:
                reject("quality", "quality run failed", run=run_id)
            record = self.registry.version(*reference)
            if (
                record["digest"] != result["version_digest"]
                or record["state"] != "active"
            ):
                reject("quality", "quality evidence is stale")
            covered.add(reference)
            results.append(result)
        if covered != needed:
            reject("quality", "quality evidence does not cover all selected versions")
        return {
            "release": release_id,
            "covered": len(covered),
            "runs": [item["id"] for item in results],
            "evidence_digest": digest(results),
            "passed": True,
        }


def evaluate_rule(rule, schema, rows):
    kind = rule["kind"]
    violations = []
    observed = None
    if kind == "row_count":
        observed = len(rows)
        passed = rule["minimum"] <= observed <= rule["maximum"]
    elif kind == "sum":
        numbers = [
            decimal(row[rule["field"]])
            for row in rows
            if row.get(rule["field"]) is not None
        ]
        observed = decimal_text(sum(numbers, Decimal(0)))
        passed = abs(decimal(observed) - decimal(rule["expected"])) <= decimal(
            rule["tolerance"]
        )
    elif kind == "unique":
        seen = {}
        for index, row in enumerate(rows, 1):
            values = [row.get(field) for field in rule["fields"]]
            encoded = canonical(values)
            if encoded in seen:
                violations.append(
                    {"row": index, "previous_row": seen[encoded], "key": values}
                )
            else:
                seen[encoded] = index
        passed = not violations
    elif kind == "not_null":
        for index, row in enumerate(rows, 1):
            missing = [field for field in rule["fields"] if row.get(field) is None]
            if missing:
                violations.append({"row": index, "fields": missing})
        passed = not violations
    elif kind == "monotonic":
        previous = None
        for index, row in enumerate(rows, 1):
            values = [row.get(field) for field in rule["fields"]]
            if any(value is None for value in values):
                violations.append({"row": index, "reason": "null_key"})
                continue
            normalized = tuple(
                decimal(value)
                if schema.by_name[field].type in {"integer", "decimal"}
                else value
                for field, value in zip(rule["fields"], values)
            )
            if previous is not None and normalized < previous:
                violations.append({"row": index, "reason": "decreasing"})
            previous = normalized
        passed = not violations
    else:
        for index, row in enumerate(rows, 1):
            left, right = row.get(rule["left"]), row.get(rule["right"])
            if left is None or right is None:
                violations.append({"row": index, "reason": "null_operand"})
                continue
            if schema.by_name[rule["left"]].type in {"integer", "decimal"}:
                left, right = decimal(left), decimal(right)
            comparisons = {
                "eq": left == right,
                "le": left <= right,
                "lt": left < right,
                "ge": left >= right,
                "gt": left > right,
            }
            if not comparisons[rule["operator"]]:
                violations.append({"row": index, "reason": "relation_failed"})
        passed = not violations
    return {
        "id": rule["id"],
        "kind": kind,
        "passed": passed,
        "observed": observed,
        "violation_count": len(violations),
        "violations": violations[:1000],
        "truncated": len(violations) > 1000,
    }
