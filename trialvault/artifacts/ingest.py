import csv
import io
from ..validation import (
    name,
    parse_json,
    canonical,
    digest,
    reject,
    VaultError,
)


class Ingestion:
    def __init__(self, database, registry, artifacts):
        self.db = database
        self.registry = registry
        self.artifacts = artifacts

    def parse(self, content, format_name, schema):
        if not isinstance(content, bytes) or len(content) > 32000000:
            reject("input", "dataset input must be bounded bytes")
        try:
            text = content.decode("utf-8-sig")
        except UnicodeError:
            reject("input", "dataset input must be UTF-8")
        if format_name == "jsonl":
            rows = []
            for number, line in enumerate(text.splitlines(), 1):
                if line.strip():
                    try:
                        rows.append((number, parse_json(line)))
                    except VaultError as exc:
                        reject(
                            "row",
                            "invalid JSONL record",
                            line=number,
                            error=exc.as_dict(),
                        )
        elif format_name == "csv":
            reader = csv.reader(io.StringIO(text, newline=""), strict=True)
            try:
                header = next(reader)
            except StopIteration:
                reject("input", "CSV requires header")
            if len(set(header)) != len(header):
                reject("input", "duplicate CSV columns")
            rows = []
            try:
                for values in reader:
                    if len(values) != len(header):
                        reject("row", "CSV row width mismatch", line=reader.line_num)
                    converted = {}
                    for column, value in zip(header, values):
                        field = schema.by_name.get(column)
                        if field is None:
                            converted[column] = value
                        elif value == "" and field.nullable:
                            converted[column] = None
                        elif field.type == "integer":
                            try:
                                converted[column] = int(value)
                            except ValueError:
                                reject(
                                    "row",
                                    "CSV integer malformed",
                                    line=reader.line_num,
                                    column=column,
                                )
                        elif field.type == "boolean":
                            if value not in {"true", "false"}:
                                reject(
                                    "row",
                                    "CSV boolean must be true or false",
                                    column=column,
                                )
                            converted[column] = value == "true"
                        else:
                            converted[column] = value
                    rows.append((reader.line_num, converted))
            except csv.Error as exc:
                reject("row", "CSV syntax error", reason=str(exc))
        else:
            reject("input", "format must be csv or jsonl")
        if len(rows) > 100000:
            reject("input", "dataset row limit exceeded")
        result = []
        keys = set()
        for number, row in rows:
            try:
                normalized = schema.validate(row)
                primary = schema.row_key(normalized)
                if primary in keys:
                    reject("row", "duplicate primary key")
                keys.add(primary)
                result.append(normalized)
            except VaultError as exc:
                reject("row", "row violates contract", line=number, error=exc.as_dict())
        result.sort(key=schema.row_key)
        return result

    def ingest(
        self,
        request_id,
        dataset_id,
        schema_version,
        content,
        format_name,
        classification,
        scopes,
        parents=None,
        metadata=None,
        expected_version=0,
    ):
        name(request_id)
        dataset = self.registry.dataset(dataset_id)
        schema = self.registry.schema(dataset["schema_id"], schema_version)
        rows = self.parse(content, format_name, schema)
        normalized = (
            "\n".join(canonical(row) for row in rows) + ("\n" if rows else "")
        ).encode()
        request = {
            "dataset": dataset_id,
            "schema_version": schema_version,
            "rows": rows,
            "classification": classification,
            "scopes": scopes,
            "parents": parents or [],
            "metadata": metadata or {},
            "expected_version": expected_version,
        }
        checksum = digest(request)
        with self.db.transaction() as con:
            old = self.db.one("SELECT * FROM ingestion WHERE id=?", (request_id,))
            if old:
                if old["digest"] != checksum:
                    reject("conflict", "ingestion id has different normalized content")
                return {**parse_json(old["result"]), "reused": True}
            artifact = self.artifacts.put(normalized)
            version = self.registry.add_version(
                dataset_id,
                schema_version,
                artifact["sha256"],
                len(rows),
                classification,
                scopes,
                parents or [],
                metadata or {},
                expected_version,
            )
            result = {
                "dataset": dataset_id,
                "version": version["version"],
                "artifact": artifact["sha256"],
                "rows": len(rows),
                "reused": False,
            }
            con.execute(
                "INSERT INTO ingestion VALUES(?,?,?)",
                (request_id, checksum, canonical(result)),
            )
            self.db.audit("dataset_ingested", request_id, result)
        return result

    def rows(self, dataset_id, version):
        record = self.registry.version(dataset_id, version)
        content = self.artifacts.read(record["artifact"]).decode("utf-8")
        return [parse_json(line) for line in content.splitlines() if line]
