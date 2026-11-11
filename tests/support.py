from trialvault.validation import canonical


def schema(version=1, **changes):
    result = {
        "id": "schema",
        "version": version,
        "fields": [
            {"name": "id", "type": "text"},
            {"name": "value", "type": "decimal", "minimum": "0"},
            {"name": "group", "type": "enum", "choices": ["A", "B"]},
        ],
        "primary_key": ["id"],
    }
    result.update(changes)
    return result


def content(rows=None):
    if rows is None:
        rows = [
            {"id": "one", "value": "10", "group": "A"},
            {"id": "two", "value": "20", "group": "B"},
        ]
    return ("\n".join(canonical(row) for row in rows) + ("\n" if rows else "")).encode()


def register(vault, dataset="data", owner="owner"):
    vault.registry.add_schema(schema())
    vault.registry.register(dataset, "Synthetic dataset", "schema", owner)


def ingest(
    vault,
    request="req",
    dataset="data",
    rows=None,
    expected=0,
    classification="internal",
    scopes=None,
    parents=None,
):
    return vault.ingestion.ingest(
        request,
        dataset,
        1,
        content(rows),
        "jsonl",
        classification,
        scopes or ["research"],
        parents=parents or [],
        expected_version=expected,
    )


def release(vault, release_id="release", references=None):
    vault.releases.plan(
        release_id,
        references or [{"dataset": "data", "version": 1}],
        "research",
        "internal",
    )
    return vault.releases.approve(
        release_id, 1, "reviewer", "Checked synthetic evidence"
    )
