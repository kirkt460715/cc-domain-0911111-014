import argparse
import json
import sys
from pathlib import Path
from .validation import object_value, parse_json, canonical, reject, VaultError
from .service import Vault
from .releases.exporter import verify_directory


def dispatch(vault, request):
    object_value(request, required={"action"})
    options = dict(request)
    action = options.pop("action")
    if action == "schema":
        return vault.registry.add_schema(
            options["schema"], options.get("allow_breaking", False)
        )
    if action == "register":
        return vault.registry.register(**options)
    if action == "ingest":
        path = Path(options.pop("file"))
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 32000000:
            reject("path", "ingestion file must be bounded regular file")
        return vault.ingestion.ingest(content=path.read_bytes(), **options)
    if action == "version":
        return vault.registry.version(options["dataset"], options["version"])
    if action == "versions":
        return vault.registry.list_versions(options["dataset"])
    if action == "withdraw":
        return vault.registry.withdraw(
            options["dataset"], options["version"], options["reason"]
        )
    if action == "withdrawal_impact":
        return vault.checks.withdrawal_impact(**options)
    if action == "lineage":
        return vault.registry.closure(options["references"])
    if action == "project":
        return vault.derivations.project(**options)
    if action == "join":
        return vault.derivations.join(**options)
    if action == "aggregate":
        return vault.derivations.aggregate(**options)
    if action == "profile":
        return vault.checks.profile(**options)
    if action == "compare":
        return vault.checks.compare(**options)
    if action == "foreign_key":
        return vault.checks.foreign_key(**options)
    if action == "release_plan":
        return vault.releases.plan(**options)
    if action == "release_approve":
        return vault.releases.approve(**options)
    if action == "release_revoke":
        return vault.releases.revoke(**options)
    if action == "release":
        return vault.releases.get(options["id"])
    if action == "release_compare":
        return vault.releases.compare(options["left"], options["right"])
    if action == "export":
        return vault.exporter.export(**options)
    if action == "exports":
        return vault.exporter.pending()
    if action == "verify":
        return verify_directory(options["path"])
    if action == "check":
        return vault.checks.inspect()
    if action == "unused_artifacts":
        return vault.checks.unused_artifacts(options.get("dry_run", True))
    if action == "patch_plan":
        return vault.patches.plan(**options)
    if action == "patch_apply":
        return vault.patches.apply(options["id"])
    if action == "patch_cancel":
        return vault.patches.cancel(options["id"], options["reason"])
    if action == "quality_define":
        return vault.quality.define(**options)
    if action == "quality_run":
        return vault.quality.run(**options)
    if action == "quality_attest":
        return vault.quality.attest_release(**options)
    if action == "migration_preview":
        return vault.migrations.preview(**options)
    if action == "migration_apply":
        return vault.migrations.apply(options["id"])
    if action == "migrations":
        return vault.migrations.pending(options.get("dataset"))
    if action == "retention_review":
        return vault.retention.review(**options)
    if action == "retention_verify":
        return vault.retention.verify(options["id"])
    if action == "search":
        return vault.queries.search(**options)
    if action == "sample":
        return vault.queries.sample(**options)
    if action == "lookup":
        return vault.queries.lookup(**options)
    if action == "storage_usage":
        return vault.queries.storage_usage()
    if action == "experiment_define":
        return vault.experiments.define(options["id"], options["definition"])
    if action == "experiment_run":
        return vault.experiments.run(options["id"])
    if action == "experiment":
        return vault.experiments.get(options["id"])
    if action == "experiments":
        return vault.experiments.list()
    if action == "experiment_validity":
        return vault.experiments.validity(options["id"])
    if action == "experiment_predict":
        experiment_id = options.pop("id")
        return vault.experiments.predict(experiment_id, **options)
    reject("input", "unknown action", action=action)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="trialvault")
    parser.add_argument("--root", default="vault-data")
    parser.add_argument("command", choices=["run", "demo", "check", "verify"])
    parser.add_argument("file", nargs="?")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            from .demonstration import demonstrate

            result = demonstrate()
        elif args.command == "verify":
            result = verify_directory(args.file)
        else:
            with Vault(args.root) as vault:
                if args.command == "check":
                    result = vault.checks.inspect()
                else:
                    if not args.file:
                        reject("input", "action document required")
                    result = dispatch(vault, parse_json(Path(args.file).read_bytes()))
    except VaultError as exc:
        print(canonical({"error": exc.as_dict()}), file=sys.stderr)
        return 2
    except (OSError, KeyError, TypeError) as exc:
        print(
            canonical({"error": {"code": "request", "message": str(exc)}}),
            file=sys.stderr,
        )
        return 3
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
