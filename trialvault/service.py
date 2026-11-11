from .catalog.patches import VersionPatches
from .contracts.suites import QualitySuites
from .contracts.migration import SchemaMigration
from .releases.retention import RetentionReview
from .catalog.query import CatalogQueries
from pathlib import Path
from .validation import reject
from .artifacts.store import ArtifactStore
from .artifacts.ingest import Ingestion
from .catalog.database import CatalogDatabase
from .catalog.registry import Registry
from .catalog.derive import Derivations
from .catalog.checks import DatasetChecks
from .releases.planner import ReleasePlanner
from .releases.exporter import Exporter
from .experiments.service import Experiments


class Vault:
    def __init__(self, root):
        root = Path(root)
        if root.is_symlink():
            reject("path", "vault root cannot be symlink")
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.artifacts = ArtifactStore(root / "artifacts")
        self.database = CatalogDatabase(root / "catalog.sqlite3")
        self.registry = Registry(self.database, self.artifacts)
        self.ingestion = Ingestion(self.database, self.registry, self.artifacts)
        self.derivations = Derivations(self.registry, self.ingestion)
        self.checks = DatasetChecks(
            self.database, self.registry, self.ingestion, self.artifacts
        )
        self.releases = ReleasePlanner(self.database, self.registry, self.artifacts)
        self.exporter = Exporter(self.database, self.releases, self.artifacts)
        self.patches = VersionPatches(self.database, self.registry, self.ingestion)
        self.quality = QualitySuites(
            self.database, self.registry, self.ingestion, self.checks
        )
        self.migrations = SchemaMigration(self.database, self.registry, self.ingestion)
        self.retention = RetentionReview(self.database, self.registry, self.releases)
        self.queries = CatalogQueries(self.database, self.registry, self.ingestion)
        self.experiments = Experiments(
            self.database, self.registry, self.ingestion, self.artifacts
        )

    def close(self):
        self.database.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
