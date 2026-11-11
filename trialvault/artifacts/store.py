import os
import tempfile
import hashlib
from pathlib import Path
from ..validation import reject


class ArtifactStore:
    def __init__(self, root):
        self.root = Path(root)
        if self.root.is_symlink():
            reject("path", "artifact root cannot be symbolic link")
        self.root.mkdir(parents=True, exist_ok=True)
        self.objects = self.root / "objects"
        self.staging = self.root / "staging"
        for directory in (self.objects, self.staging):
            if directory.is_symlink():
                reject("path", "artifact directory cannot be symbolic link")
            directory.mkdir(exist_ok=True)

    def path(self, checksum):
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or any(c not in "0123456789abcdef" for c in checksum)
        ):
            reject("input", "invalid SHA256 checksum")
        directory = self.objects / checksum[:2]
        if directory.is_symlink():
            reject("path", "object partition cannot be symbolic link")
        path = directory / checksum[2:]
        if path.is_symlink():
            reject("path", "artifact cannot be symbolic link")
        return path

    def put(self, content, maximum=64000000):
        if not isinstance(content, bytes) or len(content) > maximum:
            reject("input", "artifact must be bounded bytes")
        checksum = hashlib.sha256(content).hexdigest()
        path = self.path(checksum)
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            self.verify(checksum)
            return {"sha256": checksum, "bytes": len(content), "existing": True}
        fd, temporary = tempfile.mkstemp(prefix="write-", dir=self.staging)
        try:
            with os.fdopen(fd, "wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                self.verify(checksum)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return {"sha256": checksum, "bytes": len(content), "existing": False}

    def import_file(self, path):
        path = Path(path)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 64000000:
            reject("input", "artifact input must be bounded regular file")
        return self.put(path.read_bytes())

    def read(self, checksum):
        path = self.path(checksum)
        if not path.is_file():
            reject("missing", "artifact unavailable", sha256=checksum)
        if path.stat().st_size > 64000000:
            reject("integrity", "artifact exceeds size limit")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != checksum:
            reject("integrity", "artifact content hash mismatch", sha256=checksum)
        return content

    def verify(self, checksum):
        content = self.read(checksum)
        return {"sha256": checksum, "bytes": len(content), "verified": True}

    def inventory(self):
        result = []
        for directory in sorted(self.objects.iterdir()):
            if directory.is_symlink() or not directory.is_dir():
                reject("integrity", "invalid object partition")
            for path in sorted(directory.iterdir()):
                checksum = directory.name + path.name
                result.append(self.verify(checksum))
        return result

    def remove_unreferenced(self, referenced, dry_run=True):
        if not isinstance(dry_run, bool):
            reject("input", "dry_run must be boolean")
        referenced = set(referenced)
        candidates = [
            item for item in self.inventory() if item["sha256"] not in referenced
        ]
        if not dry_run:
            for item in candidates:
                self.path(item["sha256"]).unlink()
        return {
            "dry_run": dry_run,
            "candidates": candidates,
            "bytes": sum(item["bytes"] for item in candidates),
        }

    def clean_staging(self):
        deleted = []
        for path in sorted(self.staging.iterdir()):
            if (
                path.is_symlink()
                or not path.is_file()
                or not path.name.startswith("write-")
            ):
                reject("integrity", "unexpected staging content")
            path.unlink()
            deleted.append(path.name)
        return deleted
