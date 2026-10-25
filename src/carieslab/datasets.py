"""Discover study dataset folders in a ZIP without extracting or changing pixels.

Folder names may vary between releases. Recognisable internal layouts take
precedence over known folder aliases. Ambiguous or unsafe archives are rejected.
Only Python's standard library is used.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import stat
import tempfile
import unicodedata
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

DOMAINS = ("acta_direct", "dcbr", "bw25")
CANONICAL_FOLDERS = {"acta_direct": "ACTA-DIRECT dataset v2", "dcbr": "DCBR proximal caries lesions dataset", "bw25": "ACTA-Bw25-RefStd"}
IMAGE_EXTENSIONS = {".png", ".tif", ".tiff"}
ALIASES = {
    "acta_direct": {"actadirect", "actadirectv2", "actadirectdatasetv2", "actadirectdatasetversion2"},
    "dcbr": {"dcbr", "refineddcbr", "dcbrdataset", "refineddcbrdataset", "dcbrproximalcarieslesionsdataset"},
    "bw25": {"actabw25refstd", "actabw25refstandard", "actabw25", "actabitewings25referencestandard"},
}
REQUIRED_ARTIFACTS = {
    "acta_direct": ("3_Radiographs/2_Proximal_Sides", "4_Radiograph_Annotations/2_Proximal_Sides_Annotations"),
    "dcbr": ("bitewings", "masks", "Proximal_Side_Regions.csv"),
    "bw25": ("1_Bitewings", "2_Proximal_Caries_Lesion_Delineations", "Proximal_Side_Regions.csv"),
}


def _normalise(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).casefold()
    return re.sub(r"[^a-z0-9]", "", value)


@contextmanager
def _open_archive(archive):
    if isinstance(archive, ZipFile):
        yield archive
    else:
        with ZipFile(archive) as handle:
            yield handle


def _safe_files(handle: ZipFile) -> list[ZipInfo]:
    """Validate every member, including files outside the selected datasets."""
    files = []
    seen = {}
    folder_spelling = {}
    total = 0
    for info in handle.infolist():
        name = info.filename
        components = name.rstrip("/").split("/")
        if (not name or info.orig_filename != name or name.startswith("/") or "\\" in name or "\x00" in info.orig_filename
                or any(c in {"", ".", ".."} or ":" in c for c in components)):
            raise ValueError(f"Unsafe ZIP path: {name!r}")
        canonical = unicodedata.normalize("NFC", name.rstrip("/")).casefold()
        if canonical in seen:
            raise ValueError(f"Duplicate ZIP member or name collision: {name!r}")
        seen[canonical] = info.is_dir()
        for end in range(1, len(components) + int(info.is_dir())):
            prefix = "/".join(components[:end])
            key = unicodedata.normalize("NFC", prefix).casefold()
            previous = folder_spelling.setdefault(key, prefix)
            if previous != prefix:
                raise ValueError(f"Folder spelling collision in ZIP: {previous!r} and {prefix!r}")
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
            raise ValueError(f"ZIP symlink or special file is not supported: {name!r}")
        if info.flag_bits & 1:
            raise ValueError(f"Encrypted ZIP member is not supported: {name!r}")
        if not info.is_dir():
            total += info.file_size
            files.append(info)
    for member in seen:
        for parent in PurePosixPath(member).parents:
            if str(parent) != "." and seen.get(str(parent)) is False:
                raise ValueError(f"File/directory path collision in ZIP: {member!r}")
    if total > 128 * 1024 ** 3:
        raise ValueError("ZIP declares more than 128 GiB of uncompressed data")
    return files


def _has_image(relative_paths, prefix):
    target = tuple(_normalise(c) for c in prefix)
    return any(tuple(_normalise(c) for c in path[:len(target)]) == target
               and len(path) > len(target) and PurePosixPath(path[-1]).suffix.lower() in IMAGE_EXTENSIONS
               for path in relative_paths)


def _layout_domains(relative_paths):
    found = []
    direct_names = {_normalise(p[0]) for p in relative_paths if len(p) == 1}
    if (_has_image(relative_paths, ("3_Radiographs", "2_Proximal_Sides"))
            and _has_image(relative_paths, ("4_Radiograph_Annotations", "2_Proximal_Sides_Annotations"))):
        found.append("acta_direct")
    if (_has_image(relative_paths, ("bitewings",)) and _has_image(relative_paths, ("masks",))
            and "proximalsideregionscsv" in direct_names):
        found.append("dcbr")
    if (_has_image(relative_paths, ("1_Bitewings",))
            and _has_image(relative_paths, ("2_Proximal_Caries_Lesion_Delineations",))
            and "proximalsideregionscsv" in direct_names):
        found.append("bw25")
    return found


def _discover(handle):
    files = _safe_files(handle)
    by_prefix = defaultdict(list)
    for info in files:
        parts = PurePosixPath(info.filename).parts
        if any(p.startswith(".") or p == "__MACOSX" for p in parts):
            continue
        for end in range(1, len(parts)):
            by_prefix[parts[:end]].append((info, parts[end:]))
    strong = defaultdict(list)
    fallback = defaultdict(list)
    for prefix, entries in by_prefix.items():
        relative = [p for _, p in entries]
        matches = _layout_domains(relative)
        if len(matches) > 1:
            raise ValueError(f"One folder contains multiple dataset layouts: {'/'.join(prefix)!r}")
        if matches:
            strong[matches[0]].append(prefix)
        else:
            label = _normalise(prefix[-1])
            for domain, aliases in ALIASES.items():
                if label in aliases and any(PurePosixPath(p[-1]).suffix.lower() in IMAGE_EXTENSIONS for p in relative):
                    fallback[domain].append(prefix)
    selected = {}
    for domain in DOMAINS:
        candidates = strong[domain] or fallback[domain]
        if len(candidates) != 1:
            names = ["/".join(p) for p in candidates]
            raise ValueError(f"Expected one {domain} dataset folder; found {len(candidates)}: {names}")
        prefix = candidates[0]
        # The data readers select by a single folder component. That component
        # must therefore occur under exactly one archive prefix, not just have
        # one recognised layout among several copies.
        component = prefix[-1]
        occurrences = set()
        for info in files:
            parts = PurePosixPath(info.filename).parts
            for index, value in enumerate(parts[:-1]):
                if value == component:
                    occurrences.add(parts[:index + 1])
        if occurrences != {prefix}:
            raise ValueError(f"Dataset folder name occurs under multiple prefixes: {component!r}")
        selected[domain] = prefix
    if len(set(selected.values())) != len(DOMAINS):
        raise ValueError("A dataset folder was assigned to more than one domain")
    return files, by_prefix, selected


def detect_raw_names(archive) -> dict[str, str]:
    """Return the actual dataset folder components for the three study sources.

    The keys are ``acta_direct``, ``dcbr`` and ``bw25``. Wrapper directories are
    allowed. All three datasets must be present. This function reads ZIP member
    metadata only; it does not extract files or decode, modify or save images.
    """
    with _open_archive(archive) as handle:
        _, _, selected = _discover(handle)
        return {domain: prefix[-1] for domain, prefix in selected.items()}


def _csv_inventory(handle, info):
    if info.file_size > 16 * 1024 ** 2:
        raise ValueError(f"Metadata CSV exceeds 16 MiB: {info.filename!r}")
    raw = handle.read(info)
    sample = raw.decode("utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(sample[:8192], delimiters=",;\t")
    except csv.Error as exc:
        raise ValueError(f"Cannot identify CSV separator: {info.filename!r}") from exc
    rows = [r for r in csv.reader(io.StringIO(sample, newline=""), dialect) if any(v.strip() for v in r)]
    if not rows:
        raise ValueError(f"Empty metadata CSV: {info.filename!r}")
    header = [v.strip() for v in rows[0]]
    if len(header) != len(set(header)):
        raise ValueError(f"Duplicate metadata CSV columns: {info.filename!r}")
    if any(len(row) != len(header) for row in rows[1:]):
        raise ValueError(f"Inconsistent metadata CSV row width: {info.filename!r}")
    result = {"columns": header, "rows": len(rows) - 1, "sha256": hashlib.sha256(raw).hexdigest()}
    if "Bitewing" in header:
        index = header.index("Bitewing")
        result["bitewings"] = len({int(row[index]) for row in rows[1:]})
    return result


def inspect_archive(archive, *, include_members: bool = True) -> dict:
    """Return a JSON-serialisable inventory without redistributing image pixels.

    CSV metadata may be read to count rows. Image membership is recorded only
    by relative path, byte count and ZIP CRC. No archive member is extracted.
    """
    with _open_archive(archive) as handle:
        files, by_prefix, selected = _discover(handle)
        report = {"schema_version": 1, "file_count": len(files), "uncompressed_bytes": sum(i.file_size for i in files), "datasets": {}}
        for domain in DOMAINS:
            prefix = selected[domain]
            entries = by_prefix[prefix]
            names = {"/".join(p): info for info, p in entries}
            relative = [p for _, p in entries]
            required = {}
            for artifact in REQUIRED_ARTIFACTS[domain]:
                required[artifact] = artifact in names if artifact.endswith(".csv") else _has_image(relative, PurePosixPath(artifact).parts)
            item = {"folder": prefix[-1], "prefix": "/".join(prefix), "files": len(entries),
                    "image_files": sum(PurePosixPath(p[-1]).suffix.lower() in IMAGE_EXTENSIONS for p in relative),
                    "extensions": dict(sorted(Counter(PurePosixPath(p[-1]).suffix.lower() for p in relative).items())),
                    "required_artifacts": required, "all_required_artifacts_present": all(required.values()), "metadata": {}}
            for name, info in names.items():
                if PurePosixPath(name).suffix.lower() == ".csv":
                    item["metadata"][name] = _csv_inventory(handle, info)
            if include_members:
                item["members"] = [{"path": "/".join(p), "bytes": info.file_size, "crc32": f"{info.CRC:08x}"}
                                   for info, p in sorted(entries, key=lambda x: x[1])]
            report["datasets"][domain] = item
        return report


class MissingDatasetFilesError(ValueError):
    """Required exact-byte inputs were not found; no output ZIP was committed."""

    def __init__(self, missing_files):
        self.missing_files = list(missing_files)
        details = "\n".join(f"  {r['domain']}/{r['path']}  SHA256 {r['sha256']}" for r in self.missing_files[:20])
        remainder = len(self.missing_files) - 20
        if remainder > 0:
            details += f"\n  ... {remainder} further files are listed in this exception's missing_files attribute."
        super().__init__(f"Cannot assemble dataset: {len(self.missing_files)} required exact-byte files are missing. No partial output was written.\n{details}")


def _fingerprint_records(fingerprints):
    if fingerprints is None:
        fingerprints = Path(__file__).resolve().parents[2] / "configs/main/data_fingerprints.json"
    if isinstance(fingerprints, (str, Path)):
        with Path(fingerprints).open(encoding="utf-8") as stream:
            fingerprints = json.load(stream)
    records = fingerprints.get("files") if isinstance(fingerprints, dict) else fingerprints
    if not isinstance(records, list) or not records:
        raise ValueError("Dataset fingerprints must contain a nonempty files list")
    seen = set()
    result = []
    for item in records:
        if not isinstance(item, dict) or not {"domain", "path", "sha256"}.issubset(item):
            raise ValueError("Every fingerprint needs domain, path and sha256")
        domain, name, sha = item["domain"], item["path"], item["sha256"]
        if domain not in DOMAINS or not isinstance(name, str) or not isinstance(sha, str):
            raise ValueError("Invalid dataset fingerprint types or domain")
        parts = name.split("/")
        if (not name or name.startswith("/") or "\\" in name or "\x00" in name
                or any(p in {"", ".", ".."} or ":" in p for p in parts)):
            raise ValueError(f"Unsafe fingerprint path: {name!r}")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", sha):
            raise ValueError(f"Invalid SHA256 fingerprint: {domain}/{name}")
        key = (domain, unicodedata.normalize("NFC", name).casefold())
        if key in seen:
            raise ValueError(f"Duplicate or colliding fingerprint target: {domain}/{name}")
        seen.add(key)
        result.append({"domain": domain, "path": name, "sha256": sha.lower()})
    return sorted(result, key=lambda r: (r["domain"], r["path"]))


def _input_files(inputs, output):
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    if inputs is None:
        raise ValueError("At least one dataset input file or directory is required")
    found = set()
    for raw in inputs:
        path = Path(raw)
        if path.is_symlink():
            raise ValueError(f"Dataset input is a symlink: {path}")
        if not path.exists():
            raise FileNotFoundError(f"Dataset input does not exist: {path}")
        if path.is_dir():
            for directory, dirs, files in os.walk(path, followlinks=False):
                for name in dirs:
                    if (Path(directory) / name).is_symlink():
                        raise ValueError(f"Dataset input contains a directory symlink: {Path(directory) / name}")
                for name in files:
                    child = Path(directory) / name
                    if child.is_symlink() or not child.is_file():
                        raise ValueError(f"Dataset input is not a regular file: {child}")
                    if child.resolve() != output:
                        found.add(child.resolve())
        elif path.is_file():
            if path.resolve() != output:
                found.add(path.resolve())
        else:
            raise ValueError(f"Dataset input is not a regular file or directory: {path}")
    return sorted(found, key=lambda p: str(p))


def _stream_sha256(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def assemble_archives(inputs, output, fingerprints=None) -> dict:
    """Assemble the exact study data from downloaded ZIPs or directories.

    Files are matched by SHA256, independently of their input names or folder
    layout. The manifest provides their canonical destination paths. Byte-
    identical duplicate sources are allowed. Images are copied without decoding
    or re-encoding. All ZIP members are checked for unsafe paths and links.

    ``fingerprints`` defaults to ``configs/main/data_fingerprints.json``. It may
    also be a path, a dictionary containing ``files`` or a list of fingerprint
    records. Missing inputs raise ``MissingDatasetFilesError`` with the complete
    ``missing_files`` list. The destination is replaced atomically only after
    every required file is found, copied and checked again.
    """
    records = _fingerprint_records(fingerprints)
    output = Path(output).resolve()
    sources = _input_files(inputs, output)
    wanted = {r["sha256"] for r in records}
    found = {}
    scanned = 0
    for path in sources:
        if path.suffix.lower() == ".zip":
            with ZipFile(path) as handle:
                for info in _safe_files(handle):
                    with handle.open(info) as stream:
                        sha = _stream_sha256(stream)
                    scanned += 1
                    if sha in wanted:
                        found.setdefault(sha, (path, info.filename))
        else:
            with path.open("rb") as stream:
                sha = _stream_sha256(stream)
            scanned += 1
            if sha in wanted:
                found.setdefault(sha, (path, None))
    missing = [r for r in records if r["sha256"] not in found]
    if missing:
        raise MissingDatasetFilesError(missing)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False) as temp:
            temp_path = Path(temp.name)
        with ExitStack() as stack, ZipFile(temp_path, "w", compression=ZIP_DEFLATED, compresslevel=6) as destination:
            handles = {}
            for record in records:
                path, member = found[record["sha256"]]
                if member is None:
                    source_stream = path.open("rb")
                else:
                    if path not in handles:
                        handles[path] = stack.enter_context(ZipFile(path))
                    source_stream = handles[path].open(member)
                target = f"Datasets/{CANONICAL_FOLDERS[record['domain']]}/{record['path']}"
                info = ZipInfo(target, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                digest = hashlib.sha256()
                with source_stream as source, destination.open(info, "w", force_zip64=True) as sink:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(block)
                        sink.write(block)
                if digest.hexdigest() != record["sha256"]:
                    raise ValueError(f"Input changed while copying {record['domain']}/{record['path']}; no output was committed")
        with temp_path.open("rb") as stream:
            output_sha256 = _stream_sha256(stream)
        os.replace(temp_path, output)
        temp_path = None
        return {"status": "complete", "output": str(output), "sha256": output_sha256,
                "file_count": len(records), "source_files_scanned": scanned,
                "domain_files": dict(sorted(Counter(r["domain"] for r in records).items()))}
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
