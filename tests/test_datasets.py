"""Dataset discovery and exact-byte assembly checks; no image libraries required."""
from __future__ import annotations

import hashlib
import io
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from carieslab.datasets import (CANONICAL_FOLDERS, MissingDatasetFilesError,
                                assemble_archives, detect_raw_names, inspect_archive)


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def layouts(laboratory="Laboratory images", clinical="Clinical development", assessment="Clinical assessment"):
    return {
        f"wrapper/{laboratory}/3_Radiographs/2_Proximal_Sides/0001.1.l.tif": b"unchanged image bytes",
        f"wrapper/{laboratory}/4_Radiograph_Annotations/2_Proximal_Sides_Annotations/0001.1.l.tif": b"unchanged mask bytes",
        f"wrapper/{clinical}/bitewings/001.png": b"unchanged image bytes",
        f"wrapper/{clinical}/masks/001.png": b"unchanged mask bytes",
        f"wrapper/{clinical}/Proximal_Side_Regions.csv": b"Bitewing,Surface,x1,y1,x2,y2\n001,16D,0,0,1,1\n",
        f"wrapper/{assessment}/1_Bitewings/01.tif": b"unchanged image bytes",
        f"wrapper/{assessment}/2_Proximal_Caries_Lesion_Delineations/01.tif": b"unchanged mask bytes",
        f"wrapper/{assessment}/Proximal_Side_Regions.csv": b"Bitewing,Surface,x1,y1,x2,y2\n1,16D,0,0,1,1\n",
    }


def memory_zip(members, extras=()):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as handle:
        for name, data in members.items():
            handle.writestr(name, data)
        for info, data in extras:
            handle.writestr(info, data)
    buffer.seek(0)
    return ZipFile(buffer)


class DatasetDiscoveryTests(unittest.TestCase):
    def test_arbitrary_folder_names_are_identified_by_layout(self):
        with memory_zip(layouts()) as archive:
            self.assertEqual(detect_raw_names(archive), {
                "acta_direct": "Laboratory images", "dcbr": "Clinical development", "bw25": "Clinical assessment"})
            inventory = inspect_archive(archive, include_members=False)
            self.assertEqual(inventory["datasets"]["dcbr"]["metadata"]["Proximal_Side_Regions.csv"]["rows"], 1)
            self.assertNotIn("members", inventory["datasets"]["dcbr"])

    def test_known_folder_aliases_are_preserved(self):
        with memory_zip(layouts("ACTA-DIRECT dataset v2", "Refined DCBR", "ACTA-Bw25-RefStd")) as archive:
            self.assertEqual(detect_raw_names(archive)["dcbr"], "Refined DCBR")

    def test_duplicate_dataset_prefixes_are_rejected(self):
        members = layouts()
        members.update({name.replace("wrapper/Clinical development", "another/Clinical development"): data
                        for name, data in layouts().items() if name.startswith("wrapper/Clinical development")})
        with memory_zip(members) as archive, self.assertRaises(ValueError):
            detect_raw_names(archive)

    def test_selected_component_cannot_occur_under_an_unrelated_prefix(self):
        members = layouts()
        members["another/Clinical development/unrelated.txt"] = b"extra"
        with memory_zip(members) as archive, self.assertRaises(ValueError):
            detect_raw_names(archive)

    def test_unselected_traversal_and_symlinks_are_rejected(self):
        members = layouts()
        members["../outside.txt"] = b"extra"
        with memory_zip(members) as archive, self.assertRaises(ValueError):
            detect_raw_names(archive)
        link = ZipInfo("unselected_link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        with memory_zip(layouts(), [(link, b"elsewhere")]) as archive, self.assertRaises(ValueError):
            detect_raw_names(archive)


class DatasetAssemblyTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / "results"
        temporary_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.base = Path(self.temporary.name)
        self.downloads = self.base / "downloads"
        self.downloads.mkdir()
        (self.downloads / "unexpected_name.bin").write_bytes(b"unchanged image bytes")
        self.archive = self.downloads / "download.zip"
        with ZipFile(self.archive, "w") as handle:
            handle.writestr("another annotation family/mask.dat", b"unchanged mask bytes")
        self.records = [
            {"domain": "acta_direct", "path": "3_Radiographs/2_Proximal_Sides/0001.1.l.tif", "sha256": checksum(b"unchanged image bytes")},
            {"domain": "acta_direct", "path": "4_Radiograph_Annotations/2_Proximal_Sides_Annotations/0001.1.l.tif", "sha256": checksum(b"unchanged mask bytes")},
            {"domain": "acta_direct", "path": "4_Radiograph_Annotations/2_Proximal_Sides_Annotations/0001.1.r.tif", "sha256": checksum(b"unchanged mask bytes")},
        ]

    def tearDown(self):
        self.temporary.cleanup()

    def test_exact_bytes_are_found_by_hash_not_input_name(self):
        output = self.base / "assembled.zip"
        result = assemble_archives(self.downloads, output, self.records)
        self.assertEqual(result["file_count"], 3)
        with ZipFile(output) as handle:
            self.assertEqual(len(handle.namelist()), 3)
            for record in self.records:
                target = f"Datasets/{CANONICAL_FOLDERS[record['domain']]}/{record['path']}"
                self.assertEqual(checksum(handle.read(target)), record["sha256"])

    def test_output_is_deterministic_with_different_input_and_manifest_order(self):
        first, second = self.base / "first.zip", self.base / "second.zip"
        assemble_archives(self.downloads, first, self.records)
        assemble_archives([self.archive, self.downloads / "unexpected_name.bin"], second, list(reversed(self.records)))
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_missing_input_never_commits_partial_output(self):
        output = self.base / "existing.zip"
        output.write_bytes(b"existing output")
        missing = {"domain": "dcbr", "path": "masks/001.png", "sha256": checksum(b"required missing bytes")}
        for destination in (output, self.base / "absent.zip"):
            with self.assertRaises(MissingDatasetFilesError) as captured:
                assemble_archives(self.downloads, destination, self.records + [missing])
            self.assertEqual(captured.exception.missing_files, [missing])
        self.assertEqual(output.read_bytes(), b"existing output")
        self.assertFalse((self.base / "absent.zip").exists())
        self.assertEqual(list(self.base.glob(".existing.zip.*.tmp")), [])

    def test_manifest_traversal_and_target_collisions_are_rejected(self):
        invalid = {"domain": "dcbr", "path": "../outside", "sha256": checksum(b"x")}
        duplicate = dict(self.records[0], path=self.records[0]["path"].upper())
        for records in ([invalid], self.records + [duplicate]):
            with self.assertRaises(ValueError):
                assemble_archives(self.downloads, self.base / "output.zip", records)
        self.assertFalse((self.base / "output.zip").exists())

    def test_unsafe_extra_input_is_rejected_after_matching_files_exist(self):
        bad = self.base / "unsafe.zip"
        with ZipFile(bad, "w") as handle:
            handle.writestr("../outside", b"not selected")
        output = self.base / "existing.zip"
        output.write_bytes(b"existing output")
        with self.assertRaises(ValueError):
            assemble_archives([self.downloads, bad], output, self.records)
        self.assertEqual(output.read_bytes(), b"existing output")


if __name__ == "__main__":
    unittest.main()
