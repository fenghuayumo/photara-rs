"""Validate version bumping and release metadata consistency."""
import tempfile
from pathlib import Path
import unittest

from version import check_version, read_version, sync_addon, validate_version


class VersionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'photara_rs_engine').mkdir()
        (self.root / 'VERSION').write_text('1.2.3\n')
        (self.root / 'photara_rs_engine/__init__.py').write_text(
            'bl_info = {"name": "Photara", "version": (1, 2, 3)}\n')

    def test_shared_release_version(self):
        self.assertEqual(read_version(self.root), '1.2.3')
        self.assertEqual(check_version(self.root), '1.2.3')

    def test_mismatched_addon_blocks_packaging(self):
        (self.root / 'VERSION').write_text('1.3.0\n')
        with self.assertRaises(ValueError):
            check_version(self.root)
        sync_addon(read_version(self.root), self.root)
        self.assertEqual(check_version(self.root), '1.3.0')

    def test_bump_preserves_other_addon_metadata(self):
        sync_addon('2.0.0', self.root)
        source = (self.root / 'photara_rs_engine/__init__.py').read_text()
        self.assertIn('"name": "Photara"', source)
        self.assertIn('(2, 0, 0)', source)

    def test_rejects_invalid_release_versions(self):
        for value in ('v1.2.3', '1.2', '01.2.3', '1.2.3rc1', '1.2.3\n2.0.0'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_version(value)


if __name__ == '__main__':
    unittest.main()
