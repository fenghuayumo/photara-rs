from pathlib import Path
import sysconfig

from setuptools import Distribution, setup

_PACKAGE_DIR = Path(__file__).parent / "photara_rs"


def _check_native_payload() -> None:
    missing = []
    if not list(_PACKAGE_DIR.glob("_photara*.pyd")):
        missing.append("_photara*.pyd")
    else:
        expected = "_photara" + sysconfig.get_config_var("EXT_SUFFIX")
        modules = list(_PACKAGE_DIR.glob("_photara*.pyd"))
        if [module.name for module in modules] != [expected]:
            raise SystemExit(
                f"This interpreter requires {expected}; found "
                + ", ".join(module.name for module in modules)
                + ". Stage only the matching native extension before building a wheel."
            )
    if not list((_PACKAGE_DIR / "bin").glob("*.dll")):
        missing.append("bin/*.dll")
    if not list((_PACKAGE_DIR / "lib").glob("*.zst")):
        missing.append("lib/*.zst")
    if missing:
        raise SystemExit(
            "Native Photara payload is missing from photara_rs/: "
            + ", ".join(missing)
            + ". Build the extension and stage the runtime first:\n"
            "  scripts\\build_python_windows.bat\n"
            "  python scripts\\stage_python_package.py"
        )


class BinaryDistribution(Distribution):
    """Mark the wheel as platform-specific because it ships native binaries."""

    def has_ext_modules(self) -> bool:
        return True


_check_native_payload()
setup(distclass=BinaryDistribution)
