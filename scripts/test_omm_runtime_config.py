"""Exercise the real CMake OMM runtime validator with isolated SDK layouts."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True, help="Matching Windows OMM DLL")
    args = parser.parse_args()
    module = Path(__file__).resolve().parents[1]/"src/cmake/omm_runtime.cmake"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sdk = root/"sdk"
        (sdk/"bin").mkdir(parents=True)
        runtime = sdk/"bin"/"omm-lib.dll"
        shutil.copy2(args.runtime, runtime)
        bad = root/"bad.dll"
        bad.write_bytes(b"invalid dll"*10)
        empty = root/"empty.dll"
        empty.touch()
        cases = [
            ("discover", "", False, True),
            ("explicit", runtime.as_posix(), False, True),
            ("missing", (root/"missing.dll").as_posix(), False, False),
            ("directory", sdk.as_posix(), False, False),
            ("empty", empty.as_posix(), False, False),
            ("invalid_binary", bad.as_posix(), False, False),
            ("static_ignores_runtime", (root/"missing.dll").as_posix(), True, True),
        ]
        for name, path, static, expected in cases:
            script = root/(name+".cmake")
            script.write_text(f'''set(WIN32 TRUE)
set(PHOTARA_OMM_ROOT "{sdk.as_posix()}")
set(PHOTARA_OMM_LIBRARY "{sdk.as_posix()}/lib/omm-lib.lib")
set(PHOTARA_OMM_RUNTIME "{path}")
set(PHOTARA_OMM_STATIC {"ON" if static else "OFF"})
include("{module.as_posix()}")
photara_validate_omm_runtime()
''')
            result = subprocess.run(["cmake", "-P", str(script)], capture_output=True, text=True)
            if (result.returncode == 0) != expected:
                raise AssertionError(f"{name}: {result.stdout}\n{result.stderr}")
            print(f"PASS {name}")
        runtime.unlink()
        result = subprocess.run(["cmake", "-P", str(root/"discover.cmake")], capture_output=True)
        if result.returncode == 0:
            raise AssertionError("Discovery succeeded without an SDK runtime")
        print("PASS missing_discovered_runtime")


if __name__ == "__main__":
    main()
