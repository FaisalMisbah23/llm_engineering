import subprocess
import tempfile
from typing import Dict, Union


def compile_cpp(code: str, std: str = "c++17") -> Dict[str, Union[str, bool]]:
    with tempfile.TemporaryDirectory() as d:
        src = f"{d}/main.cpp"
        out = f"{d}/main.o"
        with open(src, "w", encoding="utf-8") as f:
            f.write(code)

        p = subprocess.run(
            ["g++", f"-std={std}", "-c", src, "-O2", "-o", out],
            capture_output=True,
            text=True,
        )
        return {"ok": p.returncode == 0, "stdout": p.stdout, "stderr": p.stderr}
