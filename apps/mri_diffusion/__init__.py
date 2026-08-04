"""SS2D-Mamba diffusion MRI reconstruction on Arm CPU.

Importing this package puts the repo's `python/` directory on `sys.path` so
`import arm_scan` resolves from a plain source checkout, with no install step
and no per-caller `sys.path` incantation. A real installed `arm_scan` (wheel)
already on the path always wins — this only appends a fallback.
"""

import sys
from pathlib import Path

_PY = Path(__file__).resolve().parents[2] / "python"
if _PY.is_dir() and str(_PY) not in sys.path:
    sys.path.append(str(_PY))
