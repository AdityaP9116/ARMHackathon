"""SS2D-Mamba diffusion MRI reconstruction — DEMOTED, not abandoned.

This is no longer on the project's critical path (see docs/project/PROJECT_CONCEPT.md, the
Aug 6-7 amendment). It is kept, and kept CI-gated, for one concrete reason: it
is the **only end-to-end exercise of the SS2D cross-scan kernel** in the repo,
so deleting it would remove that coverage rather than merely tidy up.

Known state, so nobody rediscovers it: the Phase-D quality gate has never
passed, and there is no trained prior. Both are diagnosed in
docs/archive/PHASE_D_DIAGNOSIS.md, which is worth reading before touching the
sampler — the sampler itself is provably exact (an oracle denoiser reconstructs
to 151 dB); the causes are the evaluation data, the mask, and a sigma range
outside the prior's trained support.

Plan of record for this app: docs/archive/MRI_DIFFUSION_IMPLEMENTATION_PLAN.md
"""
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
