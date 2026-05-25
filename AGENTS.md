# AGENTS.md

This project implements a tensor contraction benchmark suite.

Rules:
- Follow docs/spec.md.
- Make small, reviewable changes.
- Do not implement multiple milestones in one change.
- Prefer correctness and reproducibility over performance in early milestones.
- Every benchmark result must conform to schemas/result.schema.json.
- Every benchmark case must conform to schemas/case.schema.json.
- Do not add heavy dependencies without explaining why.
- TBLIS support must be optional behind TCB_ENABLE_TBLIS.
- GPU support is out of scope until the C++ reference and TBLIS backends are stable.
