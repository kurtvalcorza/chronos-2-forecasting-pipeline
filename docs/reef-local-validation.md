# Reef capstone local validation — 2026-09-28

Candidate only. No hosted GPU/model qualification claimed.

- Pre-change non-integration suite: exit 0 after setting `PYTHONPATH=src`.
- Final repository suite: **407 passed, 22 integration tests deselected**, exit 0,
  158.72 seconds. The 23 new reef tests cover 15 core cases and 8 notebook/integration cases.
- Final source lint (`ruff check src tests tools`): pass.
- Release asset validator: pass after registering the new deterministic notebook.
- Existing single-model notebook build `--check`: pass.
- Reef notebook format validation, cell parsing, deterministic generation, assembled namespace,
  embedded-data integrity, and optional BYOD code compilation: pass.
- Actual NOAA CPU baseline execution: 299 eligible region-origin pairs, two arms, 28 days each.
- Chronos config SHA and model safetensors digest checked against the immutable upstream revision.
- Snapshot member checksums and actual calendar-aware DHW checks verified in root-run tests.
- Export/report integration uses an **explicit model double**. Fresh-process real-model parity
  exists in the default notebook but has not been executed with real weights in this work.

The full suite emitted 233 upstream Matplotlib/NumPy date deprecation warnings; no test failed.
Local package versions differ from the isolated Colab lock. Hosted testing remains necessary.

After the full suite, the setup cell was changed to pinned uv environment creation/sync to avoid
depending on Colab's system ensurepip support. Final deterministic/format validation and BYOD
compilation were rerun (2 passed), as were lint and release asset checks. No scientific calculation
changed in that final setup adjustment.

The reviewer identified missing independent metric parity and missing pre-test resource gating;
both were implemented and checked. Stage timing now includes weight acquisition/model loading.
Runtime projection remains an estimate, not complete-run timing evidence.

No commit, push, PR, or merge performed for this build.
