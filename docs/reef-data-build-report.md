# Reef data asset build report

Scope: immutable data packaging and dataset documentation only. No GPU, model,
notebook execution, publication, commit, or push was performed by this data task.

The five downloaded source files match the original audit SHA-256 and sizes.
The archive preserves original member bytes, fixes ZIP metadata and member order,
and contains no executable payload. The original audit is copied byte-for-byte.

The independent CPU verification result is recorded in
[reef-data-build-report.json](reef-data-build-report.json), including per-region
calendar gaps and complete-window DHW reconstruction error. A second build in a
temporary directory must reproduce both archive and manifest exactly.

The notebook's embedded immutable data archive is an explicit amendment to the
proposed remote data download; the source URLs remain provenance, not immutable
download pins. Hosted qualification remains outside this task.
