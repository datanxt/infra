# infra

Operations tooling that spans services and therefore belongs to no single one.

Nothing here is a service: these scripts are run by a human against a deployed
environment. That matters for the architecture guards in `tests/` -- `infra` is
exempt from the Milvus and DuckDB isolation rules for the same reason the shared
DuckDB server container is, so a repair tool may talk to a datastore directly.

## monitoring/

`redrive_ocr_chunk_input.py` -- re-drive a document stranded mid-pipeline by
purging what it produced (pages, chunks, Milvus rows, Redis coordination keys)
and republishing its upload event. Read the module docstring before running it:
it covers the dry-run default, the SSH tunnel the nightly Milvus needs, and why
a purge that cannot verify the Milvus side refuses to republish.
