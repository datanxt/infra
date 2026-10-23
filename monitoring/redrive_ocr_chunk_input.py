#!/usr/bin/env python3
"""Re-drive documents stranded mid-pipeline by republishing `ocr_chunk_input`.

Runs entirely from the local machine over the VPN -- no ssh to the target box.
Reaches MinIO, the broker and SurrealDB directly, pulls each document's PDF,
and republishes the upload event exactly as document_service does in
``enhanced_upload_pipeline.py`` (send_thumbnail_and_ocr_chunking_tasks), with
FRESH batch/entity UUIDs -- never None, which would poison the milvus state.

Use when a document sits at ``chunks='processing'`` (or ``'error'``) because
its OCR batches were never consumed or died. Verify FIRST that the document has
zero chunks; if partial chunk data exists, clean it up before re-driving (see
the "full re-drive recipe" -- this script only handles the republish step).

Existing ``page`` rows are reused, not duplicated: the thumbnail service creates
them with ``hash = sha256(f"{doc_id}_page_{n}")``, the same key chunk_service
looks up.

Usage:
    # dry run (default) -- shows exactly what would be published
    python3 redrive_ocr_chunk_input.py --target nightly <doc_id> [<doc_id> ...]

    # actually publish
    python3 redrive_ocr_chunk_input.py --target nightly --execute <doc_id> ...

    # list candidates instead of re-driving
    python3 redrive_ocr_chunk_input.py --target nightly --list-stuck

REACHING MILVUS. The nightly Milvus publishes NO host port, so for `--target nightly`
MILVUS_HOST/PORT resolve to localhost, where nothing serves the document. Open a tunnel
to the container first and point the purge at it:

    ssh -i Masterkey_prod.pem -f -N -L 19533:172.24.0.15:19530 datanxt@10.8.0.5
    MILVUS_HOST=127.0.0.1 MILVUS_PORT=19533 python3 redrive_ocr_chunk_input.py \
        --target nightly --purge --execute <doc_id>

Without it the purge now ABORTS instead of reporting `milvus=none` and republishing on
top of the old vectors -- which is how one document ended up with 9,113 rows for 3,876
chunks (2026-10-22) and voided every measurement taken on that index.

DOCUMENTS WITHOUT reporting_scope. Chunks indexed before CHUNK_SCOPE_TAGGING carry no
scope, and nothing needs to be done about them: the retrieval-side scope filter fails
open when a document has no tags, so those documents retrieve exactly as they always
did (measured). Re-ingest only a document whose answers actually depend on telling the
consolidated statements from the parent company's -- a bank or insurer that publishes
both -- and never as a bulk pass over a whole collection.
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import re
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

try:
    import pika
    from minio import Minio
except ImportError as exc:  # pragma: no cover
    sys.exit(f"missing dependency: {exc} (need `pika` and `minio`)")

REPO = Path(__file__).resolve().parents[2]

# Per-target config sources. Each entry names the env files that hold the
# SurrealDB/MinIO settings and the broker settings for that stack. A
# "docker:<container>" source reads `printenv` of a running container instead.
TARGETS = {
    # Local dev stack (surtr-local): env taken live from the document_service
    # container. Its docker-internal hostnames (minio:9000 / infra-rabbitmq-1 /
    # redis) do not resolve from the host, so load_env rewrites them to the
    # published localhost ports (HOST_REWRITES).
    "local": {
        "doc_env": "docker:document_service-document_dev-1",
        "mq_env": "docker:document_service-document_dev-1",
        "surreal_url": "https://nightly-surreal.datanxt.de/sql",
    },
    "nightly": {
        "doc_env": REPO / "scratch_migrate/nightly_env/.env.document",
        "mq_env": REPO / "nightly_deployment/ocr_service/.env",
        # Moved 2026-10-01: dev/surtr-local AND dev/bulbasaur both live here now, and
        # the old host no longer serves bulbasaur ("database does not exist").
        "surreal_url": "https://nightly-surreal.datanxt.de/sql",
    },
}

BUCKET = "documents"
EXCHANGE = "document_events"
ROUTING_KEY = "ocr_chunk_input"
MAX_DOCUMENT_SIZE_MB = 100  # chunk_service rejects anything larger
FLUSH_TIMEOUT_S = 30  # Milvus flush after the row delete; see purge_document

# docker-internal hostname -> host-published port, applied to "docker:" env sources.
HOST_REWRITES = {"MINIO_ENDPOINT": "localhost:9000", "RABBITMQ_HOST": "localhost"}


def load_env(source: Path | str) -> dict[str, str]:
    if isinstance(source, str) and source.startswith("docker:"):
        container = source.split(":", 1)[1]
        proc = subprocess.run(["docker", "exec", container, "printenv"], capture_output=True, text=True)
        if proc.returncode != 0:
            sys.exit(f"cannot read env of container {container}: {proc.stderr.strip()}")
        text = proc.stdout
    else:
        path = Path(source)
        if not path.exists():
            sys.exit(f"env file not found: {path}")
        text = path.read_text()
    env: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key] = value.strip().strip('"').strip("'")
    if isinstance(source, str) and source.startswith("docker:"):
        env.update(HOST_REWRITES)
        if "REDIS_URL" in env:
            env["REDIS_URL"] = re.sub(r"://[^:/@]+", "://localhost", env["REDIS_URL"])
    return env


def surreal_query(cfg: dict, doc_env: dict[str, str], query: str, attempts: int = 6):
    """POST a SurrealQL statement over HTTP and return its first result set.

    A statement that SurrealDB rejects (status != OK) is retried with backoff and
    finally raised -- the HTTP call itself succeeds, so the old code silently returned
    the error text instead of rows. Seen 2026-09-19: the purge's
    `UPDATE document_processing SET chunks='processing'` was rejected with
    "Found NONE for field `tables` ... expected a string" on every pre-2026-09-17
    record (SCHEMAFULL field added without backfill), the record stayed chunks='done'
    and the driver accepted the doc without a single chunk record.
    """
    req = urllib.request.Request(cfg["surreal_url"], data=query.encode(), method="POST")
    auth = base64.b64encode(
        f"{doc_env['SURREALDB_USERNAME']}:{doc_env['SURREALDB_PASSWORD']}".encode()
    ).decode()
    for header, value in (
        ("Accept", "application/json"),
        ("Content-Type", "text/plain"),
        # SurrealDB wants these header names lowercase.
        ("surreal-ns", doc_env["SURREALDB_NAMESPACE"]),
        ("surreal-db", doc_env["SURREALDB_DATABASE"]),
        ("Authorization", "Basic " + auth),
    ):
        req.add_header(header, value)
    for attempt in range(1, attempts + 1):
        with urllib.request.urlopen(req, timeout=90) as resp:
            first = json.loads(resp.read())[0]
        if first.get("status", "OK") == "OK":
            return first.get("result")
        err = str(first.get("result"))
        if attempt == attempts:
            raise RuntimeError(f"SurrealQL error after {attempts} attempts: {err} -- {query[:200]}")
        print(f"surreal retry {attempt}/{attempts}: {err[:160]} -- {query[:80]}", file=sys.stderr)
        time.sleep(random.uniform(0.3, 1.0) * attempt)


def minio_client(doc_env: dict[str, str]) -> Minio:
    return Minio(
        doc_env["MINIO_ENDPOINT"],
        access_key=doc_env["MINIO_ACCESS_KEY"],
        secret_key=doc_env["MINIO_SECRET_KEY"],
        secure=doc_env.get("MINIO_SECURE", "false").lower() == "true",
    )


def find_object_key(client: Minio, doc_id: str) -> str | None:
    """Locate the stored PDF.

    The key is ``<doc_id>.<ext>`` where ext comes from splitting the ORIGINAL
    filename on its last dot -- a name ending in "(3216575000.51)" yields the
    key "<doc_id>.51)". So never assume ".pdf"; always list by prefix.
    """
    objects = list(client.list_objects(BUCKET, prefix=doc_id, recursive=True))
    return objects[0].object_name if objects else None


# --- purge (added 2026-09-15) -------------------------------------------------
# A bare re-drive DUPLICATES: chunk_service inserts fresh chunk rows next to the
# old ones in SurrealDB, and milvus embedding is a plain collection.insert
# (embedding_consumer.py:246), so the per-doc collection keeps every stale vector.
# --purge removes the document's chunk graph, its Milvus collection and its Redis
# coordination keys first, then resets document_processing so --list-stuck tracks
# the re-run. Order matters: edges before records (SurrealDB does not cascade).
DELETE_BATCH = 25          # larger /sql DELETEs 504 behind nginx
REDIS_KEY_PREFIX = "milvus:doc:"


def _sql_list(items: list[str]) -> str:
    return "[" + ",".join(items) + "]"


def collect_chunk_graph(cfg: dict, doc_env: dict[str, str], doc_id: str) -> dict:
    """Pages and chunks of *doc_id* via flat, indexed edge scans.

    Never the reverse traversal document<-page_of<-page<-starts_on_page<-chunk:
    it is O(edge scan) per hop and takes ~8s for a 784-chunk document.
    """
    pages = surreal_query(cfg, doc_env,
                          f"SELECT VALUE in FROM page_of WHERE out = document:{doc_id};") or []
    pages = [str(p) for p in pages]
    chunks: list[str] = []
    for i in range(0, len(pages), 15):
        rows = surreal_query(cfg, doc_env,
                             f"SELECT VALUE in FROM starts_on_page WHERE out IN {_sql_list(pages[i:i+15])};") or []
        chunks.extend(str(c) for c in rows)
    return {"pages": pages, "chunks": sorted(set(chunks))}


_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")
_RECORD_RE = re.compile(r"^[a-z_]+:[A-Za-z0-9_]{6,}$")


def _surql_value(v) -> str:
    """JSON (as read back from /sql) -> SurrealQL literal: datetimes and record ids cast back."""
    if v is None:
        return "NONE"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        if _ISO_RE.match(v):
            return f"<datetime>{json.dumps(v)}"
        if _RECORD_RE.match(v):
            return v
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(_surql_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{json.dumps(k)}: {_surql_value(x)}" for k, x in v.items()) + " }"
    return json.dumps(str(v))


def restore_document(cfg: dict, doc_env: dict[str, str], doc_id: str, workdir: Path) -> bool:
    """Recreate the `document` record and its owned_by / published_by / includes_document
    edges from redrive_snapshot_<doc>.json after chunk_service's permanent-OCR-failure
    cleanup deleted them (2026-09-19: CUDA OOM on the shared dev GPU -> 'Failed to
    convert PDF' -> permanent_error -> delete_document_data). Edge ids are not kept.
    Returns False when there is no usable snapshot."""
    snap = workdir / f"redrive_snapshot_{doc_id}.json"
    if not snap.exists():
        return False
    graph = json.loads(snap.read_text())
    doc = graph.get("document")
    if not isinstance(doc, dict) or str(doc.get("id")) != f"document:{doc_id}":
        return False
    if surreal_query(cfg, doc_env, f"SELECT VALUE id FROM document:{doc_id};"):
        return True  # already there
    content = {k: v for k, v in doc.items() if k != "id"}
    surreal_query(cfg, doc_env, f"CREATE document:{doc_id} CONTENT {_surql_value(content)};")
    for edge, rows in (graph.get("edges") or {}).items():
        for row in rows:
            fields = {k: v for k, v in row.items() if k not in ("id", "in", "out")}
            surreal_query(cfg, doc_env,
                          f"RELATE {row['in']}->{edge}->{row['out']} CONTENT {_surql_value(fields)};")
    print(f"{doc_id:24} RESTORED document record + edges from {snap.name} "
          f"({', '.join(f'{k}={len(v)}' for k, v in (graph.get('edges') or {}).items())})")
    return True


def _milvus_collection_candidates(doc_id: str) -> list[str]:
    """retrieval_engine._sanitize_collection_name over BOTH id forms the pipeline
    uses (bare id vs 'document:<id>'), since the message payload form varies."""
    import re as _re
    out = []
    for form in (doc_id, f"document:{doc_id}"):
        out.append("doc_" + _re.sub(r"[^a-zA-Z0-9_]", "_", form)[:200])
    return out


def ocr_in_flight(doc_env: dict[str, str], doc_id: str) -> bool:
    """True when chunk_service holds live OCR aggregation state for the doc
    (ocr:doc:<id> exists and ocr:doc:<id>:finalized does not)."""
    try:
        import redis as _redis
        r = _redis.Redis.from_url(doc_env.get("REDIS_URL", "redis://localhost:6379"))
        return bool(r.exists(f"ocr:doc:{doc_id}")) and not bool(r.exists(f"ocr:doc:{doc_id}:finalized"))
    except Exception:
        return False


def purge_document(cfg: dict, doc_env: dict[str, str], doc_id: str, *, execute: bool,
                   allow_missing_collection: bool = False,
                   workdir: Path = Path(".")) -> dict:
    """Delete everything a re-drive would otherwise duplicate. Dry run by default."""
    graph = collect_chunk_graph(cfg, doc_env, doc_id)
    chunks, pages = graph["chunks"], graph["pages"]
    report = {"doc_id": doc_id, "pages": len(pages), "chunks": len(chunks),
              "milvus_dropped": [], "redis_keys_deleted": 0, "executed": execute,
              "milvus_target": None}

    # chunk_service's permanent-OCR-failure cleanup (surreal_db_client.delete_document_data)
    # deletes the `document` record and its owned_by / includes_document / published_by
    # edges too (lost the 3-page test PDF on 2026-09-19 when MinIO was full). Keep enough
    # to recreate them by hand: CREATE document:<id> CONTENT {...}; RELATE ...
    graph["document"] = (surreal_query(cfg, doc_env, f"SELECT * FROM ONLY document:{doc_id};") or None)
    graph["edges"] = {
        edge: surreal_query(cfg, doc_env, f"SELECT * FROM {edge} WHERE in = document:{doc_id};") or []
        for edge in ("owned_by", "published_by")
    }
    graph["edges"]["includes_document"] = surreal_query(
        cfg, doc_env, f"SELECT * FROM includes_document WHERE out = document:{doc_id};") or []
    snap = workdir / f"redrive_snapshot_{doc_id}.json"
    snap.write_text(json.dumps(graph, indent=1, default=str))
    report["snapshot"] = str(snap)

    # Milvus: delete the document's ROWS, never drop the collection. milvus_service
    # caches Collection handles per process (simplified_collection_setup._COLLECTION_CACHE)
    # and an insert through a handle whose collection was dropped externally RAISES
    # "collection not found" (verified 2026-09-15) -- i.e. every replica with a warm
    # cache would lose the re-driven chunks. Row deletion keeps handles valid.
    # NOTE: this box runs TWO Milvus instances; host port 19530 is the OLD v2.4
    # standalone, the live hybrid v2.6 one milvus_service uses is on 19531.
    try:
        from pymilvus import Collection, connections, utility  # lazy: not needed for plain re-drives
        host = doc_env.get("MILVUS_HOST", "localhost")
        port = doc_env.get("MILVUS_PORT", "19531")
        connections.connect(alias="redrive", host=host, port=port)
        report["milvus_target"] = f"{host}:{port}"
        found_any = False
        for name in _milvus_collection_candidates(doc_id):
            if not utility.has_collection(name, using="redrive"):
                continue
            found_any = True
            coll = Collection(name, using="redrive")
            coll.load()
            rows = coll.query(expr='chunk_id != ""', output_fields=["chunk_id"], limit=16384)
            ids = [str(r["chunk_id"]) for r in rows]
            if execute and ids:
                for i in range(0, len(ids), 500):
                    quoted = ", ".join('"' + x.replace('"', '') + '"' for x in ids[i:i + 500])
                    coll.delete(expr=f"chunk_id in [{quoted}]")
                # Deletes are visible as soon as delete() returns; flush only persists the
                # delta. On the dev hybrid 2.6.19 the Flush RPC can hang for a collection
                # whose rows were all deleted (2026-09-19, doc 7m8emhde50emgvm7gjrg: every
                # other collection flushed in 0.5 s) and pymilvus retries it for >10 min
                # without a timeout. Bound it and verify emptiness by query instead.
                try:
                    coll.flush(timeout=FLUSH_TIMEOUT_S)
                except Exception as exc:
                    report.setdefault("milvus_flush_timeout", []).append(f"{name}: {type(exc).__name__}")
                left = coll.query(expr='chunk_id != ""', output_fields=["chunk_id"], limit=1)
                if left:
                    raise RuntimeError(f"{name}: rows still present after delete")
            report["milvus_dropped"].append(f"{name}:{len(ids)} rows")
        connections.disconnect("redrive")
        # A document whose collection is ABSENT is indistinguishable from one that was
        # never embedded -- and the far more likely cause is that this process reached the
        # wrong Milvus. MEASURED 2026-10-22: this box runs two instances, the purge
        # reported `milvus=none` with no error, the re-drive proceeded, and the collection
        # kept its 3,880 stale vectors beside the 3,876 new ones (9,113 rows for 3,876
        # chunks). Every measurement taken on that index was void. So silence is not an
        # option: say which instance was asked, and make the operator decide.
        if not found_any:
            if allow_missing_collection:
                report["milvus_dropped"].append(
                    f"no collection at {host}:{port} (allowed by --allow-missing-collection)")
            else:
                report["milvus_error"] = (
                    f"no collection for this document at {host}:{port} "
                    f"(tried {', '.join(_milvus_collection_candidates(doc_id))}). "
                    f"Either this is the wrong Milvus instance or the document was never "
                    f"embedded; re-run with --allow-missing-collection if you are sure.")
    except ImportError:
        report["milvus_error"] = "pymilvus not installed; collection NOT dropped"
    except Exception as exc:  # report, do not abort the SurrealDB part
        report["milvus_error"] = f"{type(exc).__name__}: {exc}"

    # Redis coordination keys: milvus_service (milvus:doc:<id>:*) AND chunk_service's
    # OCR aggregation state (ocr:doc:<id>*, ocr:response:seen:<id>:*). The latter
    # matters: response dedup is keyed by (doc_id, batch_id) with a 24h TTL, so a
    # concurrent second dispatch of the same document makes the aggregator drop one
    # run's batches as duplicates and NEVER finalize (AMCOR, 2026-09-15: 0 chunks).
    try:
        import redis as _redis
        r = _redis.Redis.from_url(doc_env.get("REDIS_URL", "redis://localhost:6379"))
        keys = [k for k in r.scan_iter(match=f"{REDIS_KEY_PREFIX}*{doc_id}*", count=500)]
        keys += [k for k in r.scan_iter(match=f"ocr:doc:{doc_id}*", count=500)]
        keys += [k for k in r.scan_iter(match=f"ocr:response:seen:{doc_id}:*", count=500)]
        if execute and keys:
            r.delete(*keys)
        report["redis_keys_deleted"] = len(keys)
    except ImportError:
        report["redis_error"] = "redis-py not installed; coordination keys NOT cleared"
    except Exception as exc:
        report["redis_error"] = f"{type(exc).__name__}: {exc}"

    if not execute:
        return report

    # SurrealDB: delete the chunk RECORDS only, in record-target form.
    # Measured on dev 2026-09-15 (162k chunks, 162k part_of_section rows):
    #   DELETE chunk WHERE id IN [25 ids]          13.4 s  (a WHERE clause is a full scan, even on ids)
    #   DELETE chunk:a, chunk:b, ...   (25 ids)    50 ms   (direct record targets)
    #   DELETE part_of_section WHERE in IN [...]   6.5-47 s (no index on `in`)
    # and SurrealDB CASCADES: deleting an endpoint record removes its RELATE edges
    # (verified: starts_on_page / part_of_section rows vanish with the chunk), so no
    # edge statements are needed at all.
    for i in range(0, len(chunks), DELETE_BATCH):
        surreal_query(cfg, doc_env, "DELETE " + ", ".join(chunks[i:i+DELETE_BATCH]) + ";")
    surreal_query(cfg, doc_env,
                  f"UPDATE document_processing SET chunks = 'processing', updated_at = time::now() "
                  f"WHERE document_id = document:{doc_id};")
    status = surreal_query(cfg, doc_env,
                           f"SELECT VALUE chunks FROM document_processing WHERE document_id = document:{doc_id};")
    if status != ["processing"]:
        raise RuntimeError(f"document_processing.chunks is {status!r} after the purge, expected ['processing']")
    return report


def list_stuck(cfg: dict, doc_env: dict[str, str]) -> None:
    rows = surreal_query(
        cfg,
        doc_env,
        "SELECT document_id, overall_status, chunks, thumbnails, visuals, updated_at "
        "FROM document_processing WHERE chunks != 'done' ORDER BY updated_at;",
    ) or []
    if not rows:
        print("no documents with chunks != 'done'")
        return
    print(f"{'document_id':40} {'chunks':<12} {'thumbnails':<12} {'visuals':<12} updated_at")
    for row in rows:
        print(
            f"{str(row['document_id']):40} {str(row.get('chunks')):<12} "
            f"{str(row.get('thumbnails')):<12} {str(row.get('visuals')):<12} {row.get('updated_at')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("doc_ids", nargs="*", help="bare document ids, without the 'document:' prefix")
    parser.add_argument("--target", default="nightly", choices=sorted(TARGETS), help="which stack to act on")
    parser.add_argument("--execute", action="store_true", help="actually publish (default is a dry run)")
    parser.add_argument("--purge", action="store_true",
                        help="delete the doc's chunks/edges, its Milvus collection and Redis keys before re-driving")
    parser.add_argument("--allow-missing-collection", action="store_true",
                        help="purge even though no Milvus collection exists for the document "
                             "(use only when you know it was never embedded)")
    parser.add_argument("--skip-publish", action="store_true",
                        help="with --purge --execute: purge only, do not republish")
    parser.add_argument("--force", action="store_true",
                        help="dispatch even if chunk_service shows OCR still in flight for the doc "
                             "(double dispatch corrupts aggregation; only use after purging its state)")
    parser.add_argument("--list-stuck", action="store_true", help="list documents with chunks != 'done' and exit")
    parser.add_argument("--footprint", default=None, help="footprint to stamp on the events")
    parser.add_argument("--env-file", default=None,
                        help="env file (SurrealDB/MinIO/broker/Redis settings) overriding the target's sources")
    parser.add_argument("--workdir", type=Path, default=Path("."),
                        help="directory for the redrive_snapshot_<doc>.json files (default: CWD)")
    args = parser.parse_args()

    cfg = dict(TARGETS[args.target])
    if args.env_file:
        cfg["doc_env"] = cfg["mq_env"] = args.env_file
    doc_env = load_env(cfg["doc_env"])

    if args.list_stuck:
        list_stuck(cfg, doc_env)
        return 0

    if not args.doc_ids:
        parser.error("give at least one doc_id, or use --list-stuck")

    doc_ids = [d.split(":", 1)[-1] for d in args.doc_ids]
    mq_env = load_env(cfg["mq_env"])
    client = minio_client(doc_env)

    # Pull name + publication metadata so the republished event matches the original.
    thing_list = ",".join(f"document:{d}" for d in doc_ids)
    rows = surreal_query(
        cfg,
        doc_env,
        f"SELECT id, name, publication_date, publication_date_source FROM document WHERE id IN [{thing_list}];",
    ) or []
    meta = {str(r["id"]).split(":", 1)[1]: r for r in rows}

    missing = [d for d in doc_ids if d not in meta]
    if missing:
        # chunk_service's nuclear cleanup may have deleted the record; bring it back
        # from this workdir's snapshot before giving up.
        restored = [d for d in missing if restore_document(cfg, doc_env, d, args.workdir)]
        if restored:
            rows = surreal_query(cfg, doc_env,
                                 "SELECT id, name, publication_date, publication_date_source FROM document "
                                 f"WHERE id IN [{','.join(f'document:{d}' for d in restored)}];") or []
            meta.update({str(r["id"]).split(":", 1)[1]: r for r in rows})
            missing = [d for d in doc_ids if d not in meta]
    if missing:
        sys.exit(f"not found in SurrealDB: {', '.join(missing)}")

    footprint = args.footprint or f"redrive-{uuid.uuid4().hex[:8]}"

    channel = connection = None
    if args.execute:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(
                host=mq_env["RABBITMQ_HOST"],
                port=int(mq_env.get("RABBITMQ_PORT", 5672)),
                credentials=pika.PlainCredentials(mq_env["RABBITMQ_USER"], mq_env["RABBITMQ_PASSWORD"]),
                heartbeat=600,
                blocked_connection_timeout=300,
            )
        )
        channel = connection.channel()
        channel.confirm_delivery()

    print(f"target={args.target}  mode={'EXECUTE' if args.execute else 'DRY RUN'}  footprint={footprint}")
    print(f"{'doc_id':24} {'pdf_MB':>7}  key / batch_uuid")

    published = 0
    for doc_id in doc_ids:
        key = find_object_key(client, doc_id)
        if not key:
            print(f"{doc_id:24} {'--':>7}  SKIPPED: no object in bucket '{BUCKET}'")
            continue

        if ocr_in_flight(doc_env, doc_id) and not args.force:
            print(f"{doc_id:24} {'--':>7}  SKIPPED: OCR aggregation still in flight "
                  f"(ocr:doc:{doc_id} without :finalized). A second dispatch is dropped as "
                  f"duplicate and neither run finalizes. Re-run with --purge --force to reset it.")
            continue

        if args.purge:
            # Only purge what can be regenerated: the PDF must exist (checked above).
            rep = purge_document(cfg, doc_env, doc_id, execute=args.execute, workdir=args.workdir,
                                 allow_missing_collection=args.allow_missing_collection)
            mode = "PURGED" if args.execute else "would purge"
            print(f"{doc_id:24} {mode}: pages={rep['pages']} chunks={rep['chunks']} "
                  f"milvus={rep['milvus_dropped'] or 'none'}@{rep.get('milvus_target') or '?'} "
                  f"redis_keys={rep['redis_keys_deleted']} "
                  f"snapshot={rep['snapshot']}"
                  + (f"  !! milvus: {rep['milvus_error']}" if rep.get('milvus_error') else "")
                  + (f"  (milvus flush timed out, rows verified gone: {rep['milvus_flush_timeout']})"
                     if rep.get('milvus_flush_timeout') else "")
                  + (f"  !! redis: {rep['redis_error']}" if rep.get('redis_error') else ""))
            if args.skip_publish:
                continue
            if rep.get("milvus_error") or rep.get("redis_error"):
                # Publishing on top of a half-done purge is exactly the duplication
                # --purge exists to prevent (stale vectors / stale ocr:doc:* keys).
                print(f"{doc_id:24} {'--':>7}  SKIPPED: purge incomplete (see !! above), not republished")
                continue

        response = client.get_object(BUCKET, key)
        try:
            raw = response.read()
        finally:
            response.close()
            response.release_conn()

        size_mb = len(raw) / 1048576
        if size_mb > MAX_DOCUMENT_SIZE_MB:
            print(f"{doc_id:24} {size_mb:7.2f}  SKIPPED: exceeds MAX_DOCUMENT_SIZE_MB={MAX_DOCUMENT_SIZE_MB}")
            continue

        row = meta[doc_id]
        batch_uuid = str(uuid.uuid4())
        entity_uuid = str(uuid.uuid4())
        event = {
            "doc_id": doc_id,
            "filename": row["name"],
            "pdf_base64": base64.b64encode(raw).decode("ascii"),
            "document_batch_uuid": batch_uuid,
            "document_entity_uuid": entity_uuid,
            "footprint": footprint,
            "event_id": entity_uuid,
            "publication_date": row.get("publication_date"),
            "publication_date_source": row.get("publication_date_source") or "default_system_date",
            # A re-drive has no live WS session, so there is nowhere to push
            # upload-progress updates. The pipeline completes without them.
            "session_id": None,
            "task_id": None,
            "username": None,
            "status_update_routing": None,
            "temp_doc_id": None,
        }

        print(f"{doc_id:24} {size_mb:7.2f}  {key} / {batch_uuid}")

        if args.execute:
            channel.basic_publish(
                exchange=EXCHANGE,
                routing_key=ROUTING_KEY,
                body=json.dumps(event).encode("utf-8"),
                properties=pika.BasicProperties(
                    content_type="application/json",
                    delivery_mode=2,  # persistent
                    correlation_id=batch_uuid,
                ),
            )
            published += 1

    if args.execute:
        connection.close()
        print(f"\npublished {published} message(s) to {EXCHANGE} -> {ROUTING_KEY}")
    else:
        print("\ndry run only; re-run with --execute to publish")
    return 0


if __name__ == "__main__":
    sys.exit(main())
