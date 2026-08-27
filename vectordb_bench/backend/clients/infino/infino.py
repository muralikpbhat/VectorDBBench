import logging
import os
import tempfile
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

import infino
import numpy as np
import pyarrow as pa

from ..api import VectorDB
from .config import InfinoIndexConfig

log = logging.getLogger(__name__)

_VECTOR_FIELD = "emb"
_ID_FIELD = "id"

# VectorDBBench feeds rows in small batches (its default is 100 rows), and each
# Infino append() commits a superfile. Committing one superfile per fed batch
# would fragment a large load into thousands of tiny superfiles, making both the
# load and the following optimize pathologically slow. insert_embeddings instead
# buffers fed rows and commits them as one combined append once this many have
# accumulated; the remainder is flushed when the load's init() scope exits, so a
# corpus smaller than this threshold is still fully persisted. The result is a
# handful of large superfiles regardless of the harness batch size.
_FLUSH_ROWS = 100_000


class Infino(VectorDB):
    """VectorDBBench client for Infino, an embedded vector/search engine.

    Infino is in-process: each benchmark worker connects to the same on-disk
    catalog. The instance holds only picklable config so it survives the
    ProcessPoolExecutor(spawn) boundary; the connection and table are opened
    lazily in init().
    """

    # Serialize the load: concurrent writes to a single table are not supported.
    thread_safe: bool = False

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: InfinoIndexConfig,
        collection_name: str = "vdbbench_infino",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "Infino"
        self.dim = dim
        self.data_path = db_config["data_path"]
        # A cache budget without a cache dir is a silent no-op in the
        # engine (no disk cache is created); default the cache next to the
        # catalog so warm queries are actually warm.
        if db_config.get("cache_budget_bytes") and not db_config.get("cache_dir"):
            db_config = {**db_config, "cache_dir": str(Path(self.data_path) / "cache")}
        # Connection tuning (cache budget, cache dir, object-store options); pass only what is set.
        self._connect_opts = {
            k: db_config[k]
            for k in ("cache_budget_bytes", "cache_dir", "storage_options")
            if db_config.get(k) is not None
        }
        self.table_name = collection_name
        self.metric = db_case_config.index_param()["metric"]
        # Vector serving path + serve-time beam, bridged to the engine config
        # before connect (see _apply_search_mode_config).
        self._search_mode = db_case_config.search_mode
        self._ef = db_case_config.ef

        self._conn = None
        self._table = None
        # _id -> dataset-id arrays; None until built (init() skips the build
        # when the table is empty — the load-phase open) and after unpickling.
        self._map_keys = None
        self._map_vals = None
        # Rows accumulated by insert_embeddings as Arrow batches, committed as one
        # large append at _FLUSH_ROWS and when the load's init() scope exits.
        self._buf_batches: list[pa.RecordBatch] = []
        self._buf_rows = 0
        # Build the schema once so table creation and every append stay in lockstep.
        self._schema = self._build_schema()

        # Open the connection once and keep it: create/drop the table here, and
        # reuse the same handle in init() (reopening is costly and can deadlock).
        # __getstate__ drops it so a spawned worker reopens its own.
        Path(self.data_path).mkdir(parents=True, exist_ok=True)
        self._conn = self._connect()
        if drop_old:
            # A drop invalidates the persisted _id map: a fresh ingest reassigns
            # engine _ids, so the map (trusted whenever its row count matches)
            # would be wrongly reused after re-ingesting the same number of rows.
            self._id_map_path().unlink(missing_ok=True)
            if self.table_name in self._conn.list_tables():
                self._conn.drop_table(self.table_name, purge=True)
        if self.table_name not in self._conn.list_tables():
            self._conn.create_table(self.table_name, self._schema, self._index_spec())

    def _apply_search_mode_config(self):
        """Bridge ``search_mode`` and the serve-time beam to the engine's config.

        The engine reads ``vector.search_mode`` and ``vector.hnsw_ef_search`` from
        ``$XDG_CONFIG_HOME/infino/config.yaml``; ``connect()`` has no equivalent
        keywords, so a config file is the only way to select them without touching
        the engine's public API. The engine loads that config lazily and caches
        it for the process's lifetime, so the ``XDG_CONFIG_HOME`` override must
        persist (it cannot be restored right after connect) — acceptable here
        because each benchmark worker is a dedicated infino process. The engine
        default is ``ivf`` with the stamped k->ef curve, so only values that
        diverge from that are written and a pure-default run is byte-for-byte
        unchanged. Idempotent; re-applied in each spawned worker before its first
        connect.
        """
        mode = self._search_mode
        ef = self._ef or 0
        write_mode = bool(mode) and mode != "ivf"
        if not write_mode and ef <= 0:
            return
        lines = ["vector:"]
        if write_mode:
            lines.append(f"  search_mode: {mode}")
        if ef > 0:
            lines.append(f"  hnsw_ef_search: {ef}")
        cfg_root = Path(self.data_path) / "_infino_engine_cfg"
        (cfg_root / "infino").mkdir(parents=True, exist_ok=True)
        (cfg_root / "infino" / "config.yaml").write_text("\n".join(lines) + "\n")
        os.environ["XDG_CONFIG_HOME"] = str(cfg_root)

    def _connect(self):
        self._apply_search_mode_config()
        return infino.connect(self.data_path, **self._connect_opts)

    def __getstate__(self) -> dict:
        # Drop the non-picklable live connection so the instance can cross a
        # process boundary. The buffer is always empty at a process boundary
        # (the load subprocess flushes at init() exit before returning), so it
        # is reset rather than shipped.
        return {
            **self.__dict__,
            "_conn": None,
            "_table": None,
            "_map_keys": None,
            "_map_vals": None,
            "_buf_batches": [],
            "_buf_rows": 0,
        }

    def _build_schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field(_ID_FIELD, pa.int64(), nullable=False),
                pa.field(_VECTOR_FIELD, pa.list_(pa.float32(), self.dim), nullable=False),
            ]
        )

    def _index_spec(self) -> infino.IndexSpec:
        return infino.IndexSpec().vector(_VECTOR_FIELD, self.dim, self.metric)

    @contextmanager
    def init(self):
        # Reuse one connection for the whole process: reopening is costly and can
        # deadlock. __init__ opens it in the constructing process; a spawned
        # worker (unpickled with _conn=None) opens its own here, once.
        if self._conn is None:
            self._conn = self._connect()
        if self._table is None:
            self._table = self._conn.open_table(self.table_name)
            self._load_or_build_id_map()
        try:
            yield
        finally:
            # Commit any rows still buffered from the load. This runs in the same
            # (sub)process that inserted them, before it returns — the only point
            # at which a load smaller than _FLUSH_ROWS would otherwise never be
            # persisted. A no-op outside the load path (the buffer is empty).
            self._flush()

    # _id -> dataset-id translation, the same build-once / persist / reload
    # pattern the engine's own bench uses for its ground-truth bin: one scan
    # per TABLE (not per process), stored beside the catalog, reloaded by
    # every worker in well under a second. _ids are 128-bit decimals, so the
    # sorted key array is 16-byte big-endian bytes (lexicographic == numeric).
    #
    # The build lives in init() ON PURPOSE: init() runs before the timed
    # search loops, so the scan/reload never lands inside a measured query.
    # Two guards keep that placement safe: an empty table (the loader opens
    # the connection before inserting a single row) builds and persists
    # NOTHING, and a cached map is only trusted if its row count matches the
    # table — otherwise it is rebuilt in place.
    def _id_map_path(self) -> Path:
        return Path(self.data_path) / f"{self.table_name}.idmap.npz"

    def _table_row_count(self) -> int:
        res = self._conn.query_sql(f"SELECT COUNT(*) FROM {self.table_name}")
        return int(res.column(0).to_pylist()[0])

    def _load_or_build_id_map(self) -> None:
        n_rows = self._table_row_count()
        if n_rows == 0:
            # Load-phase init(): drop_old has just recreated the table and no
            # rows exist yet. Persisting an empty map here would poison every
            # later search process (the cache is trusted once written).
            return
        path = self._id_map_path()
        if path.exists():
            data = np.load(path)
            if len(data["keys"]) == n_rows:
                self._map_keys, self._map_vals = data["keys"], data["vals"]
                return
            # Row count mismatch: the cache belongs to a previous incarnation
            # of the table (dropped and reloaded at a different size). Fall
            # through and rebuild; os.replace keeps concurrent rebuilds safe.
        m = self._conn.query_sql(f"SELECT _id, {_ID_FIELD} FROM {self.table_name}")
        keys = np.array(
            [int(v).to_bytes(16, "big") for v in m.column("_id").to_pylist()],
            dtype="S16",
        )
        vals = np.array(m.column(_ID_FIELD).to_pylist(), dtype=np.int64)
        order = np.argsort(keys)
        keys, vals = keys[order], vals[order]
        # Suffix must end in .npz or np.savez appends it and orphans the file.
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp.npz")
        os.close(fd)
        np.savez(tmp, keys=keys, vals=vals)
        Path(tmp).replace(path)
        self._map_keys, self._map_vals = keys, vals

    def _to_dataset_ids(self, stable_ids: list) -> list[int]:
        if not stable_ids:
            return []
        if self._map_keys is None:
            # Only reachable in a process that opened the connection while the
            # table was still empty (load) and then searched (read-write
            # cases). Ordinary search processes built the map in init().
            self._load_or_build_id_map()
        if self._map_keys is None:
            msg = (
                f"table {self.table_name!r} has no rows to map; the load stage "
                "did not run (or wrote nothing) before search"
            )
            raise RuntimeError(msg)
        q = np.array([int(v).to_bytes(16, "big") for v in stable_ids], dtype="S16")
        n = len(self._map_keys)
        idx = np.searchsorted(self._map_keys, q)
        # Every returned _id must be a key we mapped; a miss means the cached
        # map belongs to a different table state — fail loudly, wrong ids
        # here silently corrupt recall.
        if (idx >= n).any() or (self._map_keys[idx.clip(max=n - 1)] != q).any():
            msg = (
                f"search returned _ids absent from the id map for {self.table_name!r}; "
                f"stale cache at {self._id_map_path()} — delete it and rerun"
            )
            raise RuntimeError(msg)
        return self._map_vals[idx].tolist()

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception | None]:
        # Buffer fed rows and commit them as a few large superfiles rather than
        # one per call (see _FLUSH_ROWS); the remainder is flushed at init() exit
        # so every fed row is persisted before search.
        try:
            arrays = [
                pa.array(metadata, type=pa.int64()),
                pa.array(embeddings, type=pa.list_(pa.float32(), self.dim)),
            ]
            self._buf_batches.append(pa.record_batch(arrays, schema=self._schema))
            self._buf_rows += len(metadata)
            if self._buf_rows >= _FLUSH_ROWS:
                self._flush()
        except Exception as e:
            log.exception("Failed to insert embeddings into Infino")
            return 0, e
        return len(metadata), None

    def _flush(self) -> None:
        """Commit all buffered rows as a single superfile and clear the buffer.

        Concatenates the buffered batches into one contiguous batch so the load
        commits large superfiles. Inserts are serialized (the client is not
        thread-safe, so the runner drives a single worker), so no lock is needed.
        """
        if not self._buf_batches:
            return
        table = pa.Table.from_batches(self._buf_batches, schema=self._schema).combine_chunks()
        for batch in table.to_batches():
            self._table.append(batch)
        self._buf_batches = []
        self._buf_rows = 0

    def search_embedding(self, query: list[float], k: int = 100, **kwargs) -> list[int]:
        # Vector serving is engine-decided; the call carries no tuning kwargs.
        hits = self._table.vector_search(_VECTOR_FIELD, query, k)
        return self._to_dataset_ids(hits.column("_id").to_pylist())

    def optimize(self, data_size: int | None = None):
        with self.init():
            self._table.optimize()

    def need_normalize_cosine(self) -> bool:
        return True
