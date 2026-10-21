"""Local dashboard server for business entity resolution results."""

from __future__ import annotations

import csv
import json
import logging
import os
import sqlite3
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qs, unquote, urlsplit


ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = ROOT / "output" / "matching_results.tsv"
SOURCE_DATASETS = {
    "source1": ROOT / "dataset" / "test" / "test_source1.tsv",
    "source2": ROOT / "dataset" / "test" / "test_source2.tsv",
    "source3": ROOT / "dataset" / "test" / "test_source3.tsv",
}
STATIC_PATH = Path(__file__).resolve().parent / "static"
CACHE_DIR = Path(tempfile.gettempdir()) / "business_entity_resolution_ui"
CACHE_PATH = CACHE_DIR / "results.sqlite3"
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200
BATCH_SIZE = 5000
SCHEMA_VERSION = "2"
LOGGER = logging.getLogger("business_entity_resolution_ui")


def split_matches(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_dataset_rows(path: Path) -> Iterator[tuple[str, str, str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header != ["entity_id", "business_name", "business_address", "country"]:
            raise ValueError(f"Unexpected columns in {path.name}")
        for row in reader:
            if len(row) < 2 or not row[0].strip():
                continue
            yield (
                row[0].strip(),
                row[1].strip() or "Name unavailable",
                row[2].strip() if len(row) > 2 else "",
                row[3].strip() if len(row) > 3 else "",
            )


def insert_batches(
    connection: sqlite3.Connection,
    table: str,
    path: Path,
    source: str | None = None,
) -> int:
    count = 0
    batch: list[tuple[str, ...]] = []
    for entity_id, business_name, address, country in read_dataset_rows(path):
        batch.append(
            (entity_id, business_name, address, country, source)
            if source
            else (entity_id, business_name, address, country)
        )
        if len(batch) == BATCH_SIZE:
            if source:
                connection.executemany(
                    f"INSERT OR REPLACE INTO {table}(entity_id, business_name, business_address, country, source) VALUES (?, ?, ?, ?, ?)",
                    batch,
                )
            else:
                connection.executemany(
                    f"INSERT OR REPLACE INTO {table}(entity_id, business_name, business_address, country) VALUES (?, ?, ?, ?)",
                    batch,
                )
            count += len(batch)
            batch.clear()
    if batch:
        if source:
            connection.executemany(
                f"INSERT OR REPLACE INTO {table}(entity_id, business_name, business_address, country, source) VALUES (?, ?, ?, ?, ?)",
                batch,
            )
        else:
            connection.executemany(
                f"INSERT OR REPLACE INTO {table}(entity_id, business_name, business_address, country) VALUES (?, ?, ?, ?)",
                batch,
            )
        count += len(batch)
    return count


def build_cache(source: Path, cache_path: Path) -> None:
    required = [source, *SOURCE_DATASETS.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required results file(s) were not found: {', '.join(missing)}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    try:
        if temporary_path.exists():
            temporary_path.unlink()
        connection = sqlite3.connect(temporary_path)
        malformed_rows = 0
        try:
            connection.executescript(
                """
                PRAGMA journal_mode = DELETE;
                PRAGMA synchronous = FULL;
                CREATE TABLE source1 (
                    entity_id TEXT PRIMARY KEY,
                    business_name TEXT NOT NULL,
                    business_address TEXT NOT NULL,
                    country TEXT NOT NULL
                );
                CREATE TABLE businesses (
                    entity_id TEXT PRIMARY KEY,
                    business_name TEXT NOT NULL,
                    business_address TEXT NOT NULL,
                    country TEXT NOT NULL,
                    source TEXT NOT NULL
                );
                CREATE TABLE results (
                    source1_entity_id TEXT PRIMARY KEY,
                    matched_entity_ids TEXT NOT NULL,
                    match_count INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('matched', 'unmatched'))
                );
                """
            )
            connection.execute("BEGIN")
            source1_count = insert_batches(connection, "source1", SOURCE_DATASETS["source1"])
            source2_count = insert_batches(
                connection, "businesses", SOURCE_DATASETS["source2"], "Source 2"
            )
            source3_count = insert_batches(
                connection, "businesses", SOURCE_DATASETS["source3"], "Source 3"
            )
            with source.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle, delimiter="\t")
                header = next(reader, None)
                if header != ["source1_entity_id", "matched_entity_ids"]:
                    raise ValueError(
                        "Results file must have columns source1_entity_id and matched_entity_ids"
                    )
                results_batch = []
                for row in reader:
                    if len(row) != 2 or not row[0].strip():
                        malformed_rows += 1
                        continue
                    matches = split_matches(row[1])
                    results_batch.append((
                        row[0].strip(),
                        ",".join(matches),
                        len(matches),
                        "matched" if matches else "unmatched",
                    ))
                    if len(results_batch) == BATCH_SIZE:
                        connection.executemany(
                            """
                            INSERT OR REPLACE INTO results
                            (source1_entity_id, matched_entity_ids, match_count, status)
                            VALUES (?, ?, ?, ?)
                            """,
                            results_batch,
                        )
                        results_batch.clear()
                if results_batch:
                    connection.executemany(
                        """
                        INSERT OR REPLACE INTO results
                        (source1_entity_id, matched_entity_ids, match_count, status)
                        VALUES (?, ?, ?, ?)
                        """,
                        results_batch,
                    )
            connection.executescript(
                """
                CREATE INDEX source1_name_idx ON source1(business_name COLLATE NOCASE);
                CREATE INDEX results_status_idx ON results(status);
                CREATE INDEX results_source_id_idx ON results(source1_entity_id);
                CREATE INDEX businesses_source_idx ON businesses(source);
                """
            )
            connection.execute(
                "CREATE INDEX businesses_name_idx ON businesses(business_name COLLATE NOCASE)"
            )
            connection.execute(
                "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            metadata = [
                ("schema_version", SCHEMA_VERSION),
                ("source1_count", str(source1_count)),
                ("source2_count", str(source2_count)),
                ("source3_count", str(source3_count)),
                ("malformed_rows", str(malformed_rows)),
            ]
            for key, path in [("results", source), *SOURCE_DATASETS.items()]:
                stat = path.stat()
                metadata.extend([
                    (f"{key}_mtime_ns", str(stat.st_mtime_ns)),
                    (f"{key}_size", str(stat.st_size)),
                ])
            connection.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", metadata)
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary_path, cache_path)
        if malformed_rows:
            LOGGER.warning("Skipped %d malformed result rows", malformed_rows)
        LOGGER.info("Cached %d Source-1, %d Source-2, and %d Source-3 records",
                    source1_count, source2_count, source3_count)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def cache_is_current(cache_path: Path) -> bool:
    if not cache_path.is_file():
        return False
    try:
        connection = sqlite3.connect(cache_path)
        try:
            version = connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            if version is None or version[0] != SCHEMA_VERSION:
                return False
            for key, path in [("results", SOURCE_PATH), *SOURCE_DATASETS.items()]:
                if not path.is_file():
                    return False
                row = connection.execute(
                    "SELECT value FROM metadata WHERE key = ?",
                    (f"{key}_mtime_ns",),
                ).fetchone()
                if row is None or int(row[0]) != path.stat().st_mtime_ns:
                    return False
            return True
        finally:
            connection.close()
    except (OSError, sqlite3.Error, ValueError):
        return False


def ensure_cache() -> None:
    if not cache_is_current(CACHE_PATH):
        LOGGER.info("Building results cache from output and test datasets")
        build_cache(SOURCE_PATH, CACHE_PATH)


def open_cache() -> sqlite3.Connection:
    connection = sqlite3.connect(CACHE_PATH)
    connection.execute("PRAGMA case_sensitive_like = ON")
    connection.row_factory = sqlite3.Row
    return connection


def business_payload(row: sqlite3.Row) -> dict[str, str]:
    return {
        "entity_id": row["entity_id"],
        "business_name": row["business_name"] or "Name unavailable",
        "business_address": row["business_address"] or "Address unavailable",
        "country": row["country"] or "Country unavailable",
        "source": row["source"],
    }


def result_payload(connection: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    matched_ids = split_matches(row["matched_entity_ids"])
    businesses = []
    if matched_ids:
        placeholders = ",".join("?" for _ in matched_ids)
        matches = connection.execute(
            f"""
            SELECT entity_id, business_name, business_address, country, source FROM businesses
            WHERE entity_id IN ({placeholders})
            """,
            matched_ids,
        ).fetchall()
        by_id = {match["entity_id"]: business_payload(match) for match in matches}
        businesses = [
            by_id.get(
                entity_id,
            {
                "entity_id": entity_id,
                "business_name": "Name unavailable",
                "business_address": "Address unavailable",
                "country": "Country unavailable",
                "source": "Unknown",
            },
        )
            for entity_id in matched_ids
        ]
    return {
        "source1_entity_id": row["source1_entity_id"],
        "source1_business_name": row["source1_business_name"] or "Name unavailable",
        "source1_business_address": row["source1_business_address"] or "Address unavailable",
        "source1_country": row["source1_country"] or "Country unavailable",
        "status": row["status"],
        "match_count": row["match_count"],
        "matched_businesses": businesses,
    }


def parse_page_params(query: dict[str, list[str]]) -> tuple[int, int, str, str]:
    def one(name: str, default: str) -> str:
        values = query.get(name, [default])
        if len(values) != 1:
            raise ValueError(f"{name} must be specified once")
        return values[0]

    try:
        page = int(one("page", "1"))
        page_size = int(one("page_size", str(DEFAULT_PAGE_SIZE)))
    except ValueError as error:
        raise ValueError("page and page_size must be integers") from error
    status, search = one("status", "all"), one("q", "").strip()
    if page < 1:
        raise ValueError("page must be at least 1")
    if not 1 <= page_size <= MAX_PAGE_SIZE:
        raise ValueError(f"page_size must be between 1 and {MAX_PAGE_SIZE}")
    if status not in {"all", "matched", "unmatched"}:
        raise ValueError("status must be all, matched, or unmatched")
    if len(search) > 200:
        raise ValueError("q must be 200 characters or fewer")
    return page, page_size, status, search


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "BusinessEntityResolutionUI/1.0"

    def do_GET(self) -> None:
        try:
            parsed = urlsplit(self.path)
            if parsed.path == "/":
                self.send_static("index.html", "text/html; charset=utf-8")
            elif parsed.path.startswith("/static/"):
                self.send_static(unquote(parsed.path[len("/static/"):]), None)
            elif parsed.path == "/api/summary":
                self.send_summary()
            elif parsed.path == "/api/results":
                self.send_results(parse_qs(parsed.query, keep_blank_values=True))
            elif parsed.path == "/api/search":
                self.send_search(parse_qs(parsed.query, keep_blank_values=True))
            elif parsed.path.startswith("/api/results/") and parsed.path.endswith("/matches"):
                entity_id = unquote(parsed.path[len("/api/results/"):-len("/matches")])
                self.send_matches(entity_id, parse_qs(parsed.query, keep_blank_values=True))
            elif parsed.path.startswith("/api/results/"):
                entity_id = unquote(parsed.path[len("/api/results/"):])
                if not entity_id or "/" in entity_id:
                    self.send_error_json(HTTPStatus.BAD_REQUEST, "Invalid entity ID")
                else:
                    self.send_detail(entity_id)
            else:
                self.send_error_json(HTTPStatus.NOT_FOUND, "Not found")
        except FileNotFoundError as error:
            LOGGER.error("%s", error)
            self.send_error_json(HTTPStatus.NOT_FOUND, "Required data file is unavailable")
        except (sqlite3.Error, OSError) as error:
            LOGGER.exception("Request failed")
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
        except ValueError as error:
            self.send_error_json(HTTPStatus.BAD_REQUEST, str(error))

    def send_static(self, relative_name: str, content_type: str | None) -> None:
        requested = (STATIC_PATH / relative_name).resolve()
        if STATIC_PATH not in requested.parents or not requested.is_file():
            self.send_error_json(HTTPStatus.NOT_FOUND, "Not found")
            return
        types = {".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8"}
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or types.get(requested.suffix, "application/octet-stream"))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with requested.open("rb") as handle:
            self.wfile.write(handle.read())

    def send_summary(self) -> None:
        ensure_cache()
        with open_cache() as connection:
            totals = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       COALESCE(SUM(status = 'matched'), 0) AS matched,
                       COALESCE(SUM(status = 'unmatched'), 0) AS unmatched
                FROM results
                """
            ).fetchone()
        total, matched = totals["total"], totals["matched"]
        self.send_json({
            "total_records": total,
            "matched_records": matched,
            "unmatched_records": totals["unmatched"],
            "match_rate": matched / total if total else 0,
        })

    def send_results(self, query: dict[str, list[str]]) -> None:
        page, page_size, status, search = parse_page_params(query)
        ensure_cache()
        clauses: list[str] = []
        parameters: list[Any] = []
        if status != "all":
            clauses.append("r.status = ?")
            parameters.append(status)
        if search:
            clauses.append("(r.source1_entity_id LIKE ? OR s.business_name LIKE ?)")
            parameters.extend([f"{search}%", f"%{search}%"])
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with open_cache() as connection:
            total = connection.execute(
                f"""
                SELECT COUNT(*) FROM results r JOIN source1 s
                ON s.entity_id = r.source1_entity_id{where}
                """,
                parameters,
            ).fetchone()[0]
            rows = connection.execute(
                f"""
                SELECT r.source1_entity_id, s.business_name AS source1_business_name,
                       s.business_address AS source1_business_address, s.country AS source1_country,
                       r.matched_entity_ids, r.match_count, r.status
                FROM results r JOIN source1 s ON s.entity_id = r.source1_entity_id
                {where}
                ORDER BY r.source1_entity_id
                LIMIT ? OFFSET ?
                """,
                parameters + [page_size, (page - 1) * page_size],
            ).fetchall()
            payload = [result_payload(connection, row) for row in rows]
        self.send_json({
            "page": page,
            "page_size": page_size,
            "total_records": total,
            "total_pages": (total + page_size - 1) // page_size,
            "results": payload,
        })

    def send_detail(self, entity_id: str) -> None:
        ensure_cache()
        with open_cache() as connection:
            row = connection.execute(
                """
                SELECT r.source1_entity_id, s.business_name AS source1_business_name,
                       s.business_address AS source1_business_address, s.country AS source1_country,
                       r.matched_entity_ids, r.match_count, r.status
                FROM results r JOIN source1 s ON s.entity_id = r.source1_entity_id
                WHERE r.source1_entity_id = ?
                """,
                (entity_id,),
            ).fetchone()
            if row is None:
                self.send_error_json(HTTPStatus.NOT_FOUND, "Source-1 entity was not found")
                return
            payload = result_payload(connection, row)
        self.send_json(payload)

    def send_search(self, query: dict[str, list[str]]) -> None:
        values = query.get("q", [""])
        if len(values) != 1 or not values[0].strip():
            raise ValueError("q is required")
        search = values[0].strip()
        if len(search) > 200:
            raise ValueError("q must be 200 characters or fewer")
        try:
            page_size = int(query.get("page_size", ["8"])[0])
        except ValueError as error:
            raise ValueError("page_size must be an integer") from error
        if not 1 <= page_size <= 20:
            raise ValueError("page_size must be between 1 and 20")
        ensure_cache()
        with open_cache() as connection:
            rows = connection.execute(
                """
                SELECT entity_id, business_name
                FROM source1
                WHERE entity_id LIKE ? OR business_name LIKE ?
                ORDER BY business_name COLLATE NOCASE, entity_id
                LIMIT ?
                """,
                (f"{search}%", f"%{search}%", page_size),
            ).fetchall()
        self.send_json({
            "results": [
                {"entity_id": row["entity_id"], "business_name": row["business_name"]}
                for row in rows
            ],
        })

    def send_matches(self, entity_id: str, query: dict[str, list[str]]) -> None:
        if not entity_id or "/" in entity_id:
            self.send_error_json(HTTPStatus.BAD_REQUEST, "Invalid entity ID")
            return
        values = query.get("page", ["1"])
        size_values = query.get("page_size", [str(DEFAULT_PAGE_SIZE)])
        try:
            page, page_size = int(values[0]), int(size_values[0])
        except (IndexError, ValueError) as error:
            raise ValueError("page and page_size must be integers") from error
        if page < 1 or not 1 <= page_size <= MAX_PAGE_SIZE:
            raise ValueError("Invalid pagination parameters")
        ensure_cache()
        with open_cache() as connection:
            row = connection.execute(
                "SELECT matched_entity_ids FROM results WHERE source1_entity_id = ?",
                (entity_id,),
            ).fetchone()
            if row is None:
                self.send_error_json(HTTPStatus.NOT_FOUND, "Source-1 entity was not found")
                return
            matched_ids = split_matches(row["matched_entity_ids"])
            total = len(matched_ids)
            page_ids = matched_ids[(page - 1) * page_size:page * page_size]
            if page_ids:
                placeholders = ",".join("?" for _ in page_ids)
                rows = connection.execute(
                    f"""
                    SELECT entity_id, business_name, business_address, country, source
                    FROM businesses WHERE entity_id IN ({placeholders})
                    """,
                    page_ids,
                ).fetchall()
                by_id = {item["entity_id"]: business_payload(item) for item in rows}
                payload = [
                    by_id.get(entity, {
                        "entity_id": entity,
                        "business_name": "Name unavailable",
                        "business_address": "Address unavailable",
                        "country": "Country unavailable",
                        "source": "Unknown",
                    })
                    for entity in page_ids
                ]
            else:
                payload = []
        self.send_json({
            "page": page,
            "page_size": page_size,
            "total_records": total,
            "total_pages": (total + page_size - 1) // page_size,
            "results": payload,
        })

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status: HTTPStatus, message: str) -> None:
        self.send_json({"error": message}, status)

    def log_message(self, format_string: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), format_string % args)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ensure_cache()
    server = ThreadingHTTPServer(("127.0.0.1", 8000), DashboardHandler)
    LOGGER.info("Business Entity Resolution Results: http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
