from __future__ import annotations

import csv
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "dataset" / "test"))
RESULTS_PATH = Path(os.getenv("MATCHING_RESULTS_PATH", ROOT / "output" / "matching_results.tsv"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "src" / "artifacts"))
DB_PATH = Path(
    os.getenv("ENTITY_DB_PATH", str(Path(__file__).resolve().parent / "entity_resolution_lookup.sqlite3"))
)
THRESHOLD_DEFAULT = 0.84

st.set_page_config(page_title="Business Entity Resolution", page_icon="🔎", layout="wide")

# Add the project root so the existing src package can be imported.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _safe_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _source_paths() -> Dict[str, Path]:
    return {
        "S1": DATA_DIR / "test_source1.tsv",
        "S2": DATA_DIR / "test_source2.tsv",
        "S3": DATA_DIR / "test_source3.tsv",
    }

def _connect_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH), timeout=60)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            source TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            business_name TEXT,
            business_address TEXT,
            country TEXT,
            PRIMARY KEY (source, entity_id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS matches (
            source1_entity_id TEXT PRIMARY KEY,
            matched_entity_ids TEXT
        )
    """)

    return conn


def _db_ready(conn: sqlite3.Connection) -> bool:
    row = conn.execute("SELECT value FROM metadata WHERE key='ready'").fetchone() if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='metadata'"
    ).fetchone() else None
    return bool(row and row[0] == "1")


def _load_tsv_into_db(conn: sqlite3.Connection, source: str, path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file not found: {path}")
    batch = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"entity_id", "business_name", "business_address", "country"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{path.name} must contain columns: {', '.join(sorted(required))}")
        for row in reader:
            batch.append((
                source,
                _safe_text(row.get("entity_id")),
                _safe_text(row.get("business_name")),
                _safe_text(row.get("business_address")),
                _safe_text(row.get("country")),
            ))
            if len(batch) >= 10000:
                conn.executemany(
                    "INSERT OR REPLACE INTO entities VALUES (?, ?, ?, ?, ?)", batch
                )
                conn.commit()
                batch.clear()
        if batch:
            conn.executemany("INSERT OR REPLACE INTO entities VALUES (?, ?, ?, ?, ?)", batch)
            conn.commit()


def _ensure_lookup_db() -> None:
    paths = _source_paths()
    missing = [str(p) for p in paths.values() if not p.exists()]
    if not RESULTS_PATH.exists():
        missing.append(str(RESULTS_PATH))
    if missing:
        st.warning("Some dataset/result files are not present in this deployment.")
        st.code("\n".join(missing))
        st.info("The app can still show the model report, but lookup and prediction need these files.")
        return

    conn = _connect_db()
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
        ready = conn.execute("SELECT value FROM metadata WHERE key='ready'").fetchone()
        if ready and ready[0] == "1":
            return
        with st.status("Preparing searchable entity index (first startup can take several minutes)...", expanded=True) as status:
            for source, path in paths.items():
                st.write(f"Indexing {source}: {path.name}")
                _load_tsv_into_db(conn, source, path)
            st.write("Loading precomputed matching results...")
            conn.execute("DELETE FROM matches")
            batch = []
            with RESULTS_PATH.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    batch.append((
                        _safe_text(row.get("source1_entity_id")),
                        _safe_text(row.get("matched_entity_ids")),
                    ))
                    if len(batch) >= 10000:
                        conn.executemany("INSERT OR REPLACE INTO matches VALUES (?, ?)", batch)
                        conn.commit()
                        batch.clear()
                if batch:
                    conn.executemany("INSERT OR REPLACE INTO matches VALUES (?, ?)", batch)
                    conn.commit()
            conn.execute("INSERT OR REPLACE INTO metadata VALUES ('ready', '1')")
            conn.commit()
            status.update(label="Search index ready", state="complete", expanded=False)
    finally:
        conn.close()


def _entity(conn, source: str, entity_id: str):
    return conn.execute(
        "SELECT source, entity_id, business_name, business_address, country "
        "FROM entities WHERE source=? AND entity_id=?",
        (source, entity_id),
    ).fetchone()


def _entity_dict(row):
    if not row:
        return None
    return {
        "Source": row[0],
        "Entity ID": row[1],
        "Business name": row[2] or "",
        "Address": row[3] or "",
        "Country": row[4] or "",
    }


@st.cache_resource(show_spinner="Loading trained model...")
def load_model_artifact():
    path = MODEL_DIR / "model.joblib"
    if not path.exists():
        return None
    artifact = joblib.load(path)
    return artifact


@st.cache_data(show_spinner=False)
def load_training_report():
    path = MODEL_DIR / "training_report.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Preparing candidate index from Source 2 and Source 3...")
def load_prediction_index():
    # Use the project's existing inference components, not a separate approximate model.
    from src.blocking import BlockingIndex, MAX_CANDIDATES_PER_ENTITY
    from src.train import stream_tsv_rows

    paths = _source_paths()
    index = BlockingIndex(max_candidates=MAX_CANDIDATES_PER_ENTITY)
    lookup = {}
    index.add_records(stream_tsv_rows(str(paths["S2"])), record_lookup=lookup)
    index.add_records(stream_tsv_rows(str(paths["S3"])), record_lookup=lookup)
    index.finalize()
    return index, lookup


def _format_candidate_rows(candidate_ids: List[str], lookup: dict, scores: List[float]) -> pd.DataFrame:
    rows = []
    for candidate_id, score in zip(candidate_ids, scores):
        record = lookup.get(candidate_id)
        if record is None:
            continue
        # stream_tsv_rows / BlockingIndex record lookup stores (name, address, country)
        name, address, country = record
        rows.append({
            "Entity ID": candidate_id,
            "Business name": name,
            "Address": address,
            "Country": country,
            "Match confidence": float(score),
        })
    return pd.DataFrame(rows)


def _predict_one(name: str, address: str, country: str, threshold: float):
    from src.features import build_feature_matrix

    artifact = load_model_artifact()
    if artifact is None:
        raise FileNotFoundError(f"Model artifact not found: {MODEL_DIR / 'model.joblib'}")
    model = artifact["model"]
    index, lookup = load_prediction_index()

    candidates = index.candidates_for(name, address, country)
    # Deduplicate while preserving the blocking rank order.
    seen = set()
    candidates = [cid for cid in candidates if not (cid in seen or seen.add(cid))]
    if not candidates:
        return pd.DataFrame(), 0

    # The project's feature builder expects Source-1 ID -> (name, address, country).
    s1_lookup = {"USER_QUERY": (name, address, country)}
    pairs = [("USER_QUERY", cid) for cid in candidates]
    X, kept_pairs = build_feature_matrix(pairs, lookup, s1_lookup=s1_lookup)
    if len(X) == 0:
        return pd.DataFrame(), len(candidates)

    probs = model.predict_proba(X)[:, 1]
    rows = []
    for (s1_id, candidate_id), probability in zip(kept_pairs, probs):
        if probability >= threshold and candidate_id in lookup:
            rec_name, rec_address, rec_country = lookup[candidate_id]
            rows.append({
                "Entity ID": candidate_id,
                "Business name": rec_name,
                "Address": rec_address,
                "Country": rec_country,
                "Match confidence": float(probability),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Match confidence", ascending=False).reset_index(drop=True)
        df["Match confidence"] = df["Match confidence"].map(lambda x: f"{x:.1%}")
    return df, len(candidates)


def existing_search(query: str, limit: int = 30):
    conn = _connect_db()
    try:
        # Exact ID match first, then case-insensitive name/address search.
        rows = conn.execute(
            """SELECT source, entity_id, business_name, business_address, country
               FROM entities
               WHERE entity_id = ?
                  OR business_name LIKE ? COLLATE NOCASE
                  OR business_address LIKE ? COLLATE NOCASE
               LIMIT ?""",
            (query, f"%{query}%", f"%{query}%", limit),
        ).fetchall()
        out = []
        for row in rows:
            entity = _entity_dict(row)
            if row[0] == "S1":
                match_row = conn.execute(
                    "SELECT matched_entity_ids FROM matches WHERE source1_entity_id=?",
                    (row[1],),
                ).fetchone()
                ids = [x for x in (match_row[0].split(",") if match_row and match_row[0] else []) if x]
                matched = []
                for cid in ids[:100]:
                    # The IDs in matching_results can refer to either Source 2 or Source 3.
                    rec = conn.execute(
                        "SELECT source, entity_id, business_name, business_address, country "
                        "FROM entities WHERE entity_id=? AND source IN ('S2','S3') LIMIT 1",
                        (cid,),
                    ).fetchone()
                    if rec:
                        matched.append(_entity_dict(rec))
                out.append({"source1": entity, "matches": matched})
            else:
                # Search S2/S3 entities too; find S1 rows whose precomputed match list contains this ID.
                out.append({"source1": entity, "matches": [], "note": "This is a Source 2/3 record. Search its Source 1 counterpart by ID or name."})
        return out
    finally:
        conn.close()


def main():
    st.title("🔎 Business Entity Resolution")
    st.caption("LightGBM-powered matching across three business data sources")

    report = load_training_report()
    artifact = load_model_artifact()
    threshold = float(artifact.get("threshold", report.get("best_threshold", THRESHOLD_DEFAULT))) if artifact else float(report.get("best_threshold", THRESHOLD_DEFAULT))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", str(artifact.get("backend", "LightGBM") if artifact else report.get("backend", "LightGBM")))
    c2.metric("Decision threshold", f"{threshold:.2f}")
    c3.metric("Validation Macro F0.5", f"{float(report.get('val_macro_f05', 0)):.3f}" if report else "N/A")
    c4.metric("Blocking recall ceiling", f"{float(report.get('blocking_recall_ceiling', 0)):.1%}" if report else "N/A")

    tab_existing, tab_predict, tab_about = st.tabs(["🔍 Existing results", "🧠 Predict a new business", "📊 Model details"])

    with tab_existing:
        st.subheader("Search existing predictions")
        st.write("Search by Source 1 business name, address, or entity ID to view its precomputed matches.")
        query = st.text_input("Business name, address, or entity ID", key="existing_query")
        if st.button("Search existing results", type="primary", key="existing_btn"):
            if not query.strip():
                st.warning("Enter a search term first.")
            else:
                try:
                    _ensure_lookup_db()
                    if DB_PATH.exists():
                        results = existing_search(query.strip())
                        if not results:
                            st.info("No matching records found.")
                        for result in results:
                            entity = result["source1"]
                            st.markdown(f"**{entity['Business name'] or '(unnamed business)'}** · {entity['Source']} · ID `{entity['Entity ID']}`")
                            st.caption(f"{entity['Address']} · {entity['Country']}")
                            if result.get("matches"):
                                st.write("Predicted matches")
                                match_df = pd.DataFrame(result["matches"])
                                st.dataframe(match_df, use_container_width=True, hide_index=True)
                            elif result.get("note"):
                                st.info(result["note"])
                            else:
                                st.info("No precomputed match was found for this Source 1 entity.")
                except Exception as exc:
                    st.error(f"Search failed: {exc}")
                    st.exception(exc)

    with tab_predict:
        st.subheader("Predict matches for a new business")
        st.caption("The input business is compared against Source 2 and Source 3 using your existing blocking and LightGBM feature pipeline.")
        with st.form("new_business_form"):
            name = st.text_input("Business name", placeholder="e.g. Acme Corporation")
            address = st.text_input("Business address", placeholder="Street, city, region")
            country = st.text_input("Country", placeholder="e.g. US, GB, IN")
            user_threshold = st.slider("Minimum match confidence", min_value=0.50, max_value=0.99, value=max(0.50, min(0.99, threshold)), step=0.01)
            submitted = st.form_submit_button("Find matches", type="primary")
        if submitted:
            if not name.strip():
                st.warning("Business name is required.")
            else:
                try:
                    with st.spinner("Generating candidates and scoring them with LightGBM..."):
                        matches, candidate_count = _predict_one(name.strip(), address.strip(), country.strip(), user_threshold)
                    st.metric("Candidates examined", candidate_count)
                    if matches.empty:
                        st.info("No candidates passed the confidence threshold. Try a lower threshold or check the business details.")
                    else:
                        st.success(f"Found {len(matches)} candidate match(es).")
                        st.dataframe(matches, use_container_width=True, hide_index=True)
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    st.exception(exc)

    with tab_about:
        st.subheader("Training and evaluation")
        if report:
            st.json(report)
        else:
            st.warning("training_report.json not found.")
        st.markdown("""
        **How to interpret this app**
        - A match confidence is the model's predicted probability for a candidate pair; it is not a guarantee of identity.
        - Candidate generation (blocking) limits which records are scored. A true match omitted by blocking cannot be recovered by the classifier.
        - Validation metrics are measured on the project's validation split and may differ from live query performance.
        """)

    st.divider()
    st.caption("Business Entity Resolution · Existing predictions + interactive ML inference")


if __name__ == "__main__":
    main()
