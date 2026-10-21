"""Shared IO helpers for reading challenge TSVs and writing submission TSVs."""
import pandas as pd


def read_source(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return df


def read_ground_truth(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return df


def build_record_lookup(*dfs, id_col="entity_id", name_col="business_name",
                         addr_col="business_address", country_col="country"):
    # PERF: itertuples(name=None) + positional unpacking avoids namedtuple
    # attribute-lookup overhead, which matters at multi-million-row scale.
    lookup = {}
    cols = [id_col, name_col, addr_col, country_col]
    for df in dfs:
        for rid, name, addr, country in df[cols].itertuples(index=False, name=None):
            lookup[rid] = (name, addr, country)
    return lookup


def write_id_list_tsv(path, s1_to_ids, id_col_name, list_col_name):
    """Writes a two-column TSV: source1_entity_id <TAB> comma,joined,ids (no quoting)."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"{id_col_name}\t{list_col_name}\n")
        for s1_id, ids in s1_to_ids.items():
            # de-duplicate while preserving order
            seen = set()
            uniq = []
            for i in ids:
                if i not in seen:
                    seen.add(i)
                    uniq.append(i)
            f.write(f"{s1_id}\t{','.join(uniq)}\n")


def parse_id_list(cell):
    if cell is None or isinstance(cell, float):
        return []
    cell = str(cell).strip()
    if not cell:
        return []
    return [x.strip() for x in cell.split(",") if x.strip()]
