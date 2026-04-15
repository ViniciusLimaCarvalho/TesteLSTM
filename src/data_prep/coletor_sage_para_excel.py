# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


def request_json(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> Any:
    r = requests.get(url, headers=headers, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def fetch_sage_por_cd_id(
    base_url: str,
    endpoint: str,
    headers: Dict[str, str],
    cd_id: str,
) -> List[Dict[str, Any]]:
    url = base_url.rstrip("/") + "/" + endpoint.lstrip("/")
    params = {"cd_id": cd_id}
    payload = request_json(url, headers=headers, params=params)

    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        if "results" in payload and isinstance(payload["results"], list):
            return [x for x in payload["results"] if isinstance(x, dict)]
        if "data" in payload and isinstance(payload["data"], list):
            return [x for x in payload["data"] if isinstance(x, dict)]
        return []

    return []


def parse_tags_file(path: str) -> List[str]:
    tags: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            tags.append(t.lower())

    # remove duplicados mantendo ordem
    seen = set()
    out: List[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def load_referencias(path: str) -> pd.DataFrame:
    ref = pd.read_excel(path, engine="openpyxl")
    # tenta achar colunas prováveis
    cols_l = {str(c).strip().lower(): c for c in ref.columns}

    tag_col = None
    for cand in ("tag", "cd_id", "ponto", "id"):
        if cand in cols_l:
            tag_col = cols_l[cand]
            break
    if tag_col is None:
        tag_col = ref.columns[0]

    desc_col = None
    for cand in ("descricao", "descrição", "ds_descr", "descr", "nome"):
        if cand in cols_l:
            desc_col = cols_l[cand]
            break
    if desc_col is None and len(ref.columns) >= 3:
        desc_col = ref.columns[2]
    if desc_col is None and len(ref.columns) >= 2:
        desc_col = ref.columns[1]
    if desc_col is None:
        desc_col = ref.columns[0]

    out = pd.DataFrame()
    out["tag"] = ref[tag_col].astype(str).str.strip().str.lower()
    out["descricao"] = ref[desc_col].astype(str).str.strip()

    out = out[out["tag"].ne("")].drop_duplicates(subset=["tag"], keep="first").reset_index(drop=True)
    return out


def normalize_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df

    rename_map = {
        "cd_id": "tag",
        "dt_bh_dthr": "timestamp",
        "vl_valor": "valor",
        "ds_descr": "descricao_api",
        "ds_unidade": "unidade",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    missing = [c for c in ("tag", "timestamp", "valor") if c not in df.columns]
    if missing:
        raise ValueError(
            f"Campos obrigatórios ausentes: {missing}. Colunas disponíveis: {list(df.columns)}"
        )

    df["tag"] = df["tag"].astype(str).str.strip().str.lower()

    # Converte para datetime em UTC e depois para horário Brasil, removendo tz para Excel
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["timestamp"] = ts.dt.tz_convert("America/Sao_Paulo").dt.tz_localize(None)

    df = df.dropna(subset=["timestamp", "tag"]).copy()
    df = df.sort_values(["tag", "timestamp"]).reset_index(drop=True)

    return df


def drop_excel_timezones(d: pd.DataFrame) -> pd.DataFrame:
    # remove timezone de qualquer coluna tz-aware (se existir)
    d = d.copy()
    for col in d.columns:
        if pd.api.types.is_datetime64tz_dtype(d[col]):
            d[col] = d[col].dt.tz_convert("America/Sao_Paulo").dt.tz_localize(None)
    return d


def excel_safe_sheet_name(name: str) -> str:
    name = re.sub(r"[\[\]\:\*\?\/\\]", " ", str(name)).strip()
    if not name:
        name = "Sheet"
    return name[:31]


def write_excel(df: pd.DataFrame, out_path: str, ref: Optional[pd.DataFrame]) -> None:
    if df is None:
        raise ValueError("df veio como None. Verifique normalize_records() (deve retornar df).")

    df = drop_excel_timezones(df)

    if ref is not None and not ref.empty:
        ref = ref.copy()
        ref["tag"] = ref["tag"].astype(str).str.strip().str.lower()
        df = df.merge(ref, on="tag", how="left")
    else:
        df["descricao"] = ""

    # Garante que 'descricao' exista
    if "descricao" not in df.columns:
        df["descricao"] = ""

    resumo = (
        df.groupby(["tag", "descricao"], dropna=False)
        .agg(
            registros=("valor", "count"),
            inicio=("timestamp", "min"),
            fim=("timestamp", "max"),
        )
        .reset_index()
        .sort_values(["tag"])
    )

    resumo = drop_excel_timezones(resumo)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        resumo.to_excel(writer, index=False, sheet_name="Resumo")
        df.to_excel(writer, index=False, sheet_name="DadosBrutos")

        for tag, g in df.groupby("tag"):
            g = g.sort_values("timestamp").reset_index(drop=True)

            desc = ""
            if "descricao" in g.columns and g["descricao"].notna().any():
                desc = str(g["descricao"].dropna().iloc[0]).strip()

            sheet = excel_safe_sheet_name(f"{tag} - {desc}" if desc else tag)

            base = sheet
            n = 2
            while sheet in writer.book.sheetnames:
                suffix = f"_{n}"
                sheet = excel_safe_sheet_name(base[: (31 - len(suffix))] + suffix)
                n += 1

            g.to_excel(writer, index=False, sheet_name=sheet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--endpoint", default="/dados/sage/buscar")
    parser.add_argument("--token", default="")
    parser.add_argument("--tags-file", required=True)
    parser.add_argument("--referencias", default="")
    parser.add_argument("--saida", required=True)
    args = parser.parse_args()

    headers: Dict[str, str] = {}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"

    try:
        tags = parse_tags_file(args.tags_file)
    except FileNotFoundError:
        print(f"ERRO: arquivo de tags não encontrado: {args.tags_file}", file=sys.stderr)
        sys.exit(2)

    ref_df: Optional[pd.DataFrame] = None
    if args.referencias:
        try:
            ref_df = load_referencias(args.referencias)
        except Exception as e:
            print(f"AVISO: não foi possível ler referencias: {e}. Continuando sem referências.")
            ref_df = None

    all_records: List[Dict[str, Any]] = []
    for i, t in enumerate(tags, start=1):
        t_norm = t.strip().lower()
        try:
            recs = fetch_sage_por_cd_id(args.base_url, args.endpoint, headers, t_norm)
        except requests.RequestException as e:
            print(f"ERRO HTTP na TAG {t_norm}: {e}", file=sys.stderr)
            recs = []

        for r in recs:
            if "cd_id" not in r:
                r["cd_id"] = t_norm
        all_records.extend(recs)

        print(f"[{i}/{len(tags)}] {t_norm} -> {len(recs)} registros")

    df = normalize_records(all_records)
    write_excel(df, args.saida, ref_df)

    print(f"OK: {args.saida} | registros: {len(df)} | tags: {len(tags)}")


if __name__ == "__main__":
    main()
