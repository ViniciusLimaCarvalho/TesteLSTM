# -*- coding: utf-8 -*-
"""
Cria arquivo HDF5 combinando matrizes térmicas (CSV) com dados de sensores SAGE.

Fonte de dados SAGE (escolha uma):
  - API SAGE  : forneça --base-url e --tags-file
  - Excel     : forneça --excel (padrão, comportamento original)

Correspondência matrix ↔ sensor:
  - Por timestamp : quando todos os CSVs têm timestamp válido no nome
                    (usa pd.merge_asof com tolerância configurável)
  - Por posição   : fallback legado quando os comprimentos coincidem
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import pandas as pd

# Importa funções do coletor (mesmo diretório)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coletor_sage_para_excel import (  # noqa: E402
    fetch_sage_por_cd_id,
    normalize_records,
    parse_tags_file,
)

try:
    import requests
except ImportError:
    requests = None  # type: ignore

# --- DEFAULTS ---
DEFAULT_EXCEL = "data/raw/AD-TF2Y.xlsx"
DEFAULT_MATRIZES = "data/raw/V3.A1_CSV"
DEFAULT_SAIDA = "data/processed/dataset_axia_completo_2d.h5"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cria HDF5 com matrizes térmicas + dados SAGE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Fonte SAGE via API (opcional; padrão: Excel)")
    g.add_argument("--base-url", default="", metavar="URL",
                   help="URL base da API SAGE")
    g.add_argument("--endpoint", default="/dados/sage/buscar",
                   help="Endpoint da API")
    g.add_argument("--token", default="",
                   help="Token Bearer para autenticação")
    g.add_argument("--tags-file", default="", metavar="ARQUIVO",
                   help="Arquivo de texto com uma tag por linha")

    p.add_argument("--excel", default=DEFAULT_EXCEL,
                   help="Excel de fallback (usado quando --base-url não for fornecido)")
    p.add_argument("--matrizes", default=DEFAULT_MATRIZES,
                   help="Pasta com os CSVs das matrizes térmicas")
    p.add_argument("--saida", default=DEFAULT_SAIDA,
                   help="Caminho do arquivo HDF5 de saída")
    p.add_argument("--tolerancia-ts", type=int, default=300, metavar="SEGUNDOS",
                   help="Tolerância máxima (s) para correspondência por timestamp")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Carregamento de dados SAGE
# ---------------------------------------------------------------------------

def load_sage_excel(excel_path: str) -> pd.DataFrame:
    """Lê Excel com dados SAGE e retorna tabela pivô (timestamp × tag)."""
    abas = pd.read_excel(excel_path, sheet_name=None)
    lista: List[pd.DataFrame] = []
    for _, df in abas.items():
        if "tag" in df.columns and "valor" in df.columns:
            df_temp = df[["timestamp", "tag", "valor"]].copy()
            df_temp["timestamp"] = pd.to_datetime(df_temp["timestamp"])
            lista.append(df_temp)
    if not lista:
        raise ValueError(f"Nenhuma aba com colunas 'tag' e 'valor' em {excel_path}")
    df_total = pd.concat(lista, ignore_index=True)
    return df_total.pivot_table(
        index="timestamp", columns="tag", values="valor"
    ).sort_index()


def load_sage_api(
    base_url: str, endpoint: str, token: str, tags_file: str
) -> pd.DataFrame:
    """Coleta dados SAGE via API e retorna tabela pivô (timestamp × tag)."""
    if requests is None:
        raise ImportError("Pacote 'requests' não instalado. Execute: pip install requests")

    headers: Dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    tags = parse_tags_file(tags_file)
    if not tags:
        raise ValueError(f"Nenhuma tag encontrada em {tags_file}")

    all_records: List[Dict[str, Any]] = []
    for i, t in enumerate(tags, 1):
        t_norm = t.strip().lower()
        try:
            recs = fetch_sage_por_cd_id(base_url, endpoint, headers, t_norm)
        except requests.RequestException as e:
            print(f"ERRO HTTP na TAG {t_norm}: {e}", file=sys.stderr)
            recs = []
        for r in recs:
            if "cd_id" not in r:
                r["cd_id"] = t_norm
        all_records.extend(recs)
        print(f"  [{i}/{len(tags)}] {t_norm} → {len(recs)} registros")

    df = normalize_records(all_records)
    pivot = df.pivot_table(
        index="timestamp", columns="tag", values="valor"
    ).sort_index()
    pivot.columns.name = None
    return pivot


# ---------------------------------------------------------------------------
# Correspondência por timestamp
# ---------------------------------------------------------------------------

def extract_thermal_timestamp(nome_arquivo: str) -> Optional[pd.Timestamp]:
    """Extrai timestamp do nome do CSV de matriz térmica.

    Formato esperado: 1_3_YYYYMMDD_HHMMSSXXX_VBY_V3.A1_1.csv
    """
    partes = nome_arquivo.split("_")
    if len(partes) < 4:
        return None
    try:
        data_str = partes[2]          # YYYYMMDD
        hora_str = partes[3][:6]      # HHMMSS (descarta milissegundos)
        return pd.to_datetime(f"{data_str}{hora_str}", format="%Y%m%d%H%M%S")
    except (ValueError, IndexError):
        return None


def match_by_timestamp(
    df_sensores: pd.DataFrame,
    ts_termicas: List[Optional[pd.Timestamp]],
    tolerancia_ts: int,
) -> pd.DataFrame:
    """Usa merge_asof para associar cada matriz térmica à leitura de sensor
    mais próxima no tempo, dentro da tolerância definida.

    Retorna um DataFrame com as mesmas colunas de df_sensores, na ordem
    original das matrizes térmicas.
    """
    df_ts = pd.DataFrame({
        "ts": ts_termicas,
        "orig_idx": range(len(ts_termicas)),
    })

    df_sens_reset = df_sensores.reset_index()
    df_sens_reset.columns.name = None
    # Garante que a coluna de timestamp se chame "timestamp"
    if df_sens_reset.columns[0] != "timestamp":
        df_sens_reset = df_sens_reset.rename(columns={df_sens_reset.columns[0]: "timestamp"})

    df_matched = pd.merge_asof(
        df_ts.sort_values("ts"),
        df_sens_reset.sort_values("timestamp"),
        left_on="ts",
        right_on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerancia_ts),
    )

    # Restaura ordem original das matrizes
    df_matched = df_matched.sort_values("orig_idx").reset_index(drop=True)

    sem_match = df_matched[df_sensores.columns].isna().all(axis=1).sum()
    if sem_match:
        print(
            f"  AVISO: {sem_match}/{len(ts_termicas)} matrizes sem leitura de sensor "
            f"dentro da tolerância de {tolerancia_ts}s."
        )

    return df_matched


# ---------------------------------------------------------------------------
# Criação do HDF5
# ---------------------------------------------------------------------------

def build_hdf5(
    df_sensores: pd.DataFrame,
    arquivos_matrizes: List[str],
    saida: str,
    tolerancia_ts: int,
) -> None:
    num_amostras = len(arquivos_matrizes)
    tag_cols = list(df_sensores.columns)

    # --- Extrai timestamps das matrizes ---
    ts_termicas = [
        extract_thermal_timestamp(os.path.basename(f)) for f in arquivos_matrizes
    ]
    ts_validos = [ts for ts in ts_termicas if ts is not None]
    usar_timestamp = len(ts_validos) == num_amostras

    if usar_timestamp:
        print(f"  Estratégia: correspondência por timestamp (tolerância={tolerancia_ts}s)")
        df_matched = match_by_timestamp(df_sensores, ts_termicas, tolerancia_ts)
    else:
        # Fallback posicional
        n_sens = len(df_sensores)
        if num_amostras != n_sens:
            print(
                f"  AVISO: {num_amostras} matrizes vs {n_sens} linhas de sensor. "
                "Usando o mínimo com correspondência posicional."
            )
            num_amostras = min(num_amostras, n_sens)
            arquivos_matrizes = arquivos_matrizes[:num_amostras]
            ts_termicas = ts_termicas[:num_amostras]
        print("  Estratégia: correspondência por posição (modo legado)")
        df_matched = None

    os.makedirs(os.path.dirname(os.path.abspath(saida)), exist_ok=True)
    dataset_metadata: List[Dict[str, Any]] = []

    print(f"Criando {saida} ({num_amostras} amostras, {len(tag_cols)} tags)...")
    with h5py.File(saida, "w") as hf:
        ds = hf.create_dataset(
            "matrizes_termicas",
            shape=(num_amostras, 480, 640),
            dtype=np.float32,
        )

        for i, caminho_csv in enumerate(arquivos_matrizes):
            nome_arquivo = os.path.basename(caminho_csv)

            # Lê e grava matriz
            matriz_2d = pd.read_csv(caminho_csv, header=None).values
            ds[i] = matriz_2d

            # Monta registro de metadados
            timestamp_str = "_".join(nome_arquivo.split("_")[2:4])

            if usar_timestamp and df_matched is not None:
                linha = df_matched.iloc[i]
                linha_sensor = {col: linha[col] for col in tag_cols if col in linha.index}
                timestamp_ref = ts_termicas[i]
            else:
                linha_sensor = df_sensores.iloc[i].to_dict()
                timestamp_ref = df_sensores.index[i]

            registro: Dict[str, Any] = {
                "id_amostra": i,
                "timestamp_matriz": timestamp_str,
                "timestamp_excel_original": timestamp_ref,
                "arquivo_origem": nome_arquivo,
            }
            registro.update(linha_sensor)
            dataset_metadata.append(registro)

            if (i + 1) % 10 == 0:
                print(f"  Processado: {i + 1}/{num_amostras}")

    # Salva metadados no mesmo HDF5
    print("Salvando metadados de sensores no HDF5...")
    df_meta = pd.DataFrame(dataset_metadata)
    df_meta.to_hdf(saida, key="sensores", mode="a")

    print(
        f"Concluído: {saida} | amostras: {num_amostras} | tags: {len(tag_cols)}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- Carrega dados SAGE ---
    if args.base_url and args.tags_file:
        print(f"Coletando dados SAGE via API: {args.base_url}")
        df_sensores = load_sage_api(
            args.base_url, args.endpoint, args.token, args.tags_file
        )
    else:
        print(f"Lendo dados SAGE do Excel: {args.excel}")
        df_sensores = load_sage_excel(args.excel)

    print(
        f"  {len(df_sensores)} timestamps × {len(df_sensores.columns)} tags carregados"
    )

    # --- Carrega lista de matrizes ---
    arquivos_matrizes = sorted(glob.glob(os.path.join(args.matrizes, "*.csv")))
    if not arquivos_matrizes:
        print(f"ERRO: nenhum CSV encontrado em {args.matrizes}", file=sys.stderr)
        sys.exit(1)
    print(f"  {len(arquivos_matrizes)} matrizes encontradas em {args.matrizes}")

    # --- Cria HDF5 ---
    build_hdf5(df_sensores, arquivos_matrizes, args.saida, args.tolerancia_ts)


if __name__ == "__main__":
    main()