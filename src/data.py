import pandas as pd
from typing import Union, Iterable, Optional

__all__ = ["load_data", "parse_time"]

def load_data(path: Union[str, "os.PathLike"],
              *,
              dtype: Optional[dict] = None,
              usecols: Optional[Iterable] = None) -> pd.DataFrame:
    """
    Lê o CSV de `path` com pequenos cuidados:
      - Trata valores ausentes comuns
      - Mantém cabeçalhos como strings e remove espaços extras
      - Permite escolher colunas e dtypes, se desejado
    """
    na_vals = ["", "NA", "N/A", "NaN", "nan", "NULL", "null", "None", "none", "?"]
    df = pd.read_csv(
        path,
        dtype=dtype,
        usecols=usecols,
        na_values=na_vals,
        keep_default_na=True,
    )

    # Padroniza nomes de colunas
    df.columns = df.columns.map(lambda c: str(c).strip())

    return df


def parse_time(series: pd.Series,
               *,
               dayfirst: bool = False,
               utc: bool = False) -> pd.Series:
    """
    Converte uma série para datetime de forma vetorizada e robusta.

    - Usa `pd.to_datetime` (muito mais rápido que aplicar parse linha a linha)
    - `errors='coerce'` transforma entradas inválidas em NaT
    - `dayfirst` e `utc` configuráveis
    """
    # Garante série
    s = pd.Series(series)

    # Conversão vetorizada; tenta inferir formatos comuns
    dt = pd.to_datetime(
        s.astype("string"),
        errors="coerce",
        dayfirst=dayfirst,
        utc=utc,
        infer_datetime_format=True,
        format=None,  # deixa livre para múltiplos formatos
    )

    return dt
