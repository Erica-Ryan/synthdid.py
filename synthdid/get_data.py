import pandas as pd
import numpy as np

def california_prop99() -> pd.DataFrame:
    url = "https://github.com/d2cml-ai/Synthdid.jl/raw/stag_treat/data/california_prop99.csv"
    try:
        return pd.read_csv(url, sep=";")
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch california_prop99 data from {url}. "
            "Check your internet connection or download the file manually.\n"
            f"Original error: {e}"
        )

def quota() -> pd.DataFrame:
    url = "https://github.com/d2cml-ai/Synthdid.jl/raw/stag_treat/data/quota.csv"
    try:
        return pd.read_csv(url)
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch quota data from {url}. "
            "Check your internet connection or download the file manually.\n"
            f"Original error: {e}"
        )