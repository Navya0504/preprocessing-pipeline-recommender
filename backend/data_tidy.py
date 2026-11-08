import numpy as np
import pandas as pd


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Trim text, convert '?' and empty strings to missing values,
    and convert text columns that are mostly numeric into numbers.
    """
    df = df.copy()

    for column in df.select_dtypes(include="object").columns:
        text = (
            df[column]
            .astype("string")
            .str.strip()
            .replace({"?": pd.NA, "": pd.NA})
        )

        numeric = pd.to_numeric(text, errors="coerce")
        filled = int(text.notna().sum())

        if filled and int(numeric.notna().sum()) >= 0.9 * filled:
            df[column] = numeric.astype("float64")
        else:
            df[column] = text.astype(object).where(
              text.notna(), np.nan
    
        )

    return df
