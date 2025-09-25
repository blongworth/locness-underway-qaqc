
import polars as pl
import sqlite3
from datetime import timedelta

def read_ph_flags(flag_path: str) -> pl.DataFrame:
    ph_flags = pl.read_csv(flag_path)
    ph_flags = ph_flags.with_columns([
        pl.col("start_time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
        pl.col("end_time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
    ])
    return ph_flags


def apply_ph_flags(df, ph_flags):
    # 1. Set ph_flag to 4 where ph_total > 9
    df_flagged = df.with_columns(
        pl.when(pl.col("ph_total") > 9).then(4).otherwise(2).alias("ph_flag")
    )

    # 2. Set ph_flag to 4 during periods in ph_flags
    for row in ph_flags.iter_rows(named=True):
        start = row["start_time"]
        end = row["end_time"]
        # Use a mask to update ph_flag to 4 in the period
        mask = (df_flagged["datetime_utc"] >= start) & (df_flagged["datetime_utc"] <= end)
        # Set ph_flag to 4 where mask is True
        df_flagged = df_flagged.with_columns(
            pl.when(mask).then(4).otherwise(pl.col("ph_flag")).alias("ph_flag")
        ).with_columns(
        pl.col("ph_flag").cast(pl.String)
        )
    return df_flagged
