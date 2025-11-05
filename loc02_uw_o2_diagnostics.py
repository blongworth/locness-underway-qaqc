import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    return (pl,)


@app.cell
def _(pl):
    df = pl.read_csv("data/loc02_do_optode_qc.csv").with_columns(pl.col("datetime").str.strptime(pl.Datetime, strict=False))
    df
    return (df,)


@app.cell
def _(df, pl):
    # Sort, compute diffs, and filter gaps
    sorted_df = df.sort("datetime")
    out = (
        sorted_df
        .with_columns([
            pl.col("datetime").diff().alias("gap"),  # duration type
            pl.col("datetime").shift(1).alias("prev_ts"),
        ])
        .filter(pl.col("gap").is_not_null())
        .filter(pl.col("gap") > 5_000_000)
        .select(["prev_ts", "datetime", "gap"])
        .rename({"datetime": "curr_ts"})
    )

    out
    return


if __name__ == "__main__":
    app.run()
