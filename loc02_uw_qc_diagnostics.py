import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    alt.data_transformers.enable("vegafusion")
    return alt, pl


@app.cell
def _(pl):
    df = pl.read_parquet("loc02_uw_qc.parquet")
    df
    return (df,)


@app.cell
def _(df, pl):
    # Resample to 10-second frequency, taking the mean of all value columns
    filtered = (
        df
        .filter(pl.col("ph_flag") == "2")
        .drop("ph_flag")
    )
    filtered
    return (filtered,)


@app.cell
def _(df, pl):
    uf_res = df.group_by_dynamic(
        index_column="datetime_utc",
        every="10s",
        closed="left",
        label="left"
    ).agg([
        pl.all().exclude(["datetime_utc", "ph_flag"]).mean()
    ])
    return (uf_res,)


@app.cell
def _(filtered, pl):
    resampled = filtered.group_by_dynamic(
        index_column="datetime_utc",
        every="10s",
        closed="left",
        label="left"
    ).agg([
        pl.all().exclude("datetime_utc").mean()
    ])

    return (resampled,)


@app.cell
def _(alt, resampled):
    alt.Chart(resampled).mark_point().encode(
        x="longitude",
        y="latitude",
        color="ph_corrected",
        tooltip=["datetime_utc", "longitude", "latitude"]
    ).interactive()
    return


@app.cell
def _(alt, resampled):
    alt.Chart(resampled).mark_line().encode(
        x="datetime_utc",
        y=alt.Y(f"ph_corrected:Q", title="Calibrated pH", scale=alt.Scale(zero=False)),
    ).interactive()
    return


@app.cell
def _(alt, resampled):
    alt.Chart(resampled).mark_line().encode(
        x="datetime_utc",
        y=alt.Y(f"rho_ppb:Q", title="Rhodamine [ppb]", scale=alt.Scale(zero=False)),
    ).interactive()
    return


@app.cell
def _(alt, uf_res):
    alt.Chart(uf_res).mark_line().encode(
        x="datetime_utc",
        y=alt.Y(f"temperature:Q", title="Temperature [C]", scale=alt.Scale(zero=False)),
    ).interactive()
    return


if __name__ == "__main__":
    app.run()
