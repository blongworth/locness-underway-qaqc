import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    alt.data_transformers.enable("vegafusion")
    return alt, mo, pl


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
            .filter(~pl.col("ph_flag").is_in([3, 4]),
                    ~pl.col("salinity_flag").is_in([3, 4]),
                    ~pl.col("rho_flag").is_in([3, 4]),
                    ~pl.col("temperature_flag").is_in([3, 4]))
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
def _(alt, uf_res):
    alt.Chart(uf_res).mark_line().encode(
        x="datetime_utc",
        y=alt.Y(f"longitude:Q", title="longitude", scale=alt.Scale(zero=False)),
    ).interactive()
    return


@app.cell
def _(alt, uf_res):
    alt.Chart(uf_res).mark_line().encode(
        x="datetime_utc",
        y=alt.Y(f"latitude:Q", title="latitude", scale=alt.Scale(zero=False)),
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Small spikes do not show in 50 ppb bucket test (2025-07-29 to 2025-07-31). I suspect these are small air bubbles in the underway system. Also need to investigate whether dips and humps correlate with flow issues. Value definitely spikes when flow interrupted for ph sensor changes.""")
    return


@app.cell
def _(alt, resampled):
    alt.Chart(resampled).mark_line().encode(
        x="datetime_utc",
        y=alt.Y(f"temperature:Q", title="Temperature [C]", scale=alt.Scale(zero=False)),
    ).interactive()
    return


@app.cell
def _(alt, resampled):
    alt.Chart(resampled).mark_line().encode(
        x="datetime_utc",
        y=alt.Y(f"salinity:Q", title="Salinity [PSU]", scale=alt.Scale(zero=False)),
    ).interactive()
    return


@app.cell
def _(alt, pl, uf_res):
    uf_res_sal = uf_res.filter(pl.col("salinity") > 30)
    alt.Chart(uf_res_sal).mark_line().encode(
        x="datetime_utc",
        y=alt.Y(f"salinity:Q", title="Salinity [PSU]", scale=alt.Scale(zero=False)),
    ).interactive()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Temp is also spiky/noisy. Not sure what's going on here.""")
    return


@app.cell
def _(alt, df, pl):
    _df = df.filter((pl.col('datetime_utc') > pl.datetime(2025, 8, 17, 9)) & (pl.col('datetime_utc') < pl.datetime(2025, 8, 17, 10)))
    alt.Chart(_df).mark_point().encode(
        x="datetime_utc",
        y="salinity",
        tooltip=alt.Tooltip('datetime_utc:T', title='Date and Time', format='%Y-%m-%d %H:%M:%S'),
    ).interactive()
    return


@app.cell
def _(alt, df, pl):
    _df = df.filter(pl.col('salinity') < 30)
    alt.Chart(_df).mark_point().encode(
        x="datetime_utc",
        y="salinity",
        tooltip=alt.Tooltip('datetime_utc:T', title='Date and Time', format='%Y-%m-%d %H:%M:%S'),
    ).interactive()
    return


@app.cell
def _(alt, df, pl):
    _df = df.filter((pl.col('datetime_utc') > pl.datetime(2025, 8, 17, 9)) & (pl.col('datetime_utc') < pl.datetime(2025, 8, 17, 10)))
    alt.Chart(_df).mark_point().encode(
        x="datetime_utc",
        y="rho_ppb",
        color='rho_flag',
        tooltip=alt.Tooltip('datetime_utc:T', title='Date and Time', format='%Y-%m-%d %H:%M:%S'),
    ).interactive()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
