import marimo

__generated_with = "0.15.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import altair as alt
    return alt, pl


@app.cell
def _(pl):
    df = pl.read_parquet("data/filled_gps.parquet")
    df
    return (df,)


@app.cell
def _(df, pl):
    resampled = (
        df
        .group_by_dynamic(
            index_column="datetime_utc",
            every="10s",
            closed="left"
        )
        .agg([
            #pl.col("datetime_utc").first(),
            pl.col("latitude").first(),
            pl.col("longitude").first(),
            pl.col("source").first()
        ])
    )
    resampled
    return (resampled,)


@app.cell
def _(alt, resampled):
    alt.data_transformers.enable("vegafusion")
    alt.Chart(resampled).mark_point().encode(
        x=alt.X("longitude", scale=alt.Scale(zero=False)),
        y=alt.Y("latitude", scale=alt.Scale(zero=False)),
        color="source",
        tooltip=alt.Tooltip('datetime_utc:T', title='Timestamp', format='%Y-%m-%d %H:%M:%S'),
    ).interactive()
    return


@app.cell
def _(df, pl):
    df_cut = df.filter((pl.col("datetime_utc") > pl.datetime(2025, 8, 14, 9, 5, 0)) & (pl.col("datetime_utc") < pl.datetime(2025, 8, 14, 9, 50, 0)))
    df_cut
    return (df_cut,)


@app.cell
def _(alt, df_cut):
    alt.Chart(df_cut).mark_point().encode(
        x="longitude",
        y="latitude",
        color="source",
        tooltip=alt.Tooltip('datetime_utc:T', title='Timestamp', format='%Y-%m-%d %H:%M:%S'),
    ).interactive()
    return


if __name__ == "__main__":
    app.run()
