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
    df = pl.read_parquet('data/filled_tsg.parquet')
    df
    return (df,)


@app.cell
def _(df, pl):
    df_res  = df.group_by_dynamic(
        index_column="datetime_utc",
        every="10s",
        closed="left",
        label="left"
    ).agg([
        pl.all().exclude(["datetime_utc", "source"]).mean(),
        pl.col('source').first()
    ])
    return (df_res,)


@app.cell
def _(alt, df_res):
    alt.Chart(df_res).mark_point(size=1).encode(
        x='datetime_utc',
        y=alt.Y('temperature',scale=alt.Scale(zero=False)),
        color='source'
    ).interactive()
    return


@app.cell
def _(alt, df_res):
    alt.Chart(df_res).mark_point(size=1).encode(
        x='datetime_utc',
        y=alt.Y('salinity',scale=alt.Scale(zero=False)),
        color='source'
    ).interactive()
    return


if __name__ == "__main__":
    app.run()
