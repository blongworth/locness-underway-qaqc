import marimo

__generated_with = "0.15.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import sqlite3
    import altair as alt
    from datetime import timedelta
    return alt, mo, pl, sqlite3, timedelta


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Taking a look at mFET diagnostic data from LOC-02""")
    return


@app.cell
def _(pl, sqlite3):
    db_path = "data/locness.db"

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)

    # Read the 'ph' table into a Polars DataFrame
    df = pl.read_database("SELECT * FROM ph", connection=conn)

    # convert datetime_utc from a unix integer seconds to a datetime
    df = df.with_columns((pl.col("datetime_utc") * 1000).cast(pl.Datetime("ms")))

    # Close the connection
    conn.close()
    return (df,)


@app.cell
def _(pl):
    # Build aggregation expressions for all columns except timestamp
    def resample_ph(df):
        agg_exprs = []
        for col in df.columns:
            if col == "timestamp":
                continue
            dtype = df[col].dtype
            if dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]:
                agg_exprs.append(pl.col(col).mean())
            elif col == "ph_flag":
                agg_exprs.append(pl.col(col).max())
            else:
                agg_exprs.append(pl.col(col).first().alias(col + "_first"))
    
        # Resample to 1-minute intervals
        df_res = df.group_by_dynamic(
            index_column="datetime_utc",
            every="1m",
            closed="left"
        ).agg(agg_exprs)
        return df_res
    return (resample_ph,)


@app.cell
def _(df, resample_ph):
    df_res = resample_ph(df)
    # Display the DataFrame
    df_res
    return (df_res,)


@app.cell
def _(alt, df_res):
    alt.Chart(df_res).mark_line().encode(
        x='datetime_utc:T',
        y=alt.Y('v_bat:Q', scale=alt.Scale(zero=False))
    ).properties(
        title='mFET 05 main battery depletion curve'
    ).interactive()
    return


@app.cell
def _(alt, df_res):
    _main = alt.Chart(df_res).mark_line().encode(
        x=alt.X('datetime_utc:T', axis=alt.Axis(title=None, labels=False, ticks=False)),
        y=alt.Y('v_bat:Q', scale=alt.Scale(zero=False))
    ).properties(
        title='mFET 05 battery depletion curves'
    )

    _pos = alt.Chart(df_res).mark_line().encode(
        x=alt.X('datetime_utc:T'),
        y=alt.Y('v_bias_pos:Q', scale=alt.Scale(zero=False))
    )

    _neg = alt.Chart(df_res).mark_line().encode(
        x='datetime_utc:T',
        y=alt.Y('v_bias_neg:Q', scale=alt.Scale(zero=False))
    )

    (_pos + _neg).properties(
        width=800).interactive()
    return


@app.cell
def _(df_res, mo):
    ph_y_var = mo.ui.dropdown(
        options=df_res.columns, value="ph_total", label="Select Y-axis variable"
    )
    ph_y_var
    return (ph_y_var,)


@app.cell
def _(alt, df_res, ph_y_var):
    alt.Chart(df_res).mark_line().encode(
        x='datetime_utc:T',
        y=alt.Y(f"{ph_y_var.value}:Q", title=ph_y_var.value, scale=alt.Scale(zero=False)),
    ).properties(
        title='mFET diagnostics'
    ).interactive()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""pH Flagged data selection""")
    return


@app.cell
def _(pl):
    ph_flags = pl.read_csv("loc02-ph-qc-flags.csv")
    ph_flags = ph_flags.with_columns([
        pl.col("start_time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
        pl.col("end_time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
    ])

    ph_flags
    return (ph_flags,)


@app.cell
def _(alt, df, mo, ph_flags, ph_y_var, pl, timedelta):
    # For each period, extract and plot data with 5 min buffer
    charts = []
    for i, row in enumerate(ph_flags.iter_rows(named=True)):
        start = row["start_time"] - timedelta(minutes=20)
        end = row["end_time"] + timedelta(minutes=20)
        period_data = df.filter(
            (pl.col("datetime_utc") >= start) & (pl.col("datetime_utc") <= end)
        )
        line = alt.Chart(period_data).mark_line().encode(
            x='datetime_utc:T',
            y=alt.Y(f"{ph_y_var.value}:Q", title=ph_y_var.value, scale=alt.Scale(zero=False)))
        points = alt.Chart(period_data).mark_point().encode(
            x='datetime_utc:T',
            y=alt.Y(f"{ph_y_var.value}:Q", title=ph_y_var.value, scale=alt.Scale(zero=False)))
        chart = (line + points).properties(
            title=f"Period {i+1}: {start} to {end}",
            width=600,
            height=300
        ).interactive()
        charts.append(mo.ui.altair_chart(chart))

    mo.vstack(charts)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    flag data

    * Flag 2 is Acceptable
    * Flag 3 is questionable - conditioning, high error, etc.
    * Flag 4 is known bad - dead batteries, mFET003, conditioning, etc.
    * Will add flag 9 for missing in resampled dataset
    """
    )
    return


@app.cell
def _(df, ph_flags, pl):
    # 1. Set ph_flag to 4 where ph_total > 9
    df_flagged = df.with_columns(
        pl.when(pl.col("ph_total") > 9).then(4).otherwise(2).alias("ph_flag")
    )

    # 2. Set ph_flag to 4 during periods in ph_flags
    for _row in ph_flags.iter_rows(named=True):
        _start = _row["start_time"]
        _end = _row["end_time"]
        # Use a mask to update ph_flag to 4 in the period
        _mask = (df_flagged["datetime_utc"] >= _start) & (df_flagged["datetime_utc"] <= _end)
        # Set ph_flag to 4 where mask is True
        df_flagged = df_flagged.with_columns(
            pl.when(_mask).then(4).otherwise(pl.col("ph_flag")).alias("ph_flag")
        ).with_columns(
        pl.col("ph_flag").cast(pl.String)
        )
    return (df_flagged,)


@app.cell
def _(df_flagged, resample_ph):
    dfr = resample_ph(df_flagged)
    dfr
    return (dfr,)


@app.cell
def _(alt, dfr, ph_y_var):
    alt.Chart(dfr).mark_line().encode(
        x='datetime_utc:T',
        y=alt.Y(f"{ph_y_var.value}:Q", title=ph_y_var.value, scale=alt.Scale(zero=False)),
        color="ph_flag"
    ).properties(
        title='mFET diagnostics with QC flags'
    ).interactive()
    return


@app.cell
def _(alt, dfr, ph_y_var, pl):
    dfrf = dfr.filter(pl.col("ph_flag") != "4")
    alt.Chart(dfrf).mark_line().encode(
        x='datetime_utc:T',
        y=alt.Y(f"{ph_y_var.value}:Q", title=ph_y_var.value, scale=alt.Scale(zero=False)),
    ).properties(
        title='mFET diagnostics with QC flags'
    ).interactive()
    return


@app.cell
def _():
    # write the flagged data

    return


if __name__ == "__main__":
    app.run()
