import polars as pl
import gpxpy
import sqlite3
from datetime import timedelta

def read_gpx_to_df(gpx_path):
    with open(gpx_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
        data = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    data.append({
                        'datetime_utc': point.time,
                        'latitude': point.latitude,
                        'longitude': point.longitude,
                    })
        df = pl.DataFrame(data)
        # Convert datetime to UTC timezone explicitly
        df = df.with_columns(pl.col('datetime_utc').cast(pl.Datetime("us")))
        return df

def read_gps_table_to_polars(sqlite_path, table_name="gps"):
    conn = sqlite3.connect(sqlite_path)
    query = f"SELECT * FROM {table_name}"
    df = pl.read_database(query, conn)
    # Convert 'datetime_utc' from unix epoch integer to polars Datetime in UTC
    if "datetime_utc" in df.columns:
        df = df.with_columns(
            (pl.col("datetime_utc") * 1_000_000).cast(pl.Datetime("us"))
        )
    conn.close()
    return df

def resample_to_regular_timeseries(df: pl.DataFrame, time_col: str = "time", interval_s: int = 2) -> pl.DataFrame:
    # Ensure the time column is in datetime format
    df = df.with_columns(pl.col(time_col).cast(pl.Datetime("us")))
    start = df[time_col].min()
    end = df[time_col].max()
    # Generate regular timeseries
    times = pl.datetime_range(
        start=start,
        end=end,
        interval=timedelta(seconds=interval_s),
        eager=True
    )
    # Create regular timeseries dataframe
    regular_df = pl.DataFrame({time_col: times})
    regular_df = regular_df.with_columns(pl.col(time_col).cast(pl.Datetime("us")))
    
    # Join with original data, keeping only exact matches
    result = regular_df.join_asof(
        df,
        left_on=time_col,
        right_on=time_col,
        strategy="backward",
        tolerance=timedelta(seconds=1.5)
    )
    return result

def fill_missing(primary: pl.DataFrame, secondary: pl.DataFrame, time_col: str = "datetime_utc", value_cols: list = None) -> pl.DataFrame:
    if value_cols is None:
        value_cols = [col for col in primary.columns if col != time_col]
    # Sort both dataframes by time
    primary = primary.sort(time_col)
    secondary = secondary.sort(time_col)
    # Suffix columns in secondary to avoid name clash
    secondary_renamed = secondary.rename({col: f"{col}_sec" for col in value_cols})
    # Left join on time
    joined = primary.join_asof(
        secondary_renamed,
        left_on=time_col,
        right_on=time_col,
        strategy="backward",
        tolerance=timedelta(seconds=1.5)
    )
    # Add source column based on whether primary data is null
    source_expr = pl.when(pl.col(value_cols[0]).is_null()).then(pl.lit("gpx")).otherwise(pl.lit("gps"))
    joined = joined.with_columns(source_expr.alias("source"))
    
    # Fill missing values in primary with secondary
    for col in value_cols:
        joined = joined.with_columns(
            pl.col(col).fill_null(pl.col(f"{col}_sec")).alias(col)
        )
    # Select time, value columns, and source column
    joined = joined.select([time_col] + value_cols + ["source"])
    return joined

def main():

    db_path = "data/locness.db"
    gpx_file = "data/2025_08_13_Subhas_TimeZeroTracks.gpx"
    time_col = "datetime_utc"
    value_cols = ["latitude", "longitude"]
    interval = 1  # seconds

    # Read GPS and GPX data
    gps_df = read_gps_table_to_polars(db_path)
    print("GPS DataFrame summary:")
    print(gps_df.describe())
    print(gps_df.head(10))

    gpx_df = read_gpx_to_df(gpx_file)
    print("GPX DataFrame summary:")
    print(gpx_df.describe())
    print(gpx_df.head(10))

    # Resample GPS data to regular timeseries
    gps_regular = resample_to_regular_timeseries(gps_df, time_col=time_col, interval_s=interval)
    print("Resampled GPS DataFrame summary:")
    print(gps_regular.describe())
    
    # Print detailed summaries of each dataset before combining
    print("\nGPS Dataset Details:")
    print("First row:")
    print(gps_regular.head(1))
    print("\nLast row:")
    print(gps_regular.tail(1))
    # Calculate median time difference in seconds
    gps_diffs = (gps_regular
                 .with_columns((pl.col("datetime_utc").diff().dt.total_seconds()).alias("time_diff"))
                 .drop_nulls()
                 .get_column("time_diff"))
    gps_median_freq = float(gps_diffs.median())
    print(f"Median time frequency: {gps_median_freq:.2f} seconds")
    print(f"Number of null coordinates: {gps_regular.filter(pl.col('latitude').is_null()).height}")

    print("\nGPX Dataset Details:")
    print("First row:")
    print(gpx_df.head(1))
    print("\nLast row:")
    print(gpx_df.tail(1))
    # Calculate median time difference in seconds
    gpx_diffs = (gpx_df
                 .with_columns((pl.col("datetime_utc").diff().dt.total_seconds()).alias("time_diff"))
                 .drop_nulls()
                 .get_column("time_diff"))
    gpx_median_freq = float(gpx_diffs.median())
    print(f"Median time frequency: {gpx_median_freq:.2f} seconds")
    
    # Add 46 seconds offset to GPX times
    gpx_df = gpx_df.with_columns((pl.col("datetime_utc") + timedelta(seconds=46)))
    
    # Fill missing values from GPX data
    filled = fill_missing(gps_regular, gpx_df, time_col=time_col, value_cols=value_cols)

    # print summary
    print("\nCombined DataFrame summary:")
    print(filled.describe())
    
    # Find the first row where GPS data was null but got filled from GPX
    null_mask = gps_regular.get_column("latitude").is_null()
    first_gap_time = gps_regular.filter(null_mask).get_column("datetime_utc")[0]
    
    # Show data around the gap (5 rows before and after)
    window = timedelta(seconds=10)  # 5 rows * 2 seconds interval
    print("\nData around first gap:")
    print("\nBefore filling:")
    print(gps_regular.filter(
        (pl.col("datetime_utc") >= first_gap_time - window) &
        (pl.col("datetime_utc") <= first_gap_time + window)
    ).select(["datetime_utc", "latitude", "longitude"]))
    
    print("\nAfter filling:")
    print(filled.filter(
        (pl.col("datetime_utc") >= first_gap_time - window) &
        (pl.col("datetime_utc") <= first_gap_time + window)
    ))
    
    # Write filled DataFrame to parquet file
    output_path = "data/filled_gps.parquet"
    filled.write_parquet(output_path)
    print(f"\nWrote filled GPS data to {output_path}")

if __name__ == "__main__":
    main()
