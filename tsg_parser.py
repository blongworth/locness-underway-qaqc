"""
Parser for TSG (Thermosalinograph) data files.

This module provides functionality to read and parse TSG data files into pandas DataFrames.
The TSG files contain temperature, conductivity, salinity, and position data.
"""

import polars as pl
import sqlite3
from datetime import datetime, timedelta
import re

def parse_decdeg(coord_str):
    """Convert degrees and decimal minutes format to decimal degrees.
    
    Args:
        coord_str: String like "41 32.9983 N" or "070 41.7965 W"
    
    Returns:
        float: Decimal degrees, negative for South or West
    """
    parts = coord_str.split()
    degrees = float(parts[0])
    minutes = float(parts[1])
    hemisphere = parts[2]
    
    decimal = degrees + minutes/60.0
    
    if hemisphere in ['S', 'W']:
        decimal = -decimal
        
    return decimal

def parse_datetime(hms, dmy):
    """Convert HHMMSS and DDMMYY strings to datetime object.
    
    Args:
        hms: String in format "HHMMSS"
        dmy: String in format "DDMMYY"
    
    Returns:
        datetime: Combined date and time
    """
    # Extract components
    hour = int(hms[0:2])
    minute = int(hms[2:4]) 
    second = int(hms[4:6])
    
    day = int(dmy[0:2])
    month = int(dmy[2:4])
    year = 2000 + int(dmy[4:6]) # Assuming years 2000-2099
    
    return datetime(year, month, day, hour, minute, second)

def fill_nulls_with_increment(series):
    result = []
    last_val = None
    for val in series:
        if val is not None:
            last_val = val
            result.append(val)
        else:
            if last_val is not None:
                last_val = last_val + timedelta(seconds=2)
                result.append(last_val)
            else:
                result.append(None)
    return result


def read_tsg(filepath):
    """Read a TSG data file and return a polars DataFrame.

    Args:
        filepath: Path to the TSG data file
        
    Returns:
        polars.DataFrame with columns:
        - datetime: Combined timestamp
        - t1: Primary temperature (°C)
        - c1: Conductivity (S/m)
        - s: Salinity (PSU)
        - t2: Secondary temperature (°C) (optional)
        - latitude: Decimal degrees (optional)
        - longitude: Decimal degrees (optional)
    """
    # Initialize empty lists to store data
    data = []
    errors_reported = set()  # Track line numbers where errors were reported
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Preprocess line: if it ends with dmy=xxxxx$*, replace $ with 5 and remove trailing *
                if 'dmy=' in line and '$' in line:
                    dmy_pattern = r'dmy=(\d{5})\$\*?.*$'
                    line = re.sub(dmy_pattern, r'dmy=\g<1>5', line)
                
                # Use regex to find all key=value pairs
                pairs = re.findall(r'(\w+)=\s*([-\d.]+(?:\s+[-\d.]+\s+[NSEW])?)', line)
                if not pairs:
                    continue  # Skip empty or invalid lines silently
                
                # Initialize values for this line
                values = {'t1': None, 'c1': None, 's': None, 't2': None,
                         'latitude': None, 'longitude': None, 'datetime': None}
                
                # Store time components separately
                time_values = {}
                
                # Parse each key=value pair
                for key, value in pairs:
                    try:
                        if key in ['t1', 'c1', 's', 't2']:
                            values[key] = float(value)
                        elif key == 'lat':
                            values['latitude'] = parse_decdeg(value)
                        elif key == 'lon':
                            values['longitude'] = parse_decdeg(value)
                        elif key in ['hms', 'dmy']:
                            time_values[key] = value.strip()
                    except ValueError:
                        if line_num not in errors_reported:
                            print(f"Warning: Could not parse value for {key} at line {line_num}")
                            errors_reported.add(line_num)
                        continue
                
                # Create datetime if we have both time components
                if 'hms' in time_values and 'dmy' in time_values:
                    try:
                        hms = time_values['hms']
                        dmy = time_values['dmy']
                        # If seconds == 60, increment to next minute
                        if hms[-2:] == "60":
                            hour = int(hms[0:2])
                            minute = int(hms[2:4])
                            # increment minute, handle overflow
                            minute += 1
                            if minute == 60:
                                minute = 0
                                hour += 1
                                if hour == 24:
                                    hour = 0
                                    # increment day in dmy
                                    day = int(dmy[0:2])
                                    month = int(dmy[2:4])
                                    year = 2000 + int(dmy[4:6])
                                    dt = datetime(year, month, day)
                                    dt += timedelta(days=1)
                                    dmy = f"{dt.day:02d}{dt.month:02d}{str(dt.year)[2:]}"
                            hms = f"{hour:02d}{minute:02d}00"
                        values['datetime'] = parse_datetime(hms, dmy)
                    except ValueError as e:
                        if line_num not in errors_reported:
                            print(f"Warning: Invalid datetime at line {line_num}: {str(e)}")
                            errors_reported.add(line_num)
                            values['datetime'] = None
                
                # Check for minimum required fields (temperature, conductivity and salinity)
                if any(values[field] is None for field in ['t1', 'c1', 's']):
                    if line_num not in errors_reported:
                        missing = [f for f in ['t1', 'c1', 's'] if values[f] is None]
                        print(f"Warning: Missing required fields {missing} at line {line_num}")
                        errors_reported.add(line_num)
                    continue
                
                data.append(values)
                
            except Exception as e:
                if line_num not in errors_reported:
                    print(f"Warning: Error processing line {line_num}: {str(e)}")
                    errors_reported.add(line_num)
                continue
    
    if not data:
        print("Error: No valid data rows found in file")
        return None
        
    # Create DataFrame
    df = pl.DataFrame(data)
    if df['datetime'].null_count() > 0:
        filled = fill_nulls_with_increment(df["datetime"].to_list())

        # Replace the column in the DataFrame
        df = df.with_columns(
            pl.Series("ts", filled)
        )
    return df
    

def read_tsg_table_to_polars(sqlite_path, table_name="tsg"):
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
        tolerance=timedelta(seconds=2.5)
    )
    # Add source column based on whether primary data is null
    source_expr = pl.when(pl.col(value_cols[0]).is_null()).then(pl.lit("ship")).otherwise(pl.lit("uw"))
    joined = joined.with_columns(source_expr.alias("source"))
    
    # Fill missing values in primary with secondary
    for col in value_cols:
        joined = joined.with_columns(
            pl.col(col).fill_null(pl.col(f"{col}_sec")).alias(col)
        )
    # Select time, value columns, and source column
    joined = joined.select([time_col] + value_cols + ["source"])
    return joined


def fill_tsg(db_path="data/locness.db",tsg_file="data/TSG_2025_08_12_Subhas1.cap", output_path="data/filled_tsg.parquet"):
    uw_df = read_tsg_table_to_polars(db_path, table_name="tsg").select([
        pl.col("datetime_utc"),
        pl.col("temp").alias("temperature"),
        pl.col("cond"),
        pl.col("salinity")
    ])
    print(uw_df.columns)
    ship_df = read_tsg(tsg_file)
    ship_df = ship_df.select([
        pl.col("ts").alias("datetime_utc"),
        pl.col("t1").alias("temperature"),
        pl.col("c1").alias("cond"),
        pl.col("s").alias("salinity")
    ])

    print(ship_df.describe())

    filled_df = fill_missing(uw_df, ship_df)

    if filled_df.height > 0:
        # Get first and last times
        first_row = filled_df.row(0)
        last_row = filled_df.row(-1)
        # Get the datetime column index
        datetime_idx = filled_df.get_column_index('datetime_utc')
        first_time = first_row[datetime_idx]
        last_time = last_row[datetime_idx]
        
        print(f"Time range: {first_time} to {last_time}")
        print(f"Number of records: {filled_df.height}")
        print("\nSummary Statistics:")
        print(filled_df.describe())
    else:
        print("DataFrame is empty")
    
    filled_df.write_parquet(output_path)
    return filled_df
    
if __name__ == '__main__':
    fill_tsg()
        