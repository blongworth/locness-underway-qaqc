import os
import pandas as pd
import sqlite3
import boto3
import numpy as np
from locness_datamanager.config import get_config


# Load the config file
config = get_config()

# Get the parquet path from the config
parquet_path = config.get('parquet_path')

# Read the parquet dataset
if parquet_path and os.path.exists(parquet_path):
    df = pd.read_parquet(parquet_path)

def check_data_ranges(df, label):
    """Check for values outside expected ranges"""
    print(f"\n--- {label} Data Range Checks ---")
    
    # Define expected ranges for common oceanographic parameters
    ranges = {
        'temperature': (10, 25),  # Celsius
        'salinity': (25, 37),      # PSU
        'ph': (7.5, 9.5),             # pH units
        'rho_ppb': (-1, 1000),             # pH units
        'latitude': (41.4, 43),    # degrees
        'longitude': (-71, -69), # degrees
        'conductivity': (0, 70),  # mS/cm
    }
    
    for col in df.columns:
        if col.lower() in ranges and pd.api.types.is_numeric_dtype(df[col]):
            min_val, max_val = ranges[col.lower()]
            out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
            if not out_of_range.empty:
                print(f"  {col}: {len(out_of_range)} values out of range [{min_val}, {max_val}]")
                print(f"    Min: {df[col].min()}, Max: {df[col].max()}")
            else:
                print(f"  {col}: All values within expected range [{min_val}, {max_val}]")

def check_data_consistency(df, label):
    """Check for logical inconsistencies between related fields"""
    print(f"\n--- {label} Data Consistency Checks ---")
    
    # Check if temperature and salinity are consistent with conductivity
    if all(col in df.columns for col in ['temperature', 'salinity', 'conductivity']):
        # Basic sanity check: higher salinity should generally mean higher conductivity
        temp_sal_cond = df[['temperature', 'salinity', 'conductivity']].dropna()
        if len(temp_sal_cond) > 10:
            sal_cond_corr = temp_sal_cond['salinity'].corr(temp_sal_cond['conductivity'])
            print(f"  Salinity-Conductivity correlation: {sal_cond_corr:.3f}")
            if sal_cond_corr < 0.5:
                print("    Warning: Low correlation between salinity and conductivity")
    
    # Check GPS coordinates for sudden jumps
    if all(col in df.columns for col in ['latitude', 'longitude', 'datetime_utc']):
        gps_data = df[['latitude', 'longitude', 'datetime_utc']].dropna().sort_values('datetime_utc')
        if len(gps_data) > 1:
            # Calculate distance between consecutive points
            lat_diff = gps_data['latitude'].diff()
            lon_diff = gps_data['longitude'].diff()
            # Rough distance calculation (not precise but good for detecting jumps)
            distance = ((lat_diff ** 2 + lon_diff ** 2) ** 0.5) * 111  # km
            large_jumps = distance > 10  # More than 10 km between points
            if large_jumps.any():
                print(f"  GPS: {large_jumps.sum()} large position jumps (>10km) detected")
            else:
                print("  GPS: No large position jumps detected")

def check_running_mean_outliers(df, label, window=5, threshold=3):
    """
    Detect outliers based on deviation from running mean.
    Outlier if value deviates from running mean by more than threshold * running std.
    """
    print(f"\n--- {label} Running Mean Outlier Detection ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ['datetime_utc']:
            continue
        series = df[col]
        rolling_mean = series.rolling(window=window, center=True, min_periods=1).mean()
        rolling_std = series.rolling(window=window, center=True, min_periods=1).std()
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)
        deviation = (series - rolling_mean).abs()
        outliers = deviation > (threshold * rolling_std)
        num_outliers = outliers.sum()
        percent_outliers = (num_outliers / len(series) * 100) if len(series) > 0 else 0
        if num_outliers > 0:
            print(f"  {col}: {num_outliers} running mean outliers detected ({percent_outliers:.2f}%)")
        else:
            print(f"  {col}: No running mean outliers detected")

def check_data_completeness(df, label):
    """Check data completeness and sampling rates"""
    print(f"\n--- {label} Data Completeness ---")
    
    if 'datetime_utc' in df.columns:
        df_sorted = df.sort_values('datetime_utc')
        time_range = df_sorted['datetime_utc'].max() - df_sorted['datetime_utc'].min()
        median_interval = df_sorted['datetime_utc'].diff().median()
        if pd.notna(median_interval) and median_interval.total_seconds() > 0:
            expected_points = time_range.total_seconds() / median_interval.total_seconds()
            actual_points = len(df_sorted)
            completeness = (actual_points / expected_points) * 100 if expected_points > 0 else 0
            
            print(f"  Time range: {time_range}")
            print(f"  Expected data points: {expected_points:.0f}")
            print(f"  Actual data points: {actual_points}")
            print(f"  Data completeness: {completeness:.1f}%")
            
            # Check for data gaps
            gaps = df_sorted['datetime_utc'].diff()
            large_gaps = gaps > median_interval * 5  # Gaps 5x larger than median
            if large_gaps.any():
                print(f"  Large data gaps detected: {large_gaps.sum()}")

def check_data_quality_scores(df, label):
    """Calculate overall data quality scores"""
    print(f"\n--- {label} Data Quality Summary ---")
    
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    completeness_score = ((total_cells - missing_cells) / total_cells) * 100
    
    # Count duplicates
    duplicate_rows = df.duplicated().sum()
    uniqueness_score = ((len(df) - duplicate_rows) / len(df)) * 100 if len(df) > 0 else 0
    
    print(f"  Completeness Score: {completeness_score:.1f}%")
    print(f"  Uniqueness Score: {uniqueness_score:.1f}%")
    print(f"  Total cells: {total_cells:,}")
    print(f"  Missing cells: {missing_cells:,}")
    print(f"  Duplicate rows: {duplicate_rows}")

def check_cross_table_consistency(config):
    """Check consistency across different data sources"""
    print("\n--- Cross-Table Consistency Checks ---")
    
    db_path = config.get('db_path')
    tables_data = {}
    
    # Load data from multiple tables
    table_names = ["underway_summary", "tsg", "gps"]
    with sqlite3.connect(db_path) as conn:
        for table in table_names:
            try:
                df_table = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                if 'datetime_utc' in df_table.columns:
                    df_table['datetime_utc'] = pd.to_datetime(df_table['datetime_utc'], errors='coerce')
                    tables_data[table] = df_table
            except Exception as e:
                print(f"  Could not load {table}: {e}")
    
    # Check overlapping time ranges
    if len(tables_data) > 1:
        for table1, table2 in [(t1, t2) for i, t1 in enumerate(tables_data.keys()) 
                              for t2 in list(tables_data.keys())[i+1:]]:
            df1, df2 = tables_data[table1], tables_data[table2]
            
            overlap_start = max(df1['datetime_utc'].min(), df2['datetime_utc'].min())
            overlap_end = min(df1['datetime_utc'].max(), df2['datetime_utc'].max())
            
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                print(f"  {table1} & {table2}: {overlap_duration} overlap")
            else:
                print(f"  {table1} & {table2}: No time overlap")

def print_datetime_regularity(df, label):
    print(f"\n--- {label} datetime_utc regularity ---")
    if 'datetime_utc' not in df.columns:
        print("datetime_utc column not found!")
        return
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['datetime_utc']):
        # Try integer seconds first, then fallback to generic parse
        if pd.api.types.is_integer_dtype(df['datetime_utc']):
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], unit='s')
        else:
            df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    print(f"Total rows: {len(df)}")
    print("First 5 rows:")
    print(df.head())

    # Check for missing data in any column
    missing_counts = df.isnull().sum()
    total_rows = len(df)
    any_missing = missing_counts.any()
    if any_missing:
        print("Missing data detected:")
        for col, count in missing_counts.items():
            if count > 0:
                percent = (count / total_rows * 100) if total_rows > 0 else 0
                print(f"  {col}: {count} missing ({percent:.2f}%)")
    else:
        print("No missing data detected in any column.")

    # Check monotonicity (strictly increasing)
    diffs_monotonic = df['datetime_utc'].diff().dt.total_seconds()
    non_monotonic_idx = diffs_monotonic < 0
    num_non_monotonic = non_monotonic_idx.sum()
    percent_non_monotonic = (num_non_monotonic / total_rows * 100) if total_rows > 0 else 0
    if num_non_monotonic == 0:
        print("Timestamps are strictly increasing (monotonic).")
    else:
        print(f"Warning: Timestamps are NOT strictly increasing! Non-monotonic points: {num_non_monotonic} / {total_rows} ({percent_non_monotonic:.2f}%)")
        print("Non-monotonic rows (where time goes backwards):")
        print(df.loc[non_monotonic_idx, 'datetime_utc'])

    # Sort by datetime_utc for regularity checks
    df = df.sort_values('datetime_utc')
    # Calculate time differences in seconds
    diffs = df['datetime_utc'].diff().dt.total_seconds().dropna()
    print("Datetime regularity check:")
    print(f"Mean interval: {diffs.mean()} seconds")
    print(f"Median interval: {diffs.median()} seconds")
    print(f"Std deviation: {diffs.std()} seconds")
    print(f"Min interval: {diffs.min()} seconds")
    print(f"Max interval: {diffs.max()} seconds")
    # Check for duplicate timestamps
    duplicate_count = df['datetime_utc'].duplicated().sum()
    if duplicate_count > 0:
        print(f"Duplicate timestamps detected: {duplicate_count} out of {len(df)} rows")
    else:
        print("No duplicate timestamps detected.")
    irregular = diffs[(diffs - diffs.mean()).abs() > 2 * diffs.std()]
    total_intervals = len(diffs)
    num_irregular = len(irregular)
    percent_irregular = (num_irregular / total_intervals * 100) if total_intervals > 0 else 0
    if not irregular.empty:
        print(f"Irregular intervals detected: {num_irregular} out of {total_intervals} ({percent_irregular:.2f}%)")
        print("Irregular intervals with timestamps:")
        for idx, interval in irregular.items():
            # Get the timestamp at the beginning and end of this irregular interval
            start_time = df.iloc[idx - 1]['datetime_utc'] if idx > 0 else df.iloc[0]['datetime_utc']
            end_time = df.iloc[idx]['datetime_utc']
            print(f"  Interval: {interval:.2f}s, Start: {start_time}, End: {end_time}")
    else:
        print("All intervals are regular within 2 standard deviations.")

    # Run additional data integrity checks
    check_data_ranges(df, label)
    check_data_consistency(df, label)
    check_running_mean_outliers(df, label)
    check_data_completeness(df, label)
    check_data_quality_scores(df, label)


def check_dynamodb_datetime_regularity(config):
    dynamodb_table = config.get('dynamodb_table')
    dynamodb_region = config.get('dynamodb_region', 'us-east-1')
    if dynamodb_table:
        print(f"\n--- DynamoDB {dynamodb_table} datetime_utc regularity ---")
        try:
            dynamodb = boto3.resource('dynamodb', region_name=dynamodb_region)
            table = dynamodb.Table(dynamodb_table)
            # Scan table for all fields (no ProjectionExpression)
            response = table.scan()
            items = response.get('Items', [])
            while 'LastEvaluatedKey' in response:
                response = table.scan(
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                items.extend(response.get('Items', []))
            # Convert to DataFrame
            df_dynamo = pd.DataFrame(items)
            if df_dynamo.empty or 'datetime_utc' not in df_dynamo.columns:
                print("datetime_utc column not found or no data in DynamoDB table!")
            else:
                # DynamoDB stores as string, so parse
                df_dynamo['datetime_utc'] = pd.to_datetime(df_dynamo['datetime_utc'], errors='coerce')
                print_datetime_regularity(df_dynamo, f"DynamoDB {dynamodb_table}")
        except Exception as e:
            print(f"Error reading from DynamoDB table {dynamodb_table}: {e}")


def main():
    # Check regularity of datetime_utc in multiple tables in sqlite
    # db_path = config.get('db_path')
    # sqlite_tables = [
    #     ("underway_summary", "SQLite underway_summary"),
    #     ("rhodamine", "SQLite rhodamine"),
    #     ("ph", "SQLite ph"),
    #     ("tsg", "SQLite tsg"),
    #     ("gps", "SQLite gps"),
    # ]
    # with sqlite3.connect(db_path) as conn:
    #     for table, label in sqlite_tables:
    #         try:
    #             #df_table = pd.read_sql_query(f"SELECT datetime_utc FROM {table} ORDER BY datetime_utc", conn)
    #             df_table = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    #         except Exception as e:
    #             print(f"Error reading from SQLite table {table}: {e}")
    #             df_table = pd.DataFrame()
    #         if df_table.empty or 'datetime_utc' not in df_table.columns:
    #             print(f"datetime_utc column not found or no data in {table}!")
    #         else:
    #             print_datetime_regularity(df_table, label)

    if 'df' in locals():
        print_datetime_regularity(df, "Parquet")
    


if __name__ == "__main__":
    main()