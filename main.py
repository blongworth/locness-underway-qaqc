# Main processing script for underway data quality control
"""
This script processes and combines underway oceanographic data using Prefect for workflow management.
The workflow handles GPS, TSG (thermosalinograph), pH, and rhodamine measurements.

Dependencies:
    - prefect: Workflow management
    - polars: Data processing
    - tomli: Configuration management
"""

# TODO: store each underway product with qc flags
# TODO: qc and source flags
# TODO: qc flag for each parameter in final output

from pathlib import Path
from typing import Optional
import tomli
import logging
from prefect import task, flow
from patch_gps import main as patch_gps
from tsg_parser import main as patch_tsg
from resampler import resample_polars_dfs, add_corrected_ph
from loc_02_apply_ph_qc import read_ph_flags, apply_ph_flags
import sqlite3
import polars as pl
import xarray as xr
import pandas as pd

# Prevent truncation of polars output
pl.Config.set_tbl_cols(-1)

@task
def load_config(config_path: str = "config.toml") -> dict:
    """Load configuration from TOML file as a Prefect task.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        dict: Configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        tomli.TOMLDecodeError: If config file is invalid
    """
    with open(config_path, "rb") as f:
        return tomli.load(f)

def validate_gps_data(df: pl.DataFrame) -> bool:
    """Validate GPS data format and values.
    
    Performs checks on GPS data to ensure it meets quality requirements:
    - Has all required columns (datetime_utc, latitude, longitude)
    - Contains valid numeric data types
    - Values are within reasonable geographic ranges
    
    Args:
        df: DataFrame to validate, should contain GPS position data
        
    Returns:
        bool: True if data meets all validation criteria, False otherwise
        
    Note:
        - Latitude range: -90° to +90°
        - Longitude range: -180° to +180°
        - Logs warnings for out-of-range values but only returns False for critical errors
    """
    # Check required columns
    required_cols = ["datetime_utc", "latitude", "longitude"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing required GPS columns: {missing_cols}")
        return False
        
    # Check data types
    try:
        df = df.with_columns([
            pl.col("latitude").cast(pl.Float64),
            pl.col("longitude").cast(pl.Float64)
        ])
    except Exception as e:
        logging.error(f"Invalid data types in GPS data: {str(e)}")
        return False
        
    # Check value ranges
    lat_range = (-90, 90)
    lon_range = (-180, 180)
    
    lat_invalid = df.filter(
        (pl.col("latitude") < lat_range[0]) | 
        (pl.col("latitude") > lat_range[1])
    )
    if len(lat_invalid) > 0:
        logging.warning(f"Found {len(lat_invalid)} latitude values outside range {lat_range}")
        
    lon_invalid = df.filter(
        (pl.col("longitude") < lon_range[0]) | 
        (pl.col("longitude") > lon_range[1])
    )
    if len(lon_invalid) > 0:
        logging.warning(f"Found {len(lon_invalid)} longitude values outside range {lon_range}")
    
    return True

@task
def process_gps() -> Optional[pl.DataFrame]:
    """Process and patch GPS data as a Prefect task.
    
    Returns:
        Optional[pl.DataFrame]: Processed GPS DataFrame
        
    Raises:
        ValueError: If data validation fails
    """
    gps_df = patch_gps()
    
    # Validate data
    if not validate_gps_data(gps_df):
        raise ValueError("GPS data validation failed")
            
    return gps_df

def validate_tsg_data(df: pl.DataFrame) -> bool:
    """Validate TSG (thermosalinograph) data format and values.
    
    Performs checks on TSG data to ensure it meets quality requirements:
    - Has all required columns (datetime_utc, temperature, salinity)
    - Contains valid numeric data types
    - Values are within reasonable oceanographic ranges
    
    Args:
        df: DataFrame to validate, should contain temperature and salinity data
        
    Returns:
        bool: True if data meets all validation criteria, False otherwise
        
    Note:
        - Temperature range: -2°C to +35°C (typical ocean temperatures)
        - Salinity range: 0 to 40 PSU (practical salinity units)
        - Logs warnings for out-of-range values but only returns False for critical errors
        
    Example ranges are based on typical ocean conditions - adjust as needed for specific
    deployment regions or requirements.
    """
    # Check required columns
    required_cols = ["datetime_utc", "temperature", "salinity"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing required TSG columns: {missing_cols}")
        return False
        
    # Check data types
    try:
        df = df.with_columns([
            pl.col("temperature").cast(pl.Float64),
            pl.col("salinity").cast(pl.Float64)
        ])
    except Exception as e:
        logging.error(f"Invalid data types in TSG data: {str(e)}")
        return False
        
    # Check value ranges
    temp_range = (-2, 35)  # Reasonable ocean temperature range in °C
    salt_range = (0, 40)   # Reasonable ocean salinity range in PSU
    
    temp_invalid = df.filter(
        (pl.col("temperature") < temp_range[0]) | 
        (pl.col("temperature") > temp_range[1])
    )
    if len(temp_invalid) > 0:
        logging.warning(f"Found {len(temp_invalid)} temperature values outside range {temp_range}")
        
    salt_invalid = df.filter(
        (pl.col("salinity") < salt_range[0]) | 
        (pl.col("salinity") > salt_range[1])
    )
    if len(salt_invalid) > 0:
        logging.warning(f"Found {len(salt_invalid)} salinity values outside range {salt_range}")
    
    return True

@task
def process_tsg(config: dict) -> pl.DataFrame:
    """Process TSG (thermosalinograph) data as a Prefect task.
    
    Args:
        config: Configuration dictionary containing file paths
        
    Returns:
        pl.DataFrame: Processed TSG DataFrame
        
    Raises:
        FileNotFoundError: If TSG file not found
        ValueError: If data validation fails
    """
    #tsg_file = config["paths"].get("tsg_file", "data/underway.tsg")
    
    # if not Path(tsg_file).exists():
    #     raise FileNotFoundError(f"TSG file not found: {tsg_file}")

    tsg_df = patch_tsg()

    if tsg_df is None or len(tsg_df) == 0:
        raise ValueError("No valid TSG data found")
    
    # Validate data
    if not validate_tsg_data(tsg_df):
        raise ValueError("TSG data validation failed")
            
    return tsg_df

def validate_ph_data(df: pl.DataFrame) -> bool:
    """Validate pH sensor data format and values.
    
    Performs checks on pH sensor data to ensure it meets quality requirements:
    - Has all required columns (datetime_utc, vrse, ph_flag)
    - Contains valid numeric data types
    - Voltage readings (Vrse) are within typical ISFET sensor ranges
    - pH quality flags are valid integers within defined range
    
    Args:
        df: DataFrame to validate, should contain ISFET pH sensor data
        
    Returns:
        bool: True if data meets all validation criteria, False otherwise
        
    Note:
        - Vrse range: -1V to +1V (typical ISFET voltage range)
        - pH flags: 0-4 (representing different quality levels)
        - Returns False for invalid flags as they are critical for data quality
        - Logs warnings for out-of-range voltage values
        
    The voltage ranges and flag definitions should match the specific ISFET
    sensor configuration and quality control scheme being used.
    """
    # Check required columns
    required_cols = ["datetime_utc", "vrse", "ph_flag"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing required pH columns: {missing_cols}")
        return False
        
    # Check data types
    try:
        df = df.with_columns([
            pl.col("vrse").cast(pl.Float64),
            pl.col("ph_flag").cast(pl.Int64)
        ])
    except Exception as e:
        logging.error(f"Invalid data types in pH data: {str(e)}")
        return False
        
    # Check value ranges for Vrse (typical range for ISFET pH sensors)
    vrse_range = (-1, 1)  # Typical voltage range in V
    
    vrse_invalid = df.filter(
        (pl.col("vrse") < vrse_range[0]) | 
        (pl.col("vrse") > vrse_range[1])
    )
    if len(vrse_invalid) > 0:
        logging.warning(f"Found {len(vrse_invalid)} Vrse values outside range {vrse_range}")
        
    # Check pH flags are valid (assuming flags are 0-4)
    flag_invalid = df.filter(
        (pl.col("ph_flag") < 0) | 
        (pl.col("ph_flag") > 4)
    )
    if len(flag_invalid) > 0:
        logging.error(f"Found {len(flag_invalid)} invalid pH flag values")
        return False
    
    return True

@task
def read_ph_and_flag(db_path: str, flag_path: str) -> pl.DataFrame:
    """Read pH data from SQLite database as a Prefect task.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        pl.DataFrame: pH DataFrame
        
    Raises:
        FileNotFoundError: If database file not found
        ValueError: If data validation fails
    """
    # read ph data
    with sqlite3.connect(db_path) as conn:
        df = pl.read_database("SELECT * FROM ph", connection=conn)
        if "datetime_utc" in df.columns:
            df = df.with_columns(
            (pl.col("datetime_utc").cast(pl.Int64) * 1_000_000).cast(pl.Datetime("us"))
            )
    
    # Print first 10 rows
    print("\nFirst 10 rows of rhodamine data:")
    print(df.head(10))

    # Print dataframe schema and summary
    print("\nDataframe Schema:")
    print(df.schema)

    print("\nDataframe Summary:")
    print(df.describe())
    # Validate data
    # if not validate_ph_data(df):
    #     raise ValueError("pH data validation failed")

    # read ph flags
    flag_path = "loc02-ph-qc-flags.csv"
    ph_flags = read_ph_flags(flag_path)
    df = apply_ph_flags(df, ph_flags)
    return df

def validate_rho_data(df: pl.DataFrame) -> bool:
    """Validate rhodamine data format and values.
    
    Checks for required columns, data types, and value ranges in rhodamine measurements.
    
    Args:
        df: DataFrame to validate, should contain rhodamine concentration data
        
    Returns:
        bool: True if data meets all validation criteria, False otherwise
        
    Note:
        Logs warnings for out-of-range values but only returns False for critical errors
    """
    # Check required columns
    required_cols = ["datetime_utc", "rho_ppb"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing required rhodamine columns: {missing_cols}")
        return False
        
    # Check data types
    try:
        df = df.with_columns([
            pl.col("rho_ppb").cast(pl.Float64)
        ])
    except Exception as e:
        logging.error(f"Invalid data types in rhodamine data: {str(e)}")
        return False
        
    # Check value ranges (typical rhodamine concentrations in ppb)
    rho_range = (-1, 500)  # Adjust range based on expected concentrations
    
    rho_invalid = df.filter(
        (pl.col("rho_ppb") < rho_range[0]) | 
        (pl.col("rho_ppb") > rho_range[1])
    )
    if len(rho_invalid) > 0:
        logging.warning(f"Found {len(rho_invalid)} rhodamine values outside range {rho_range}")
    
    return True

@task
def read_rho_table_to_polars(db_path: str) -> pl.DataFrame:
    """Read rhodamine data from SQLite database as a Prefect task.
    
    Args:
        db_path: Path to SQLite database containing rhodamine measurements
        
    Returns:
        pl.DataFrame: DataFrame with columns:
            - datetime_utc: Timestamp of measurement
            - rho_ppb: Rhodamine concentration in parts per billion
            
    Raises:
        FileNotFoundError: If database file not found
        ValueError: If data validation fails
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")
            
    # Open a SQLite connection and use it for reading the table
    with sqlite3.connect(db_path) as conn:
        df = pl.read_database("SELECT * FROM rhodamine", connection=conn)
        # Convert unix integer timestamp to polars datetime (assuming seconds since epoch)
        if "datetime_utc" in df.columns:
            df = df.with_columns(
            (pl.col("datetime_utc").cast(pl.Int64) * 1_000_000).cast(pl.Datetime("us"))
            )
    # Print first 10 rows
    print("\nFirst 10 rows of rhodamine data:")
    print(df.head(10))

    # Print dataframe schema and summary
    print("\nDataframe Schema:")
    print(df.schema)

    print("\nDataframe Summary:")
    print(df.describe())
    # Validate data
    if not validate_rho_data(df):
        raise ValueError("Rhodamine data validation failed")
    
    # Add a default rho_flag
    df = df.with_columns(
        pl.lit(2, dtype=pl.Int8).alias("rho_flag")
    )
    return df

def validate_oxygen_data(df: pl.DataFrame) -> bool:
    """Validate oxygen sensor data format and values.
    
    Performs checks on oxygen sensor data to ensure it meets quality requirements:
    - Has all required columns (datetime_utc, oxygen_umol_kg)
    - Contains valid numeric data types
    - Values are within reasonable dissolved oxygen ranges
    
    Args:
        df: DataFrame to validate, should contain dissolved oxygen data
        
    Returns:
        bool: True if data meets all validation criteria, False otherwise
        
    Note:
        - Oxygen range: 200-300 umol/kg (typical dissolved oxygen range in seawater)
        - Logs warnings for out-of-range values but only returns False for critical errors
    """
    # Check required columns
    required_cols = ["datetime_utc", "oxygen_umol_kg"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing required oxygen columns: {missing_cols}")
        return False
        
    # Check data types
    try:
        df = df.with_columns([
            pl.col("oxygen_umol_kg").cast(pl.Float64)
        ])
    except Exception as e:
        logging.error(f"Invalid data types in oxygen data: {str(e)}")
        return False

    # Check value ranges (typical dissolved oxygen concentrations in umol/kg)
    oxygen_range = (200, 300)  # Reasonable dissolved oxygen range in seawater

    oxygen_invalid = df.filter(
        (pl.col("oxygen_umol_kg") < oxygen_range[0]) | 
        (pl.col("oxygen_umol_kg") > oxygen_range[1])
    )
    if len(oxygen_invalid) > 0:
        logging.warning(f"Found {len(oxygen_invalid)} oxygen values outside range {oxygen_range}")
    
    return True

@task
def read_oxygen(csv_path: str) -> pl.DataFrame:
    """Read oxygen data from CSV file as a Prefect task.
    
    Args:
        csv_path: Path to CSV file containing oxygen measurements
        
    Returns:
        pl.DataFrame: DataFrame with columns:
            - datetime_utc: Timestamp of measurement
            - oxygen_mg_l: Dissolved oxygen concentration in mg/L
            - oxygen_flag: Quality flag for oxygen data
            
    Raises:
        FileNotFoundError: If CSV file not found
        ValueError: If data validation fails
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Oxygen CSV file not found: {csv_path}")
    
    # Read CSV file
    df = pl.read_csv(csv_path)
    # Rename datetime column to datetime_utc if it exists
    if "datetime" in df.columns and "datetime_utc" not in df.columns:
        df = df.rename({"datetime": "datetime_utc"})
        
    # Convert datetime column if needed (adjust format as necessary)
    if "datetime_utc" in df.columns:
        df = df.with_columns(
            pl.col("datetime_utc")
            .str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S.%f")
            .alias("datetime_utc")
        )
    
    # Print first 10 rows
    print("\nFirst 10 rows of oxygen data:")
    print(df.head(10))

    # Print dataframe schema and summary
    print("\nDataframe Schema:")
    print(df.schema)

    print("\nDataframe Summary:")
    print(df.describe())
    
    # Add a default oxygen_flag if not present
    if "oxygen_flag" not in df.columns:
        df = df.with_columns(
            pl.lit(2, dtype=pl.Int8).alias("oxygen_flag")
        )
    
    # keep only datetime_utc and O2_calc
    df = df.select([
        pl.col("datetime_utc"),
        pl.col("O2_calc").alias("oxygen_umol_kg"),
    ])

    # Validate data
    if not validate_oxygen_data(df):
        raise ValueError("Oxygen data validation failed")
    
    return df

   
@task
def read_hydrofia(xlsx_path: str) -> pl.DataFrame:
    """Read Hydrofia TA data from Excel file as a Prefect task.
    
    Args:
        xlsx_path: Path to Excel file containing Hydrofia TA measurements
        
    Returns:
        pl.DataFrame: DataFrame with columns:
            - timestamp: Timestamp of measurement
            - TA_corrected: Corrected total alkalinity
            - Hydrofia_Flag: Quality flag for Hydrofia data
            - TA_Discrete: Discrete TA measurements
            - TA_Discrete_Flag: Quality flag for discrete TA
            
    Raises:
        FileNotFoundError: If Excel file not found
        ValueError: If data validation fails
    """
    if not Path(xlsx_path).exists():
        raise FileNotFoundError(f"Hydrofia Excel file not found: {xlsx_path}")
    
    # Read Excel file
    df = pl.read_excel(xlsx_path)

    print(df.schema)

    # Convert "NaN" string values to actual None/null
    df = df.with_columns([
        pl.col(c).replace("NaN", None) for c in df.columns if df[c].dtype == pl.Utf8
    ])
    
    # Replace NaN float values with None/null
    df = df.with_columns(
        pl.col(pl.Float64).fill_nan(None)
    )
    
    # Select required columns
    required_cols = ["timestamp", "TA_corrected", "Hydrofia_Flag", "TA_Discrete", "TA_Discrete_Flag"]
    df = df.select(required_cols)
    df = df.rename({
        "timestamp": "datetime_utc",
        "TA_corrected": "ta_hydrofia",
        "Hydrofia_Flag": "ta_hydrofia_flag",
        "TA_Discrete": "ta_discrete",
        "TA_Discrete_Flag": "ta_discrete_flag"
    })
    # Filter where ta_hydrofia_flag is 2, 3, or 4
    df = df.filter(pl.col("ta_hydrofia_flag").is_in(["2", "3", "4"]))
    df = df.with_columns([
        pl.col("ta_hydrofia_flag").cast(pl.Int8),
        pl.col("ta_discrete_flag").cast(pl.Int8)
    ])
    
    # Convert timestamp to datetime if needed
    if "datetime_utc" in df.columns:
        df = df.with_columns(
            pl.col("datetime_utc").cast(pl.Datetime("us"))
        )
    
    # Print first 10 rows
    print("\nFirst 10 rows of Hydrofia data:")
    print(df.head(10))

    # Print dataframe schema and summary
    print("\nDataframe Schema:")
    print(df.schema)

    print("\nDataframe Summary:")
    print(df.describe())
    
    return df


@task
def combine_data(
    gps_df: pl.DataFrame,
    tsg_df: pl.DataFrame,
    ph_df: pl.DataFrame,
    rho_df: pl.DataFrame,
    oxygen_df: pl.DataFrame,
    hydrofia_df: pl.DataFrame,
    config: dict
) -> pl.DataFrame:
    """Combine all data sources into a single DataFrame.
    
    Args:
        gps_df: GPS data
        tsg_df: TSG data
        ph_df: pH data
        rho_df: Rhodamine data
        oxygen_df: Oxygen data
        hydrofia_df: Hydrofia data
        config: Configuration dictionary
        
    Returns:
        pl.DataFrame: Combined and resampled dataset
    """
    # Select needed columns and combine dfs to dictionary
    dfs = {
        "gps": gps_df.select(["datetime_utc", "latitude", "longitude"]),
        "tsg": tsg_df.select(["datetime_utc", "temperature", "temperature_flag", "salinity", "salinity_flag"]),
        "ph": ph_df.select(["datetime_utc", "vrse", "ph_flag"]),
        "rho": rho_df.select(["datetime_utc", "rho_ppb", "rho_flag"]),
        "oxygen": oxygen_df.select(["datetime_utc", "oxygen_umol_kg"]),         
        "hydrofia": hydrofia_df.select(["datetime_utc", "ta_hydrofia", "ta_hydrofia_flag", "ta_discrete", "ta_discrete_flag"]),
    }
    
    resample_interval = config["resample"].get("resample_interval")
    return resample_polars_dfs(dfs, interval=resample_interval)


@task
def add_ph_corrections(df: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Add corrected pH values to the dataset.
    
    Args:
        df: DataFrame with raw pH data
        config: Configuration dictionary with calibration constants
        
    Returns:
        pl.DataFrame: DataFrame with corrected pH values
    """
    ph_k0 = config["calibration"].get("ph_k0")
    ph_k2 = config["calibration"].get("ph_k2")
    df_cor =  add_corrected_ph(df, ph_k0=ph_k0, ph_k2=ph_k2)
    df_cor = df_cor.with_columns(
        pl.when(pl.col("ph_corrected").is_nan())
        .then(None)
        .otherwise(pl.col("ph_flag"))
        .alias("ph_flag")
    )
    return df_cor

@task
def add_underway_flow_flags(df: pl.DataFrame, flow_file: str) -> pl.DataFrame:
    """Add underway flow flags to the dataset based on shutdown periods.
    
    Args:
        df: DataFrame with datetime_utc column
        flow_file: Path to CSV file with shutdown periods
        
    Returns:
        pl.DataFrame: DataFrame with added 'underway_flow_flag' column
    """
    if not Path(flow_file).exists():
        raise FileNotFoundError(f"Underway flow flag file not found: {flow_file}")
    
    flow_flags = pl.read_csv(flow_file).with_columns([
        pl.col("start_time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
        pl.col("end_time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
    ])
    
    # Iterate over each shutdown period and update flags
    for row in flow_flags.iter_rows(named=True):
        start_time = row["start_time"]
        end_time = row["end_time"]
        
        # Create a mask for the time range of the shutdown
        mask = (df["datetime_utc"] > start_time) & (df["datetime_utc"] < end_time)
        
        # Set flags to 4 (bad data) for the shutdown period
        df = df.with_columns(
            pl.when(mask).then(4).otherwise(pl.col("temperature_flag")).alias("temperature_flag"),
            pl.when(mask).then(4).otherwise(pl.col("salinity_flag")).alias("salinity_flag"),
            pl.when(mask).then(4).otherwise(pl.col("ph_flag")).alias("ph_flag"),
            pl.when(mask).then(4).otherwise(pl.col("rho_flag")).alias("rho_flag"),
        )
    
    df = df.with_columns(
        pl.when(pl.col("ph_corrected").is_null())
        .then(None)
        .otherwise(pl.col("ph_flag"))
        .alias("ph_flag"),
        pl.when(pl.col("salinity").is_null())
        .then(None)
        .otherwise(pl.col("salinity_flag"))
        .alias("salinity_flag"),
        pl.when(pl.col("temperature").is_null())
        .then(None)
        .otherwise(pl.col("temperature_flag"))
        .alias("temperature_flag"),
        pl.when(pl.col("rho_ppb").is_null())
        .then(None)
        .otherwise(pl.col("rho_flag"))
        .alias("rho_flag"),
    )
    return df


@task
def save_outputs(df: pl.DataFrame, config: dict) -> None:
    """Save the processed data to configured output formats.
    
    Args:
        df: Processed DataFrame to save
        config: Configuration dictionary with output paths
    """

    # Print first 10 rows
    print("\nFirst 10 rows of combined data:")
    print(df.head(10))

    # Print dataframe schema and summary
    print("\nDataframe Schema:")
    print(df.schema)

    print("\nDataframe Summary:")
    print(df.describe())

    if "netcdf_path" in config["paths"]:

        # Convert Polars DataFrame to pandas
        pdf = df.to_pandas()

        # If there is a datetime_utc column, set it as the time index required by xarray
        if "datetime_utc" in pdf.columns:
            pdf["datetime_utc"] = pd.to_datetime(pdf["datetime_utc"])
            pdf = pdf.set_index("datetime_utc")
            pdf.index.name = "time"

        # Create xarray Dataset from the pandas DataFrame
        ds = xr.Dataset.from_dataframe(pdf)

        # write to netCDF
        ds.to_netcdf(config["paths"]["netcdf_path"])

    if "parquet_path" in config["paths"]:
        df.write_parquet(config["paths"]["parquet_path"])
    
    if "csv_path" in config["paths"]:
        df.write_csv(config["paths"]["csv_path"], datetime_format="%Y-%m-%dT%H:%M:%S")


@flow(name="LOCNESS Underway Data Processing")
def main():
    """Main Prefect flow for processing underway data."""
    # Load configuration
    config = load_config(config_path="config.toml")
    
    # Process all data sources in parallel
    gps_df = process_gps()
    tsg_df = process_tsg(config)
    ph_df = read_ph_and_flag(config["paths"]["db_path"], "loc02-ph-qc-flags.csv")
    rho_df = read_rho_table_to_polars(config["paths"]["db_path"])
    oxygen_df = read_oxygen("data/loc02_do_optode_qc.csv")
    hydrofia_df = read_hydrofia("data/LOC02_Hydrofia_Final_Flagged_103125.xlsx")
    
    # Combine data
    combined_df = combine_data(gps_df, tsg_df, ph_df, rho_df, oxygen_df, hydrofia_df, config)

    # Add pH corrections
    corrected_df = add_ph_corrections(combined_df, config)
    
    # Add underway flow flags
    corrected_df = add_underway_flow_flags(corrected_df, "loc02_uw_shutdowns.csv")
    
    # Save outputs
    save_outputs(corrected_df, config)

    gps_df.write_parquet("output/loc02_gps_data.parquet")
    tsg_df.write_parquet("output/loc02_tsg_data.parquet")
    ph_df.write_parquet("output/loc02_ph_data.parquet")
    rho_df.write_parquet("output/loc02_rho_data.parquet")

if __name__ == "__main__":
    main()
