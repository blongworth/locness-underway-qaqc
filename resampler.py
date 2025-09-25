"""
Persistent resampler for raw sensor data.

This module provides a stateful resampler that:
1. Tracks which raw data has been processed to avoid duplicates
2. Maintains pH history for moving average calculations
3. Efficiently handles incremental resampling operations
"""

import sqlite3
import pandas as pd
import logging
from typing import Optional, Dict
from isfetphcalc import calc_ph
import polars as pl

def resample_polars_dfs(dfs: dict[str, pl.DataFrame], interval: str) -> pl.DataFrame:
    """Resample multiple Polars DataFrames to a regular time grid.
    
    Args:
        dfs: Dict of DataFrames with datetime_utc column
        interval: Resampling interval (e.g. '2s')
        
    Returns:
        Combined DataFrame resampled to regular time grid
    """
    # Handle empty input cases
    valid_dfs = {name: df for name, df in dfs.items() if not df.is_empty()}
    if not valid_dfs:
        return pl.DataFrame()
    
    # Get time range from all dataframes efficiently
    time_bounds = pl.concat([
        df.select(
            pl.col('datetime_utc').min().alias('min'),
            pl.col('datetime_utc').max().alias('max')
        ) for df in valid_dfs.values()
    ]).select(
        pl.col('min').min().alias('start'),
        pl.col('max').max().alias('end')
    )
    
    if time_bounds.is_empty():
        return pl.DataFrame()
    
    start_time, end_time = time_bounds.row(0)
    seconds = int(interval[:-1])  # Extract seconds from interval string
    
    # Create time grid using Polars datetime_range
    result = pl.DataFrame({
        'datetime_utc': pl.datetime_range(
            start_time,
            end_time,
            interval=f'{seconds}s',
            eager=True
        )
    })
    
    # Process each dataframe
    for df in valid_dfs.values():
        # Sort both dataframes by datetime
        df = df.sort('datetime_utc')
        result = result.sort('datetime_utc')
        
        # Perform asof join with 2s tolerance
        result = result.join_asof(
            df, 
            on='datetime_utc',
            strategy='backward',
            tolerance='4s'
        )
    
    return result

def resample_and_join(raw_data: Dict[str, pd.DataFrame], resample_interval: str) -> pd.DataFrame:
    """
    Resample raw data tables using mean aggregation and join into a single DataFrame.
    
    Uses mean aggregation for numeric columns (pandas automatically drops NaN values).
    For non-numeric columns, uses the first value in each time bin.
    
    Args:
        raw_data: Dict mapping table names to DataFrames
        resample_interval: Time interval for resampling (e.g., '2S' for 2 seconds)
        
    Returns:
        Resampled and joined DataFrame
    """
    expected_cols = ['datetime_utc', 'latitude', 'longitude', 'rho_ppb', 
                    'ph_total', 'vrse', 'temp', 'salinity']
    
    # Check if all DataFrames are empty
    if all(df.empty for df in raw_data.values()):
        return pd.DataFrame(columns=expected_cols)
    
    # First, find the overall time range and create a proper time grid
    all_timestamps = []
    for df in raw_data.values():
        if not df.empty and 'datetime_utc' in df.columns:
            all_timestamps.extend(df['datetime_utc'].tolist())
    
    if not all_timestamps:
        return pd.DataFrame(columns=expected_cols)
    
    # Create a regular time grid based on the resampling interval
    min_time = min(all_timestamps)
    max_time = max(all_timestamps)
    
    # Round min_time down to nearest interval boundary
    resample_freq = pd.Timedelta(resample_interval)
    min_time_rounded = min_time.floor(resample_freq)
    
    # Create time grid
    time_grid = pd.date_range(
        start=min_time_rounded,
        end=max_time + resample_freq,
        freq=resample_interval
    )
    
    # Create base DataFrame with the time grid
    result = pd.DataFrame({'datetime_utc': time_grid})
    result = result.set_index('datetime_utc')
    
    # Resample each table to the time grid using mean aggregation
    for table, df in raw_data.items():
        if df.empty or 'datetime_utc' not in df.columns:
            continue
            
        # Prepare DataFrame for resampling
        df_prep = df.copy()
        df_prep = df_prep.drop_duplicates(subset='datetime_utc')
        df_prep = df_prep.set_index('datetime_utc')
        
        # Get numeric columns for mean aggregation
        numeric_cols = df_prep.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) > 0:
            # Use mean aggregation for numeric columns (pandas already drops NaN by default)
            df_resampled = df_prep[numeric_cols].resample(resample_interval).mean()
            
            # For non-numeric columns, use first value (or could use mode)
            non_numeric_cols = df_prep.select_dtypes(exclude=['number']).columns
            if len(non_numeric_cols) > 0:
                df_non_numeric = df_prep[non_numeric_cols].resample(resample_interval).first()
                df_resampled = df_resampled.join(df_non_numeric)
        else:
            # No numeric columns, just use first value for all
            df_resampled = df_prep.resample(resample_interval).first()
        
        # Reindex to match our time grid
        df_resampled = df_resampled.reindex(time_grid)
        
        # Join to result
        result = result.join(df_resampled, how='left')
    
    # Reset index and filter to only times where we have some data
    result = result.reset_index()
    
    # Remove rows that are completely empty (no sensor data)
    data_cols = [col for col in result.columns if col != 'datetime_utc']
    result = result.dropna(subset=data_cols, how='all')
    
    # Ensure all expected columns exist
    for col in expected_cols:
        if col not in result.columns:
            result[col] = pd.NA
    
    return result[expected_cols]

def add_corrected_ph(df: pl.DataFrame, ph_k0: float, ph_k2: float) -> pl.DataFrame:
    """Add corrected pH column to Polars DataFrame using the vectorized isfetphcalc.calc_ph."""
    required_cols = {'temperature', 'salinity', 'vrse'}
    if not required_cols.issubset(df.columns):
        return df.with_columns(pl.lit(None, dtype=pl.Float64).alias('ph_corrected'))

    # Create a mask for rows with valid data
    mask = (
        pl.col('vrse').is_not_null() &
        pl.col('temperature').is_not_null() &
        pl.col('salinity').is_not_null()
    )

    # Filter the DataFrame to only include rows that can be processed
    valid_df = df.filter(mask)

    if valid_df.is_empty():
        return df.with_columns(pl.lit(None, dtype=pl.Float64).alias('ph_corrected'))

    # Convert Polars Series to NumPy arrays for the vectorized calculation
    vrse_np = valid_df['vrse'].to_numpy()
    temp_np = valid_df['temperature'].to_numpy()
    salt_np = valid_df['salinity'].to_numpy()

    try:
        # Call the vectorized function from the library
        _, ph_total_np = calc_ph(
            Vrs=vrse_np,
            Press=0,  # Assuming pressure is 0 as in the original row-wise implementation
            Temp=temp_np,
            Salt=salt_np,
            k0=ph_k0,
            k2=ph_k2,
            Pcoefs=0
        )

        # Create a new DataFrame with the original index and the calculated pH
        ph_results = valid_df.select(pl.col('datetime_utc')).with_columns(
            pl.Series(name='ph_corrected', values=ph_total_np)
        )

        # Join the results back to the original DataFrame
        return df.join(ph_results, on='datetime_utc', how='left')

    except Exception as e:
        logging.error(f"Error during vectorized pH calculation: {e}")
        # In case of an error, return the original DataFrame with a null column
        return df.with_columns(pl.lit(None, dtype=pl.Float64).alias('ph_corrected'))