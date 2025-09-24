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

def add_corrected_ph(df: pd.DataFrame, ph_k0: float, ph_k2: float) -> pd.DataFrame:
    """Add corrected pH column to DataFrame."""
    if 'temp' not in df.columns or 'salinity' not in df.columns or 'vrse' not in df.columns:
        df['ph_corrected'] = pd.NA
        return df
    
    # Create working copy to avoid modifying original data
    df_work = df.copy()
    
    # Calculate corrected pH using filled values, but only where vrse, temp, and salinity are available
    mask = df_work['vrse'].isna() | df_work['temp'].isna() | df_work['salinity'].isna()
    
    try:
        ph_free, ph_total = calc_ph(
            Vrs=df_work['vrse'],
            Press=0,
            Temp=df_work['temp'],
            Salt=df_work['salinity'],
            k0=ph_k0,
            k2=ph_k2,
            Pcoefs=0
        )
        df['ph_corrected'] = ph_total
    except Exception as e:
        logging.warning(f"Error calculating corrected pH: {e}")
        df['ph_corrected'] = pd.NA
    
    # Set NaN for rows with missing vrse data
    if mask.any():
        df.loc[mask, 'ph_corrected'] = pd.NA
    
    return df

def process_new_data(self) -> pd.DataFrame:
    """
    Process new raw data and return resampled results.
    
    This is the main method that:
    2. Resamples and joins the data
    3. Adds corrected pH
    4. Adds moving averages
    
    Returns:
        DataFrame with processed new data
    """
    # Resample and join
    df = resample_and_join(raw_data)

    if df.empty:
        logging.info("No data after resampling")
        return df
    
    # Add corrected pH
    df = add_corrected_ph(df, ph_k0=0.0, ph_k2=0.0)  # Replace with actual calibration constants
    
    logging.info(f"Generated {len(df)} resampled records")
    
    return df
    