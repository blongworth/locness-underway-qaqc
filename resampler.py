"""
resampler for raw sensor data.
"""

import logging
from isfetphcalc import calc_ph
import polars as pl

def resample_polars_dfs(dfs: dict[str, pl.DataFrame], interval: str) -> pl.DataFrame:
    """Resample multiple Polars DataFrames to a regular time grid.
     Avoids missing misaligned samples by binning data into time intervals.

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
    
    # Get time range
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
    seconds = int(interval[:-1])
    
    # Concatenate all data
    combined = pl.concat(valid_dfs.values(), how='diagonal')
    
    # Bin each timestamp to its interval boundary
    combined = combined.with_columns(
        (pl.col('datetime_utc').dt.truncate(f'{seconds}s')).alias('time_bin')
    )
    
    # Aggregate by time bin
    # For flag columns, take the first value; for others, use mean
    numeric_cols = combined.select(pl.col(pl.Float64, pl.Int8, pl.Int32, pl.Int64)).columns
    
    agg_exprs = []
    for col in numeric_cols:
        if 'flag' in col.lower():
            # Take first value for flag columns
            agg_exprs.append(pl.col(col).max().alias(col))
        else:
            # Use mean for numeric columns
            agg_exprs.append(pl.col(col).mean().alias(col))
    
    result = combined.group_by('time_bin').agg(agg_exprs).rename({'time_bin': 'datetime_utc'})
    
    # Create full time grid and left join
    # Truncate start_time to the nearest interval boundary
    start_truncated = start_time.replace(
        microsecond=0,
        second=(start_time.second // seconds) * seconds
    )
    
    time_grid = pl.DataFrame({
        'datetime_utc': pl.datetime_range(
            start_truncated,
            end_time,
            interval=f'{seconds}s',
            eager=True
        )
    })
    
    result = time_grid.join(result, on='datetime_utc', how='left')
    
    return result.sort('datetime_utc')


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