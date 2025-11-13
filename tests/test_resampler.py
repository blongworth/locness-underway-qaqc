import polars as pl
from datetime import datetime, timedelta
from resampler import resample_polars_dfs

def test_resample_polars_dfs():
    # Create sample data with overlapping time periods
    base_time = datetime(2025, 9, 25, 12, 0, 0)
    
    # First DataFrame with temperature data
    temp_times = [base_time + timedelta(seconds=i) for i in range(0, 10, 2)]
    temp_df = pl.DataFrame({
        'datetime_utc': temp_times,
        'temperature': [20.0, 20.5, 21.0, 21.5, 22.0]
    })
    
    # Second DataFrame with salinity data (slightly offset times)
    sal_times = [base_time + timedelta(seconds=i) for i in range(1, 11, 2)]
    sal_df = pl.DataFrame({
        'datetime_utc': sal_times,
        'salinity': [35.0, 35.1, 35.2, 35.3, 35.4]
    })
    
    # Create input dictionary
    dfs = {
        'temperature': temp_df,
        'salinity': sal_df
    }
    
    # Test resampling to 2-second intervals
    result = resample_polars_dfs(dfs, '2s')
    
    # Print the result for debugging
    print("\nResult DataFrame:")
    print(result)
    
    # Verify the result
    assert not result.is_empty(), "Result should not be empty"
    assert 'temperature' in result.columns, "Temperature column should be present"
    assert 'salinity' in result.columns, "Salinity column should be present"
    
    # Verify number of rows (should be 5 rows for 10 seconds with 2s intervals)
    assert len(result) == 5, f"Expected 5 rows, got {len(result)}"

if __name__ == "__main__":
    test_resample_polars_dfs()