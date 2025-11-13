#!/usr/bin/env python3
"""
Test script to compare efficiency of reading NetCDF vs Parquet files.
Compares xarray (NetCDF) vs polars (Parquet) for the same dataset.
"""

import time
import xarray as xr
import polars as pl
from pathlib import Path
import psutil
import os


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def time_netcdf_read(file_path: str, iterations: int = 5):
    """Time reading NetCDF file with xarray."""
    times = []
    memory_before = get_memory_usage()
    
    for i in range(iterations):
        start_time = time.perf_counter()
        ds = xr.open_dataset(file_path)
        # Force loading of data into memory
        ds.load()
        end_time = time.perf_counter()
        times.append(end_time - start_time)
        ds.close()
    
    memory_after = get_memory_usage()
    
    return {
        'format': 'NetCDF (xarray)',
        'mean_time': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times),
        'memory_delta': memory_after - memory_before,
        'iterations': iterations
    }


def time_parquet_read(file_path: str, iterations: int = 5):
    """Time reading Parquet file with polars."""
    times = []
    memory_before = get_memory_usage()
    
    for i in range(iterations):
        start_time = time.perf_counter()
        df = pl.read_parquet(file_path)
        # Force evaluation by accessing the data
        _ = len(df)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    memory_after = get_memory_usage()
    
    return {
        'format': 'Parquet (polars)',
        'mean_time': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times),
        'memory_delta': memory_after - memory_before,
        'iterations': iterations
    }


def get_file_info(file_path: str):
    """Get file size information."""
    if Path(file_path).exists():
        size_mb = Path(file_path).stat().st_size / 1024 / 1024
        return size_mb
    return None


def print_results(results_list):
    """Print formatted results."""
    print("\n" + "="*60)
    print("FILE FORMAT COMPARISON RESULTS")
    print("="*60)
    
    for results in results_list:
        print(f"\nFormat: {results['format']}")
        print(f"Mean read time: {results['mean_time']:.4f} seconds")
        print(f"Min read time:  {results['min_time']:.4f} seconds")
        print(f"Max read time:  {results['max_time']:.4f} seconds")
        print(f"Memory delta:   {results['memory_delta']:.2f} MB")
        print(f"Iterations:     {results['iterations']}")
    
    # Compare speeds
    if len(results_list) == 2:
        netcdf_time = next(r['mean_time'] for r in results_list if 'NetCDF' in r['format'])
        parquet_time = next(r['mean_time'] for r in results_list if 'Parquet' in r['format'])
        
        if netcdf_time < parquet_time:
            speedup = parquet_time / netcdf_time
            print(f"\nNetCDF is {speedup:.2f}x faster than Parquet")
        else:
            speedup = netcdf_time / parquet_time
            print(f"\nParquet is {speedup:.2f}x faster than NetCDF")


def main():
    """Main function to run the comparison test."""
    # Define file paths
    netcdf_path = "output/loc02_uw_qc.nc"
    parquet_path = "output/loc02_uw_qc.parquet"
    
    # Check if files exist
    if not Path(netcdf_path).exists():
        print(f"Error: NetCDF file not found: {netcdf_path}")
        print("Run the main processing script first to generate the files.")
        return
    
    if not Path(parquet_path).exists():
        print(f"Error: Parquet file not found: {parquet_path}")
        print("Run the main processing script first to generate the files.")
        return
    
    # Get file sizes
    netcdf_size = get_file_info(netcdf_path)
    parquet_size = get_file_info(parquet_path)
    
    print("FILE SIZE COMPARISON")
    print("-" * 40)
    print(f"NetCDF file size:  {netcdf_size:.2f} MB")
    print(f"Parquet file size: {parquet_size:.2f} MB")
    print(f"Size ratio (NetCDF/Parquet): {netcdf_size/parquet_size:.2f}")
    
    # Run timing tests
    print("\nRunning read performance tests...")
    print("This may take a moment...")
    
    iterations = 10  # Number of read operations to average
    
    netcdf_results = time_netcdf_read(netcdf_path, iterations)
    parquet_results = time_parquet_read(parquet_path, iterations)
    
    # Print results
    print_results([netcdf_results, parquet_results])
    
    # Additional info
    print("\nFile paths tested:")
    print(f"  NetCDF:  {netcdf_path}")
    print(f"  Parquet: {parquet_path}")


if __name__ == "__main__":
    main()