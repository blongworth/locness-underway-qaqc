# LOCNESS underway data reduction

scripts and pipelines for data reduction and quality control
of underway data from the LOC-02 cruise

## Method

main.py has most functions for the data pipeline and the main data pipeline. Steps are to process individual data streams (GPS, TSG, pH, rhodamine), filling gaps where possible from shipboard GPS and TSG streams. Quality control is performed on each data stream, with the following flags: 

- Null: No data
- 2: Good data
- 3: Data is suspect
- 4: Data is bad

Data are resampled to 2s resolution by binning and averaging where needed.
Final data are merged into a single dataframe and output as csv, parquet, and netCDF.

## Files

- main.py: main data processing pipeline
- patch_gps.py: functions to patch GPS data gaps
- tsg_parser.py: functions to parse and repair TSG data
- resampler.py: functions to resample data to common time base

## Usage

`uv run main.py` will execute the main data processing pipeline, and should set up the environment as needed.

If running outside of uv, ensure the required packages are installed in your Python environment using the pyproject.toml file.