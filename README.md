# Parcel Filter

This is a parcel filtering tool that processes parcel data and generates reports with spatial analysis. Full documentation can be found [here](https://docs.google.com/document/d/1VWgwy9jv2th4EPQpQOOINEckYdfVo1fxKOvp2Kb1hOs/edit?tab=t.0).

## Setup

### Windows Setup Instructions

1. Install Python 3.9 or later from the [official Python website](https://www.python.org/downloads/windows/)
   - During installation, make sure to check "Add Python to PATH"
   - Also check "Install pip"

2. Install Git from [git-scm.com](https://git-scm.com/download/win)
   - Use the default installation options

3. Open Command Prompt as Administrator:
   - Press Windows + X
   - Select "Windows PowerShell (Admin)" or "Command Prompt (Admin)"

4. Clone the repository and set up the environment:
```bash
# Clone the repository
git clone https://github.com/sailbot24/IDTool.git
cd IDTool

# (Optional) Set up the environment manually
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

5. Prepare the data directory:
   - Create a `data` folder in the project directory
   - Place your DuckDB database file (`LandAI.ddb`) in the `data` folder

### Mac/Linux Setup

```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
```

## Environment Management

- **Automatic:** The tool checks for a Python virtual environment (`venv/`) on every run. If it does not exist, it will be created automatically (fast, minimal overhead).
- **On Demand:** You can force a full environment setup (including requirements installation) with:
  ```bash
  python main.py --setup-env
  ```
  This will set up or repair the environment and then exit.

## Updating the Tool

- To check for and install updates from the GitHub repository, use:
  ```bash
  python main.py --update
  ```
- The update process only affects the codebase. **Your `data/` and `results/` folders are protected and never touched by updates.**
- These folders are excluded from version control by `.gitignore`.

## Usage

```bash
# Basic usage
python main.py --state az --county maricopa --min-size 50

# With Google Sheets URL
python main.py --state az --county maricopa --min-size 50 --ranking-url "https://docs.google.com/spreadsheets/d/your-sheet-id/edit"

# With custom output directory
python main.py --state az --county maricopa --min-size 50 --output ./custom_output

# Force environment setup
python main.py --setup-env

# Check for and install updates
python main.py --update
```

### Arguments

- `--state`: Two-letter state code (e.g., az)
- `--county`: County name (e.g., maricopa)
- `--min-size`: Minimum parcel size in acres (default: 50)
- `--data-dir`: Base directory containing data (default: ./data)
- `--output`: Directory for output files (default: ./results)
- `--ranking-url`: Google Sheets URL for ranking data (optional)
- `--force`: Force a full run, ignoring any checkpoints
- `--quick-view`: Generate and display an interactive map view of results
- `--update`: Check for and install updates
- `--setup-env`: Set up or repair the Python environment and exit

## Features

- **DuckDB Integration**: Uses DuckDB for efficient spatial data processing
- **Google Sheets Integration**: Can output results to Google Sheets
- **Timestamped Outputs**: Automatically creates timestamped directories for outputs
- **Spatial Analysis**: Performs spatial operations between parcels and airports
- **Activity Filtering**: Filters parcels based on LBCS activity descriptions
- **Size Filtering**: Filters parcels based on minimum size requirements
- **Safe Data Handling**: The `data/` and `results/` folders are never touched by updates or version control

## Data Structure

The project uses DuckDB to store and process all data. The database contains the following tables:

```
duckdb_database/
├── parcels
│   └── {state}_{county}  # Parcel data for specific state and county
├── Other GIS 
    └── iso_50 # The isochrone table for the drive times. This is going to be changed in the future. 
    └── civil_airport # This is the civil airport table but only contains airport boundaries rather than airport runways. 
├── Rextag
    └──ALL data from the Rextag GDB. # Will list out later 
    └──electrictransmission is the name of the electric transmission lines 

```

All spatial operations are performed within DuckDB using the spatial extension. The data is loaded into DuckDB tables and processed in-memory for efficient spatial analysis.

## Output Structure

When run, the script creates a timestamped directory with the following structure:

```
results/run_YYYYMMDD_HHMMSS/
├── filtered_parcels.parquet
├── parcels_within_5_miles.parquet
└── parcels_within_10_miles.parquet
```

## Notes

- The parcel data should include a `gisacre` column for size filtering
- The parcel data should include a `lbcs_activity_desc` column for activity filtering
- Both datasets should include a `geometry` column in WKB format for spatial operations
- For Google Sheets integration, ensure you have the necessary credentials set up

## Versioning & Data Safety

- The tool uses Git for version control and update management.
- The `data/` and `results/` folders are excluded from version control via `.gitignore` and are never modified by updates.
- Only code and configuration files are updated when using the `--update` flag.
- This ensures your input data and results are always safe.

## DuckDB Usage

The project uses DuckDB for efficient spatial data processing. Here are some important notes about DuckDB usage:

### Configuration
- DuckDB is configured with a 4GB memory limit by default
- The spatial extension is automatically loaded for geospatial operations
- Checkpoints are stored in a `.checkpoints` directory for intermediate results

### Spatial Operations
- DuckDB's spatial extension is used for all spatial operations
- Geometry columns are automatically handled by the spatial extension
- Spatial joins and distance calculations are performed using DuckDB's native functions

### Performance Considerations
- Large datasets are processed in memory-efficient chunks
- Spatial operations are optimized using DuckDB's spatial indexing
- Checkpoints are used to save intermediate results and prevent data loss

### Data Types
- Geometry columns are stored in WKB format
- Spatial operations preserve the original coordinate reference system (CRS)
- DuckDB automatically handles geometry type conversions when needed

### Error Handling
- DuckDB connection errors are caught and logged
- Spatial operation errors are handled gracefully with appropriate error messages
- Checkpoint operations include error recovery mechanisms
- Checkpoint operations include error recovery mechanisms