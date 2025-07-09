# IDTool - 

## 🚀 Features

- **Multi-criteria Parcel Filtering**: Filter parcels by size, airport proximity, transmission line distance, and utility provider
- **Advanced Ranking System**: Rank parcels using weighted criteria including zoning, land activity, site characteristics, and flood risk
- **Interactive Mapping**: Generate interactive maps with Folium for visual analysis
- **PostgreSQL Integration**: Full PostGIS support for spatial data processing
- **Google Sheets Integration**: Dynamic ranking criteria loaded from Google Sheets
- **Automated Pipeline**: Complete end-to-end processing pipeline with logging
- **Utility Provider Support**: Filter and analyze parcels by specific utility providers
- **Drive Time Analysis**: Calculate drive times using isochrone data
- **Version Management**: Built-in update system and environment management
- **Interactive Selection**: Choose state, county, and power provider interactively
- **Parcel Filtering**: Filter parcels by various criteria
- **Parcel Ranking**: Rank parcels based on multiple factors
- **Database Rebuild**: Rebuild database from raw parcel files with standardized schema
- **Schema Compliance Reporting**: Detailed reports on how well your data matches the Regrid parcel schema

## 📋 Prerequisites

- Python 3.8 or higher
- PostgreSQL with PostGIS extension
- Git

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/sailbot24/IDTool.git
cd IDTool
```

### 2. Set Up Environment
The tool includes an automated environment setup:

```bash
# Set up the Python environment and install dependencies
python main.py --setup-env
```

Or manually:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Database
Create a `db_config.txt` file in the project root:

```txt
host=localhost
port=5432
database=LandAI
user=postgres
password=your_password
```

### 4. Prepare Data Directories
```bash
mkdir data results
```

## 🗄️ Database Setup

### Required PostgreSQL Extensions
```sql
CREATE EXTENSION postgis;
CREATE EXTENSION postgis_topology;
```

### Required Schemas and Tables
The tool expects the following structure:
- `parcels.{state}_{county}` - Parcel data with geometry (created by rebuild script)
- `other_gis.iso_50` - Isochrone data for drive time analysis
- `other_gis.us_civil_airports` - Airport data for filtering

### Database Rebuild Process

The database rebuild script loads raw parcel data from the `raw/parcels/` directory and standardizes the schema based on the Regrid parcel schema. This ensures all county tables have the same columns and enables seamless state-level analysis.

#### Directory Structure
```
raw/parcels/
├── co/
│   ├── adams/
│   │   ├── adams_parcels.shp
│   │   └── adams_parcels.csv
│   └── weld/
│       ├── weld_parcels.shp
│       └── weld_parcels.csv
└── az/
    ├── maricopa/
    │   └── maricopa_parcels.shp
    └── pima/
        └── pima_parcels.shp
```

#### Rebuild Commands
```bash
# Rebuild database for a specific state
python rebuild_database.py --state co

# Rebuild database for all states
python rebuild_database.py --all-states

# Use custom data directory
python rebuild_database.py --state co --data-dir /path/to/parcels
```

#### What the Rebuild Script Does
1. **Finds Parcel Files**: Scans the `raw/parcels/` directory for shapefiles, GeoJSON, GeoPackage, and CSV files
2. **Standardizes Schema**: Maps column names to the Regrid parcel schema and adds missing columns with NULL values
3. **Creates Tables**: Creates PostgreSQL tables with standardized schema for each county
4. **Loads Data**: Loads all parcel data into the database with proper geometry handling
5. **Creates Indexes**: Adds spatial indexes for efficient querying

#### Supported File Formats
- Shapefiles (.shp)
- GeoJSON (.geojson)
- GeoPackage (.gpkg)
- CSV files (.csv)

#### Column Standardization
The script automatically maps common column name variations to the standard Regrid schema:
- `parcel_number`, `parcelid` → `parcelnumb`
- `land_use`, `use_description` → `usedesc`
- `acres`, `acreage` → `gisacre`
- `geometry`, `shape` → `geom`
- And many more...

**Note:** Run the rebuild script whenever you add new parcel data or need to recreate the database from scratch.

## 📖 Usage

### Basic Usage
```bash
python main.py --state co --county adams
```

### Advanced Options
```bash
python main.py \
  --state co \
  --county adams \
  --min-size 50.0 \
  --transmission-distance 100.0 \
  --quick-view \
  --show-graph
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--state` | Two-letter state code (required) | - |
| `--county` | County name | All counties in state |
| `--db-config` | Database config file path | `db_config.txt` |
| `--min-size` | Minimum parcel size in acres | `50.0` |
| `--transmission-distance` | Max distance to transmission lines (meters) | `100.0` |
| `--provider` | Power utility provider to filter by | Interactive selection |
| `--ranking-url` | Google Sheets URL for ranking data | Default URL |
| `--quick-view` | Create interactive map | False |
| `--show-graph` | Show ranking distribution graph | False |
| `--update` | Check for and install updates | False |
| `--setup-env` | Set up Python environment | False |

## 🔧 Pipeline Process

The tool follows a 6-step pipeline:

1. **Load Parcels**: Load and filter parcels by minimum size and unwanted activities
2. **Apply Filters**: 
   - Filter out parcels intersecting airports
   - Filter by transmission line distance
   - Filter by utility provider (if selected)
3. **Calculate Drive Times**: Use isochrone data to calculate drive times
4. **Rank Parcels**: Apply multi-criteria ranking algorithm
5. **Save Results**: Save ranked parcels to PostgreSQL database
6. **Cleanup**: Remove temporary tables and data

## 🏗️ Project Structure

```
IDTool/
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── db_config.txt          # Database configuration
├── parcel_processing.log  # Application logs
├── data/                  # Data files directory
├── results/               # Output files directory
├── venv/                  # Virtual environment
└── parcel_filter/         # Core application package
    ├── __init__.py
    │   ├── config/
    │   │   ├── __init__.py
    │   │   └── settings.py
    │   ├── core/
    │   │   ├── __init__.py
    │   │   ├── db_utils.py    # Database utilities
    │   │   ├── filter.py      # Main filtering logic
    │   │   ├── ranking.py     # Ranking algorithms
    │   │   ├── map_viewer.py  # Interactive mapping
    │   │   └── version.py     # Version management
    │   ├── tests/             # Test suite
    │   │   ├── __init__.py
    │   │   ├── conftest.py
    │   │   ├── test_filter.py
    │   │   ├── test_ranking.py
    │   │   └── test_version.py
    │   └── utils/
    │       ├── __init__.py
    │       └── logging.py
    └── utils/
        ├── __init__.py
        └── logging.py
```

## 🎯 Ranking Criteria

The ranking system uses:

- **Zoning & Subtype**: Land use zoning classifications
- **Activity**: Land activity descriptions and site characteristics
- **Site**: Drive time and accessibility factors
- **FEMA NRI**: Flood risk and natural hazard ratings
- **Flood Zone**: Flood zone classifications

Additional factors include:
- Parcel size (normalized)
- Building coverage (inverted)
- Transmission line distance (inverted)

## 🗺️ Output Formats

### Database Tables
Results are saved to PostgreSQL with the naming convention:
```
results.{state}_{county}_{utility_provider}_{timestamp}
```

### Interactive Maps
When using `--quick-view`, generates:
- Interactive HTML maps with Folium
- Parcel boundaries with ranking information
- Color-coded ranking visualization
- Popup information for each parcel

### File Outputs
- GeoPackage files with ranked parcels
- JSON files with ranking weights and statistics
- Log files with detailed processing information

## 🔄 Updates and Maintenance

### Check for Updates
```bash
python main.py --update
```

### Environment Repair
```bash
python main.py --setup-env
```

## 📊 Dependencies

### Core Dependencies
- **psycopg2-binary**: PostgreSQL adapter
- **sqlalchemy**: Database ORM
- **geoalchemy2**: Spatial database support
- **geopandas**: Geospatial data processing
- **numpy/pandas**: Data manipulation
- **shapely**: Geometric operations

### Visualization
- **matplotlib**: Static plotting
- **folium**: Interactive mapping
- **branca**: Color mapping

### Data Processing
- **requests**: HTTP requests for Google Sheets
- **pyarrow**: Parquet file support
- **fiona**: Geospatial file I/O

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Verify `db_config.txt` settings
   - Ensure PostgreSQL is running
   - Check PostGIS extension is installed

2. **Missing Data**
   - Verify required schemas exist
   - Check parcel table naming convention
   - Ensure isochrone data is available

3. **Environment Issues**
   - Run `python main.py --setup-env`
   - Verify Python version is at least 3.8 but less than 3.12
   - Environment activation

### Logs
Check `parcel_processing.log` for detailed error information.


## 🔄 Version History

- **v0.2.3**: Current version
  - Enhanced utility provider name sanitization
  - Improved PostgreSQL table naming
  - Better error handling and logging
  - Interactive provider selection

---
