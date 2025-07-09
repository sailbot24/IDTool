#!/usr/bin/env python
"""
Database Rebuild Script

This script rebuilds the parcel database from raw data files in the raw/parcels directory.
It standardizes the schema based on the Regrid parcel schema and loads all data into PostgreSQL.

Usage:
    # Test mode - analyze schema compliance without database updates
    python rebuild_database.py --raw-dir raw/parcels --test-only
    python rebuild_database.py --raw-dir raw/parcels --state co --test-only
    
    # Normal mode - rebuild database
    python rebuild_database.py --raw-dir raw/parcels
    python rebuild_database.py --raw-dir raw/parcels --state co
    python rebuild_database.py --raw-dir raw/parcels --all-states
"""

import argparse
import logging
import os
import glob
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text, inspect
from typing import Dict, List, Set, Optional, Tuple
import sys
from pathlib import Path
import json
from collections import defaultdict

# Complete Regrid Parcel Schema (all standard columns that should be present)
REGRID_COLUMNS = [
    'ogc_fid',
    'geoid',
    'parcelnumb',
    'parcelnumb_no_formatting',
    'state_parcelnumb',
    'account_number',
    'tax_id',
    'alt_parcelnumb1',
    'alt_parcelnumb2',
    'alt_parcelnumb3',
    'usecode',
    'usedesc',
    'zoning',
    'zoning_description',
    'zoning_type',
    'zoning_subtype',
    'zoning_code_link',
    'zoning_id',
    'struct',
    'structno',
    'yearbuilt',
    'numstories',
    'numunits',
    'numrooms',
    'structstyle',
    'parvaltype',
    'improvval',
    'landval',
    'parval',
    'agval',
    'homestead_exemption',
    'saleprice',
    'saledate',
    'taxamt',
    'taxyear',
    'owntype',
    'owner',
    'unmodified_owner',
    'ownfrst',
    'ownlast',
    'owner2',
    'owner3',
    'owner4',
    'previous_owner',
    'mailadd',
    'mail_address2',
    'careof',
    'mail_addno',
    'mail_addpref',
    'mail_addstr',
    'mail_addsttyp',
    'mail_addstsuf',
    'mail_unit',
    'mail_city',
    'mail_state2',
    'mail_zip',
    'mail_country',
    'mail_urbanization',
    'original_mailing_address',
    'address',
    'address2',
    'saddno',
    'saddpref',
    'saddstr',
    'saddsttyp',
    'saddstsuf',
    'sunit',
    'scity',
    'original_address',
    'city',
    'county',
    'state2',
    'szip',
    'szip5',
    'urbanization',
    'll_address_count',
    'location_name',
    'address_source',
    'legaldesc',
    'plat',
    'book',
    'page',
    'block',
    'lot',
    'neighborhood',
    'neighborhood_code',
    'subdivision',
    'lat',
    'lon',
    'fema_flood_zone',
    'fema_flood_zone_subtype',
    'fema_flood_zone_raw',
    'fema_flood_zone_data_date',
    'fema_nri_risk_rating',
    'qoz',
    'qoz_tract',
    'census_tract',
    'census_block',
    'census_blockgroup',
    'census_zcta',
    'census_elementary_school_district',
    'census_secondary_school_district',
    'census_unified_school_district',
    'll_last_refresh',
    'sourceurl',
    'recrdareano',
    'deeded_acres',
    'gisacre',
    'sqft',
    'll_gisacre',
    'll_gissqft',
    'll_bldg_footprint_sqft',
    'll_bldg_count',
    'cdl_raw',
    'cdl_majority_category',
    'cdl_majority_percent',
    'cdl_date',
    'plss_township',
    'plss_section',
    'plss_range',
    'reviseddate',
    'path',
    'll_stable_id',
    'll_uuid',
    'll_stack_uuid',
    'll_row_parcel',
    'll_updated_at',
    'precisely_id',
    'placekey',
    'dpv_status',
    'dpv_codes',
    'dpv_notes',
    'dpv_type',
    'cass_errorno',
    'rdi',
    'usps_vacancy',
    'usps_vacancy_date',
    'padus_public_access',
    'lbcs_activity',
    'lbcs_activity_desc',
    'lbcs_function',
    'lbcs_function_desc',
    'lbcs_structure',
    'lbcs_structure_desc',
    'lbcs_site',
    'lbcs_site_desc',
    'lbcs_ownership',
    'lbcs_ownership_desc',
    'housing_affordability_index',
    'population_density',
    'population_growth_past_5_years',
    'population_growth_next_5_years',
    'housing_growth_past_5_years',
    'housing_growth_next_5_years',
    'household_income_growth_next_5_years',
    'median_household_income',
    'transmission_line_distance',
    'roughness_rating',
    'highest_parcel_elevation',
    'lowest_parcel_elevation'
]

def setup_logging():
    """Configure logging settings."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def read_db_config(config_path: str = "db_config.txt") -> Dict[str, str]:
    """Read database configuration from a file."""
    config = {}
    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
        return config
    except Exception as e:
        logging.error(f"Error reading database configuration: {e}")
        raise

def discover_parcel_structure(raw_dir: str) -> Dict[str, Dict[str, List[str]]]:
    """Discover the parcel file structure from the raw directory."""
    structure = {}
    
    # Look for parcels subdirectory
    parcels_dir = os.path.join(raw_dir, 'parcels')
    if not os.path.exists(parcels_dir):
        # If no parcels subdirectory, use the raw_dir directly
        parcels_dir = raw_dir
    
    # Find all state directories
    state_dirs = [d for d in os.listdir(parcels_dir) if os.path.isdir(os.path.join(parcels_dir, d)) and not d.startswith('.')]
    
    for state in state_dirs:
        state_path = os.path.join(parcels_dir, state)
        structure[state] = {}
        
        # Look for county subdirectories
        county_dirs = [d for d in os.listdir(state_path) if os.path.isdir(os.path.join(state_path, d)) and not d.startswith('.')]
        
        if county_dirs:
            # Organized by county
            for county in county_dirs:
                county_path = os.path.join(state_path, county)
                files = []
                for ext in ['*.shp', '*.geojson', '*.gpkg', '*.csv']:
                    files.extend(glob.glob(os.path.join(county_path, ext)))
                if files:
                    structure[state][county] = files
        else:
            # Files directly in state directory
            files = []
            for ext in ['*.shp', '*.geojson', '*.gpkg', '*.csv']:
                files.extend(glob.glob(os.path.join(state_path, ext)))
            if files:
                # Use state name as county if no county subdirectories
                structure[state][state] = files
    
    return structure

def get_file_columns(file_path: str) -> List[str]:
    """Get column names from a parcel file."""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, nrows=1)  # Just read header
            return list(df.columns)
        else:
            # For spatial files
            gdf = gpd.read_file(file_path, rows=1)  # Just read first row
            return list(gdf.columns)
    except Exception as e:
        logging.warning(f"Could not read columns from {file_path}: {e}")
        return []

def check_and_add_missing_columns(columns: List[str]) -> List[str]:
    """Check if all Regrid schema columns are present and return missing columns."""
    missing_columns = [col for col in REGRID_COLUMNS if col not in columns]
    return missing_columns

def analyze_schema_compliance(file_path: str, state: str = None) -> Tuple[List[str], List[str]]:
    """Analyze how well a file complies with the Regrid schema."""
    columns = get_file_columns(file_path)
    missing_columns = check_and_add_missing_columns(columns)
    
    # Find extra columns (not in Regrid schema)
    extra_columns = [col for col in columns if col not in REGRID_COLUMNS]
    
    return missing_columns, extra_columns

def create_standardized_table(engine, table_name: str) -> None:
    """Create a table with the standardized Regrid schema."""
    with engine.connect() as conn:
        # Create parcels schema if it doesn't exist
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS parcels"))
        
        # Drop table if it exists
        conn.execute(text(f"DROP TABLE IF EXISTS parcels.{table_name}"))
        
        # Create table with standard schema
        columns_sql = []
        for col_name in REGRID_COLUMNS:
            columns_sql.append(f"{col_name} TEXT") # Assuming TEXT for all columns for simplicity
        
        create_sql = f"""
        CREATE TABLE parcels.{table_name} (
            {', '.join(columns_sql)}
        )
        """
        conn.execute(text(create_sql))
        
        # Create spatial index
        conn.execute(text(f"""
            CREATE INDEX {table_name}_geom_idx 
            ON parcels.{table_name} 
            USING GIST (geom)
        """))
        
        conn.commit()

def load_parcel_file(file_path: str, table_name: str, engine) -> int:
    """Load a parcel file into the database with standardized schema."""
    try:
        logging.info(f"Loading {file_path} into {table_name}")
        
        # Read the file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = gpd.read_file(file_path)
        
        # Check for missing columns
        missing_columns = check_and_add_missing_columns(list(df.columns))
        
        # Add missing columns with NULL values
        for col_name in missing_columns:
            df[col_name] = None
            logging.info(f"Added missing column: {col_name}")
        
        # Select only the standard columns in the right order
        df = df[REGRID_COLUMNS]
        
        # Load into database
        df.to_postgis(
            table_name,
            engine,
            schema='parcels',
            if_exists='append',
            index=False
        )
        
        row_count = len(df)
        logging.info(f"Loaded {row_count} rows from {file_path}")
        return row_count
        
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return 0

def generate_schema_report(structure: Dict[str, Dict[str, List[str]]]) -> Dict:
    """Generate a comprehensive report on schema compliance."""
    report = {
        'summary': {
            'total_states': len(structure),
            'total_counties': sum(len(counties) for counties in structure.values()),
            'total_files': sum(len(files) for counties in structure.values() for files in counties.values()),
            'total_columns_added': 0,
            'total_columns_unmapped': 0
        },
        'states': {},
        'files': []
    }
    
    for state, counties in structure.items():
        state_report = {
            'counties': len(counties),
            'files': sum(len(files) for files in counties.values()),
            'columns_added': 0,
            'columns_unmapped': 0,
            'state_columns': set()  # Track all unique columns found in this state
        }
        
        for county, files in counties.items():
            for file_path in files:
                missing, extra = analyze_schema_compliance(file_path, state)
                
                # Track all columns found in this state
                file_columns = get_file_columns(file_path)
                state_report['state_columns'].update(file_columns)
                
                file_report = {
                    'state': state,
                    'county': county,
                    'file': os.path.basename(file_path),
                    'file_path': file_path,
                    'missing_columns': missing,
                    'unmapped_columns': extra,
                    'columns_added': len(missing),
                    'columns_unmapped': len(extra),
                    'total_columns_in_file': len(file_columns)
                }
                
                report['files'].append(file_report)
                state_report['columns_added'] += len(missing)
                state_report['columns_unmapped'] += len(extra)
        
        # Convert set to list for JSON serialization
        state_report['state_columns'] = list(state_report['state_columns'])
        report['states'][state] = state_report
        report['summary']['total_columns_added'] += state_report['columns_added']
        report['summary']['total_columns_unmapped'] += state_report['columns_unmapped']
    
    return report

def print_schema_report(report: Dict) -> None:
    """Print a formatted schema compliance report."""
    print("\n" + "="*80)
    print("SCHEMA COMPLIANCE REPORT")
    print("="*80)
    
    summary = report['summary']
    print(f"\nSUMMARY:")
    print(f"  Total States: {summary['total_states']}")
    print(f"  Total Counties: {summary['total_counties']}")
    print(f"  Total Files: {summary['total_files']}")
    print(f"  Total Columns Added: {summary['total_columns_added']}")
    print(f"  Total Extra Columns Found: {summary['total_columns_unmapped']}")
    
    print(f"\nBY STATE:")
    for state, state_report in report['states'].items():
        print(f"  {state.upper()}:")
        print(f"    Counties: {state_report['counties']}")
        print(f"    Files: {state_report['files']}")
        print(f"    Columns Added: {state_report['columns_added']}")
        print(f"    Extra Columns: {state_report['columns_unmapped']}")
        print(f"    Total Unique Columns in State: {len(state_report['state_columns'])}")
        print(f"    State Columns: {', '.join(sorted(state_report['state_columns'])[:10])}{'...' if len(state_report['state_columns']) > 10 else ''}")
    
    print(f"\nDETAILED FILE ANALYSIS:")
    for file_report in report['files']:
        print(f"  {file_report['state']}/{file_report['county']}/{file_report['file']}:")
        print(f"    Total Columns: {file_report['total_columns_in_file']}")
        print(f"    Missing Columns Added: {len(file_report['missing_columns'])}")
        print(f"    Extra Columns: {len(file_report['unmapped_columns'])}")
        if file_report['missing_columns']:
            print(f"    Added: {', '.join(file_report['missing_columns'][:5])}{'...' if len(file_report['missing_columns']) > 5 else ''}")
        if file_report['unmapped_columns']:
            print(f"    Extra: {', '.join(file_report['unmapped_columns'][:5])}{'...' if len(file_report['unmapped_columns']) > 5 else ''}")
    
    print("="*80)

def rebuild_database(raw_dir: str, engine, state_filter: Optional[str] = None) -> None:
    """Rebuild the database from raw parcel files."""
    logger = logging.getLogger(__name__)
    
    # Discover parcel structure
    logger.info(f"Discovering parcel structure in {raw_dir}")
    structure = discover_parcel_structure(raw_dir)
    
    if not structure:
        logger.error(f"No parcel files found in {raw_dir}")
        return
    
    logger.info(f"Found {len(structure)} states: {list(structure.keys())}")
    
    # Generate and print schema report
    logger.info("Analyzing schema compliance...")
    report = generate_schema_report(structure)
    print_schema_report(report)
    
    # Filter by state if specified
    if state_filter:
        if state_filter not in structure:
            logger.error(f"State {state_filter} not found in {raw_dir}")
            return
        structure = {state_filter: structure[state_filter]}
    
    # Process each state
    total_rows = 0
    for state, counties in structure.items():
        logger.info(f"Processing state: {state}")
        
        for county, files in counties.items():
            table_name = f"{state}_{county}"
            logger.info(f"Processing county: {county}")
            
            # Create standardized table
            create_standardized_table(engine, table_name)
            
            # Load all files for this county
            county_rows = 0
            for file_path in files:
                rows = load_parcel_file(file_path, table_name, engine)
                county_rows += rows
            
            logger.info(f"County {county}: {county_rows} total rows")
            total_rows += county_rows
    
    logger.info(f"Total rows loaded: {total_rows}")
    
    # Save report to file
    report_file = f"schema_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path = os.path.abspath(report_file)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Schema report saved to: {report_path}")
    print(f"\nSCHEMA REPORT SAVED TO: {report_path}")
    
    # Print final summary
    logger.info("Database rebuild completed!")

def test_schema_compliance(raw_dir: str, state_filter: Optional[str] = None) -> None:
    """Analyze schema compliance in test mode (no database updates)."""
    logger = logging.getLogger(__name__)
    
    # Discover parcel structure
    logger.info(f"Discovering parcel structure in {raw_dir}")
    structure = discover_parcel_structure(raw_dir)
    
    if not structure:
        logger.error(f"No parcel files found in {raw_dir}")
        return
    
    logger.info(f"Found {len(structure)} states: {list(structure.keys())}")
    
    # Generate and print schema report
    logger.info("Analyzing schema compliance...")
    report = generate_schema_report(structure)
    print_schema_report(report)
    
    # Filter by state if specified
    if state_filter:
        if state_filter not in structure:
            logger.error(f"State {state_filter} not found in {raw_dir}")
            return
        structure = {state_filter: structure[state_filter]}
    
    # Save report to file
    report_file = f"schema_report_test_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path = os.path.abspath(report_file)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Test schema report saved to: {report_path}")
    print(f"\nTEST SCHEMA REPORT SAVED TO: {report_path}")
    
    logger.info("Test mode completed. No database updates were made.")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Rebuild parcel database from raw data files")
    parser.add_argument("--raw-dir", default='/Users/gavin/Desktop/Personal/Adams Geospatial/Data/Garrison/data/raw', help="Directory containing raw parcel data (e.g., 'raw/parcels')")
    parser.add_argument("--state", help="Two-letter state code to process (e.g., co, az)")
    parser.add_argument("--db-config", default="db_config.txt", help="Path to database configuration file")
    parser.add_argument("--test-only", action="store_true", help="Only analyze schema compliance without updating database")
    args = parser.parse_args()
    
    try:
        # Set up logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # Convert relative path to absolute path if needed
        raw_dir = args.raw_dir
        if not os.path.isabs(raw_dir):
            # Make relative to current working directory
            raw_dir = os.path.abspath(raw_dir)
            logger.info(f"Using relative path: {args.raw_dir} -> {raw_dir}")
        
        # Check if raw directory exists
        if not os.path.exists(raw_dir):
            logger.error(f"Raw directory not found: {raw_dir}")
            logger.error(f"Current working directory: {os.getcwd()}")
            logger.error(f"Available directories: {[d for d in os.listdir('.') if os.path.isdir(d)]}")
            sys.exit(1)
        
        # Read database configuration
        db_config = read_db_config(args.db_config)
        logger.info(f"Using database configuration from {args.db_config}")
        
        if args.test_only:
            # Test mode: only analyze schema compliance
            logger.info("TEST MODE: Analyzing schema compliance without database updates")
            test_schema_compliance(raw_dir, args.state)
        else:
            # Normal mode: create database connection and rebuild
            engine = create_engine(
                f"postgresql://{db_config['user']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
            rebuild_database(raw_dir, engine, args.state)
        
        logger.info("Operation completed successfully!")
        
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 