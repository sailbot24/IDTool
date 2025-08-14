#!/usr/bin/env python3
"""
Complete Database Rebuild Script

This script provides a comprehensive database rebuild system that can:
1. Rebuild the PostgreSQL database from scratch
2. Load parcel data with standardized schema
3. Load other GIS data (isochrones, airports, transmission lines, etc.)
4. Run enrichment SQL scripts in parallel
5. Handle table bloat with VACUUM operations
6. Provide progress tracking and error recovery

Usage:
    # Full rebuild with all data
    python rebuild_complete_database.py --rebuild-all
    
    # Rebuild specific components
    python rebuild_complete_database.py --rebuild-parcels
    python rebuild_complete_database.py --rebuild-other-gis
    python rebuild_complete_database.py --run-enrichment
    
    # State-specific rebuild
    python rebuild_complete_database.py --rebuild-parcels --state co
    
    # Test mode (no database changes)
    python rebuild_complete_database.py --test-only
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
import subprocess
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Complete Regrid Parcel Schema
REGRID_COLUMNS = [
    'ogc_fid', 'geoid', 'parcelnumb', 'parcelnumb_no_formatting', 'state_parcelnumb',
    'account_number', 'tax_id', 'alt_parcelnumb1', 'alt_parcelnumb2', 'alt_parcelnumb3',
    'usecode', 'usedesc', 'zoning', 'zoning_description', 'zoning_type', 'zoning_subtype',
    'zoning_code_link', 'zoning_id', 'struct', 'structno', 'yearbuilt', 'numstories',
    'numunits', 'numrooms', 'structstyle', 'parvaltype', 'improvval', 'landval', 'parval',
    'agval', 'homestead_exemption', 'saleprice', 'saledate', 'taxamt', 'taxyear',
    'owntype', 'owner', 'unmodified_owner', 'ownfrst', 'ownlast', 'owner2', 'owner3',
    'owner4', 'previous_owner', 'mailadd', 'mail_address2', 'careof', 'mail_addno',
    'mail_addpref', 'mail_addstr', 'mail_addsttyp', 'mail_addstsuf', 'mail_unit',
    'mail_city', 'mail_state2', 'mail_zip', 'mail_country', 'mail_urbanization',
    'original_mailing_address', 'address', 'address2', 'saddno', 'saddpref', 'saddstr',
    'saddsttyp', 'saddstsuf', 'sunit', 'scity', 'original_address', 'city', 'county',
    'state2', 'szip', 'szip5', 'urbanization', 'll_address_count', 'location_name',
    'address_source', 'legaldesc', 'plat', 'book', 'page', 'block', 'lot',
    'neighborhood', 'neighborhood_code', 'subdivision', 'lat', 'lon', 'fema_flood_zone',
    'fema_flood_zone_subtype', 'fema_flood_zone_raw', 'fema_flood_zone_data_date',
    'fema_nri_risk_rating', 'qoz', 'qoz_tract', 'census_tract', 'census_block',
    'census_blockgroup', 'census_zcta', 'census_elementary_school_district',
    'census_secondary_school_district', 'census_unified_school_district', 'll_last_refresh',
    'sourceurl', 'recrdareano', 'deeded_acres', 'gisacre', 'sqft', 'll_gisacre',
    'll_gissqft', 'll_bldg_footprint_sqft', 'll_bldg_count', 'cdl_raw', 'cdl_majority_category',
    'cdl_majority_percent', 'cdl_date', 'plss_township', 'plss_section', 'plss_range',
    'reviseddate', 'path', 'll_stable_id', 'll_uuid', 'll_stack_uuid', 'll_row_parcel',
    'll_updated_at', 'precisely_id', 'placekey', 'dpv_status', 'dpv_codes', 'dpv_notes',
    'dpv_type', 'cass_errorno', 'rdi', 'usps_vacancy', 'usps_vacancy_date',
    'padus_public_access', 'lbcs_activity', 'lbcs_activity_desc', 'lbcs_function',
    'lbcs_function_desc', 'lbcs_structure', 'lbcs_structure_desc', 'lbcs_site',
    'lbcs_site_desc', 'lbcs_ownership', 'lbcs_ownership_desc', 'housing_affordability_index',
    'population_density', 'population_growth_past_5_years', 'population_growth_next_5_years',
    'housing_growth_past_5_years', 'housing_growth_next_5_years',
    'household_income_growth_next_5_years', 'median_household_income',
    'transmission_line_distance', 'roughness_rating', 'highest_parcel_elevation',
    'lowest_parcel_elevation'
]

class DatabaseRebuilder:
    """Comprehensive database rebuild system with parallel processing and bloat management."""
    
    def __init__(self, db_config: Dict[str, str], data_dir: str = "data"):
        self.db_config = db_config
        self.data_dir = data_dir
        self.engine = None
        self.logger = logging.getLogger(__name__)
        
        # Directory structure
        self.raw_dir = os.path.join(data_dir, "raw")
        self.parcels_dir = os.path.join(self.raw_dir, "parcels")
        self.other_gis_dir = os.path.join(self.raw_dir, "other_gis")
        self.sql_scripts_dir = os.path.join(data_dir, "sql_scripts")
        
        # Ensure directories exist
        for dir_path in [self.raw_dir, self.parcels_dir, self.other_gis_dir, self.sql_scripts_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def setup_logging(self):
        """Configure logging settings."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('database_rebuild.log')
            ]
        )
    
    def create_database_connection(self):
        """Create database connection with proper settings."""
        try:
            # Create connection string
            password = self.db_config.get('password')
            user = self.db_config['user']
            auth = f"{user}:{password}" if password else user
            connection_string = f"postgresql://{auth}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            self.engine = create_engine(connection_string, pool_pre_ping=True)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.logger.info("Database connection established successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return False
    
    def drop_and_recreate_database(self):
        """Drop and recreate the database completely."""
        try:
            # Connect to postgres database to drop/recreate target database
            password = self.db_config.get('password')
            user = self.db_config['user']
            auth = f"{user}:{password}" if password else user
            postgres_conn_string = f"postgresql://{auth}@{self.db_config['host']}:{self.db_config['port']}/postgres"
            postgres_engine = create_engine(postgres_conn_string)
            
            with postgres_engine.connect() as conn:
                conn.execute(text("COMMIT"))  # End any open transactions
                
                # Terminate connections to target database
                conn.execute(text(f"""
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = '{self.db_config['database']}'
                    AND pid <> pg_backend_pid()
                """))
                
                # Drop database if exists
                conn.execute(text(f"DROP DATABASE IF EXISTS {self.db_config['database']}"))
                
                # Create new database
                conn.execute(text(f"CREATE DATABASE {self.db_config['database']}"))
                
                conn.commit()
            
            self.logger.info(f"Database {self.db_config['database']} recreated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to recreate database: {e}")
            return False
    
    def setup_database_extensions(self):
        """Set up required PostgreSQL extensions."""
        try:
            with self.engine.connect() as conn:
                # Create required extensions
                extensions = [
                    "CREATE EXTENSION IF NOT EXISTS postgis",
                    "CREATE EXTENSION IF NOT EXISTS postgis_topology",
                    "CREATE EXTENSION IF NOT EXISTS postgis_raster"
                ]
                
                for extension in extensions:
                    conn.execute(text(extension))
                
                conn.commit()
                self.logger.info("Database extensions installed successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to install database extensions: {e}")
            return False
    
    def create_schemas(self):
        """Create required database schemas."""
        try:
            with self.engine.connect() as conn:
                schemas = ['parcels', 'other_gis', 'results', 'enrichment', 'rextag']
                
                for schema in schemas:
                    conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
                
                conn.commit()
                self.logger.info("Database schemas created successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to create schemas: {e}")
            return False
    
    def discover_parcel_structure(self) -> Dict[str, Dict[str, List[str]]]:
        """Discover the parcel file structure from the raw directory."""
        structure = {}
        
        if not os.path.exists(self.parcels_dir):
            self.logger.warning(f"Parcels directory not found: {self.parcels_dir}")
            return structure
        
        # Find all state directories
        state_dirs = [d for d in os.listdir(self.parcels_dir) 
                     if os.path.isdir(os.path.join(self.parcels_dir, d)) and not d.startswith('.')]
        
        for state in state_dirs:
            state_path = os.path.join(self.parcels_dir, state)
            structure[state] = {}
            
            # Look for county subdirectories
            county_dirs = [d for d in os.listdir(state_path) 
                          if os.path.isdir(os.path.join(state_path, d)) and not d.startswith('.')]
            
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
    
    def process_county_parcels(self, args: Tuple[str, str, List[str]]) -> Tuple[str, int]:
        """Process a single county: create table with geometry and load all files sequentially."""
        state, county, files = args
        table_name = f"{state}_{county}"
        rows_loaded = 0

        try:
            connection_string = f"postgresql://{self.db_config['user']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            engine = create_engine(connection_string)

            with engine.connect() as conn:
                # Drop existing table to ensure clean schema with geometry
                conn.execute(text(f"DROP TABLE IF EXISTS parcels.{table_name}"))
                conn.commit()

            # Load files sequentially to avoid schema conflicts
            first_loaded = False
            for file_path in files:
                try:
                    if file_path.lower().endswith('.csv'):
                        # CSV likely lacks geometry; skip with warning
                        self.logger.warning(f"Skipping CSV without geometry for {table_name}: {file_path}")
                        continue

                    gdf: gpd.GeoDataFrame = gpd.read_file(file_path)

                    # Ensure geometry present
                    if gdf.geometry is None or gdf.geometry.is_empty.all():
                        self.logger.warning(f"No geometry in file, skipping: {file_path}")
                        continue

                    # Ensure geometry column named 'geom'
                    geom_col_name = gdf.geometry.name if gdf.geometry is not None else None
                    if geom_col_name and geom_col_name != 'geom':
                        gdf = gdf.rename(columns={geom_col_name: 'geom'}).set_geometry('geom')

                    # Write to PostGIS. First file creates table with geometry, others append
                    gdf.to_postgis(
                        name=table_name,
                        con=engine,
                        schema='parcels',
                        if_exists='replace' if not first_loaded else 'append',
                        index=False,
                        chunksize=10000,
                        dtype=None
                    )

                    rows_loaded += len(gdf)
                    first_loaded = True
                except Exception as e:
                    self.logger.error(f"Failed loading file into {table_name}: {file_path} error: {e}")

            # Create spatial index on geom
            try:
                with engine.connect() as conn:
                    conn.execute(text(f"CREATE INDEX IF NOT EXISTS {table_name}_geom_idx ON parcels.{table_name} USING GIST (geom)"))
                    conn.execute(text(f"ANALYZE parcels.{table_name}"))
                    conn.commit()
            except Exception as e:
                self.logger.warning(f"Could not create spatial index for {table_name}: {e}")

            return (table_name, rows_loaded)

        except Exception as e:
            self.logger.error(f"Error processing county {table_name}: {e}")
            return (table_name, 0)
    
    def rebuild_parcels(self, state_filter: Optional[str] = None, max_workers: int = 4):
        """Rebuild parcel data with parallel processing."""
        self.logger.info("Starting parcel data rebuild...")
        
        # Discover parcel structure
        structure = self.discover_parcel_structure()
        
        if not structure:
            self.logger.error("No parcel files found")
            return False
        
        # Filter by state if specified
        if state_filter:
            if state_filter not in structure:
                self.logger.error(f"State {state_filter} not found")
                return False
            structure = {state_filter: structure[state_filter]}
        
        # Prepare per-county tasks and process in parallel
        county_tasks: List[Tuple[str, str, List[str]]] = []
        for state, counties in structure.items():
            for county, files in counties.items():
                county_tasks.append((state, county, files))

        total_rows = 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_county_parcels, task) for task in county_tasks]
            for future in as_completed(futures):
                table_name, row_count = future.result()
                total_rows += row_count
                self.logger.info(f"Loaded {row_count} rows into parcels.{table_name}")
        
        self.logger.info(f"Parcel rebuild completed. Total rows: {total_rows}")
        return True
    
    def load_other_gis_data(self):
        """Load other GIS data (isochrones, airports, transmission lines, etc.)."""
        self.logger.info("Loading other GIS data...")
        
        if not os.path.exists(self.other_gis_dir):
            self.logger.warning(f"Other GIS directory not found: {self.other_gis_dir}")
            return False
        
        try:
            # Find all GIS files
            gis_files = []
            for ext in ['*.shp', '*.geojson', '*.gpkg', '*.csv']:
                gis_files.extend(glob.glob(os.path.join(self.other_gis_dir, f"**/{ext}"), recursive=True))
            
            if not gis_files:
                self.logger.warning("No GIS files found in other_gis directory")
                return False
            
            with self.engine.connect() as conn:
                for file_path in gis_files:
                    try:
                        # Determine table name from file name
                        file_name = os.path.splitext(os.path.basename(file_path))[0]
                        table_name = f"other_gis.{file_name}"
                        
                        # Read file
                        if file_path.endswith('.csv'):
                            df = pd.read_csv(file_path)
                        else:
                            df = gpd.read_file(file_path)
                        
                        # Load into database
                        df.to_postgis(
                            file_name,
                            self.engine,
                            schema='other_gis',
                            if_exists='replace',
                            index=False
                        )
                        
                        # Create spatial index if geometry exists
                        try:
                            conn.execute(text(f"""
                                CREATE INDEX {file_name}_geom_idx 
                                ON other_gis.{file_name} 
                                USING GIST (geom)
                            """))
                        except Exception as e:
                            self.logger.warning(f"Could not create spatial index for {file_name}: {e}")
                        
                        self.logger.info(f"Loaded {len(df)} rows into {table_name}")
                        
                    except Exception as e:
                        self.logger.error(f"Error loading {file_path}: {e}")
                
                conn.commit()
            
            self.logger.info("Other GIS data loading completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load other GIS data: {e}")
            return False

    def load_rextag_data(self):
        """Load rextag (energy infrastructure) data."""
        self.logger.info("Loading rextag data...")

        rextag_dir = os.path.join(self.raw_dir, "rextag")
        if not os.path.exists(rextag_dir):
            self.logger.warning(f"Rextag directory not found: {rextag_dir}")
            return False

        try:
            rextag_files = []
            for ext in ['*.shp', '*.geojson', '*.gpkg', '*.csv']:
                rextag_files.extend(glob.glob(os.path.join(rextag_dir, f"**/{ext}"), recursive=True))

            if not rextag_files:
                self.logger.warning("No rextag files found")
                return False

            with self.engine.connect() as conn:
                for file_path in rextag_files:
                    try:
                        file_name = os.path.splitext(os.path.basename(file_path))[0]
                        if file_path.lower().endswith('.csv'):
                            df = pd.read_csv(file_path)
                            df.to_postgis(file_name, self.engine, schema='rextag', if_exists='replace', index=False)
                        else:
                            gdf = gpd.read_file(file_path)
                            gdf.to_postgis(file_name, self.engine, schema='rextag', if_exists='replace', index=False)

                        # Try spatial index
                        try:
                            conn.execute(text(f"CREATE INDEX IF NOT EXISTS {file_name}_geom_idx ON rextag.{file_name} USING GIST (geom)"))
                        except Exception as e:
                            self.logger.debug(f"No geom or index failed for rextag.{file_name}: {e}")

                        self.logger.info(f"Loaded {file_name} into rextag schema")
                    except Exception as e:
                        self.logger.error(f"Failed loading rextag file {file_path}: {e}")
                conn.commit()

            self.logger.info("Rextag data loading completed")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load rextag data: {e}")
            return False
    
    def run_enrichment_scripts(self) -> bool:
        """Run enrichment SQL scripts in deterministic order to avoid lock contention."""
        self.logger.info("Running enrichment SQL scripts...")

        if not os.path.exists(self.sql_scripts_dir):
            self.logger.warning(f"SQL scripts directory not found: {self.sql_scripts_dir}")
            return False

        sql_files = sorted(glob.glob(os.path.join(self.sql_scripts_dir, "*.sql")))
        if not sql_files:
            self.logger.warning("No SQL scripts found")
            return False

        with self.engine.connect() as conn:
            for sql_file in sql_files:
                try:
                    self.logger.info(f"Executing {os.path.basename(sql_file)}...")
                    with open(sql_file, 'r') as f:
                        sql_content = f.read()
                    conn.execute(text(sql_content))
                    conn.commit()
                    self.logger.info(f"Completed {os.path.basename(sql_file)}")
                except Exception as e:
                    self.logger.error(f"Error running {sql_file}: {e}")
                    return False

        self.logger.info("All enrichment scripts completed")
        return True
    
    def vacuum_and_analyze(self):
        """Run VACUUM FULL and ANALYZE to handle table bloat."""
        self.logger.info("Running VACUUM FULL and ANALYZE...")
        
        try:
            with self.engine.connect() as conn:
                # Get all tables
                result = conn.execute(text("""
                    SELECT schemaname, tablename 
                    FROM pg_tables 
                    WHERE schemaname IN ('parcels', 'other_gis', 'results', 'enrichment', 'rextag')
                """))
                
                tables = result.fetchall()
                
                for schema, table in tables:
                    table_name = f"{schema}.{table}"
                    self.logger.info(f"Running VACUUM FULL on {table_name}")
                    
                    try:
                        conn.execute(text(f"VACUUM FULL {table_name}"))
                        conn.execute(text(f"ANALYZE {table_name}"))
                        self.logger.info(f"Completed VACUUM FULL and ANALYZE on {table_name}")
                    except Exception as e:
                        self.logger.warning(f"Error running VACUUM on {table_name}: {e}")
                
                conn.commit()
            
            self.logger.info("VACUUM FULL and ANALYZE completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to run VACUUM FULL: {e}")
            return False
    
    def rebuild_complete_database(self, state_filter: Optional[str] = None, max_workers: int = 4):
        """Complete database rebuild process."""
        self.logger.info("Starting complete database rebuild...")
        
        start_time = time.time()
        
        try:
            # Step 1: Drop and recreate database
            self.logger.info("Step 1: Recreating database...")
            if not self.drop_and_recreate_database():
                return False
            
            # Step 2: Create database connection
            self.logger.info("Step 2: Setting up database connection...")
            if not self.create_database_connection():
                return False
            
            # Step 3: Install extensions
            self.logger.info("Step 3: Installing database extensions...")
            if not self.setup_database_extensions():
                return False
            
            # Step 4: Create schemas
            self.logger.info("Step 4: Creating database schemas...")
            if not self.create_schemas():
                return False
            
            # Step 5: Load parcel data
            self.logger.info("Step 5: Loading parcel data...")
            if not self.rebuild_parcels(state_filter, max_workers):
                return False
            
            # Step 6: Load other GIS data
            self.logger.info("Step 6: Loading other GIS data...")
            if not self.load_other_gis_data():
                return False

            # Step 7: Load rextag data
            self.logger.info("Step 7: Loading rextag data...")
            if not self.load_rextag_data():
                self.logger.warning("Proceeding without rextag data; enrichment may be incomplete")

            # Step 8: Run enrichment scripts
            self.logger.info("Step 8: Running enrichment scripts...")
            if not self.run_enrichment_scripts():
                return False

            # Step 9: Handle table bloat
            self.logger.info("Step 9: Running VACUUM FULL and ANALYZE...")
            if not self.vacuum_and_analyze():
                return False
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Complete database rebuild finished in {elapsed_time:.2f} seconds")
            return True
            
        except Exception as e:
            self.logger.error(f"Database rebuild failed: {e}")
            return False

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

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Complete database rebuild system")
    parser.add_argument("--rebuild-all", action="store_true", help="Rebuild entire database")
    parser.add_argument("--rebuild-parcels", action="store_true", help="Rebuild parcel data only")
    parser.add_argument("--rebuild-other-gis", action="store_true", help="Rebuild other GIS data only")
    parser.add_argument("--run-enrichment", action="store_true", help="Run enrichment scripts only")
    parser.add_argument("--state", help="Two-letter state code to process (e.g., co, az)")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    parser.add_argument("--db-config", default="db_config.txt", help="Database configuration file")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum parallel workers")
    parser.add_argument("--test-only", action="store_true", help="Test mode (no database changes)")
    args = parser.parse_args()
    
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        
        # Read database configuration
        db_config = read_db_config(args.db_config)
        logger.info(f"Using database configuration from {args.db_config}")
        
        # Create rebuilder instance
        rebuilder = DatabaseRebuilder(db_config, args.data_dir)
        
        if args.test_only:
            logger.info("TEST MODE: No database changes will be made")
            # Just test the connection
            if rebuilder.create_database_connection():
                logger.info("Database connection test successful")
            else:
                logger.error("Database connection test failed")
            return
        
        # Determine what to rebuild
        if args.rebuild_all:
            logger.info("Starting complete database rebuild...")
            success = rebuilder.rebuild_complete_database(args.state, args.max_workers)
        elif args.rebuild_parcels:
            logger.info("Starting parcel data rebuild...")
            if not rebuilder.create_database_connection():
                return
            success = rebuilder.rebuild_parcels(args.state, args.max_workers)
        elif args.rebuild_other_gis:
            logger.info("Starting other GIS data rebuild...")
            if not rebuilder.create_database_connection():
                return
            success = rebuilder.load_other_gis_data()
        elif args.run_enrichment:
            logger.info("Starting enrichment scripts...")
            if not rebuilder.create_database_connection():
                return
            success = rebuilder.run_enrichment_scripts(args.max_workers // 2)
        else:
            logger.error("Please specify what to rebuild (--rebuild-all, --rebuild-parcels, etc.)")
            return
        
        if success:
            logger.info("Database rebuild completed successfully!")
        else:
            logger.error("Database rebuild failed!")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 