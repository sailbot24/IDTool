import psycopg2
import logging
import pandas as pd
import re
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from .mcda_ranking import MCDARanker
import os
import geopandas as gpd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from geoalchemy2 import Geometry
from datetime import datetime
from .db_utils import DatabaseUtils
import time

logger = logging.getLogger(__name__)

class ParcelFilter:
    def __init__(self, state: str, county: Optional[str] = None, 
                 db_config: Optional[Dict[str, str]] = None,
                 ranking_url: Optional[str] = None,
                 data_dir: str = "",
                 transmission_distance: float = 100.0):
        """
        Initialize ParcelFilter with PostgreSQL connection.
        
        Args:
            state: Two-letter state code (e.g., 'az')
            county: County name (e.g., 'maricopa'). If None, will analyze all counties in the state.
            db_config: Dictionary containing PostgreSQL connection details:
                      {
                          'host': 'localhost',
                          'port': '5432',
                          'database': 'LandAI',
                          'user': 'postgres'
                      }
            ranking_url: URL to Google Sheets document containing ranking data
            data_dir: Directory containing data files
            transmission_distance: Maximum distance to transmission lines in meters (default: 100.0)
        """
        self.state = state.lower()
        self.county = county.lower() if county else None
        self.db_config = db_config or {
            'host': 'localhost',
            'port': '5432',
            'database': 'LandAI',
            'user': 'postgres'
        }
        self.db_utils = DatabaseUtils(state, county, db_config)
        self.ranking_url = ranking_url
        self.ranker = MCDARanker(ranking_url, self.state, self.county or "") if ranking_url else None
        self.isochrone_data = None
        self.filtered_parcels = None
        self.unwanted_activities = None
        self.utility_filter = None
        self.data_dir = data_dir
        self.transmission_distance = transmission_distance
        self.filtered_parcels_table = None
        self.ranking_data = None
        self.weights = None
        self.power_provider = None
        self.final_results_table = None

    def __enter__(self):
        """Context manager entry."""
        self.db_utils.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures connection is closed."""
        if self.db_utils.engine is not None:
            self.db_utils.engine.dispose()
            self.db_utils.engine = None

    def connect(self):
        """Connect to PostgreSQL database."""
        self.db_utils.connect()

    def load_isochrone_data(self) -> None:
        """Load isochrone data from the database."""
        logger.info("Loading isochrone data from database")
        try:
            # Use GeoPandas' built-in PostgreSQL support
            self.isochrone_data = gpd.read_postgis(
                "SELECT * FROM other_gis.iso_50",
                self.db_utils.engine,
                geom_col='geom'
            )
            # Ensure CRS is set correctly
            if self.isochrone_data.crs is None:
                self.isochrone_data.set_crs(epsg=4326, inplace=True)
            self.isochrone_data = self.isochrone_data.to_crs(epsg=5070)
            logger.info(f"Loaded {len(self.isochrone_data)} isochrone polygons")
            logger.debug(f"Isochrone columns: {self.isochrone_data.columns.tolist()}")
        except Exception as e:
            logger.error(f"Error loading isochrone data: {e}")
            raise

    def load_unwanted_activities(self) -> List[str]:
        """Load unwanted activities from Google Sheets."""
        if not self.ranking_url:
            raise ValueError("No ranking URL provided")
            
        try:
            # Extract the document ID from the URL
            if 'docs.google.com/spreadsheets' in self.ranking_url:
                doc_id = self.ranking_url.split('/d/')[1].split('/')[0]
                
                # Load unwanted activities (sheet gid=1501635609)
                unwanted_url = f"https://docs.google.com/spreadsheets/d/{doc_id}/gviz/tq?tqx=out:csv&gid=1501635609"
                unwanted_df = pd.read_csv(unwanted_url)
                unwanted_df.columns = unwanted_df.columns.str.strip()
                logger.info(f"Unwanted activities columns: {unwanted_df.columns.tolist()}")
                # Convert 'Unwanted' column to boolean
                unwanted_col = unwanted_df['Unwanted']
                logger.info(f"Unique values in 'Unwanted' column: {unwanted_col.unique()}")
                # Convert to bool: treat 'TRUE', True, 1 as True
                unwanted_bool = unwanted_col.apply(lambda x: str(x).strip().upper() == 'TRUE' or x is True or x == 1)
                unwanted_activities = unwanted_df[unwanted_bool]['Land Use'].tolist()
                
                # Clean up the strings (remove any whitespace)
                unwanted_activities = [activity.strip() for activity in unwanted_activities]
                
                logger.info(f"Loaded {len(unwanted_activities)} unwanted activities from Google Sheets")
                logger.debug(f"Unwanted activities: {unwanted_activities}")
                return unwanted_activities
            else:
                raise ValueError("Invalid Google Sheets URL")
        except Exception as e:
            logger.error(f"Failed to load unwanted activities: {str(e)}")
            raise

    def load_parcels(self, min_size: float = 50.0) -> None:
        """Load parcels from the database with initial filtering."""
        try:
            # Load unwanted activities if not already loaded
            if self.unwanted_activities is None:
                self.unwanted_activities = self.load_unwanted_activities()
            
            # Create initial filtered table
            with self.db_utils.engine.connect() as conn:
                if self.county:
                    # Single county case
                    table_name = f"parcels.{self.state}_{self.county}"
                    sql = f"""
                    CREATE TEMP TABLE parcels_filtered_initial AS
                    SELECT p.*
                    FROM {table_name} p
                    WHERE p.gisacre > {min_size}
                    AND p.usedesc NOT IN :unwanted_activities
                    """
                    conn.execute(text(sql), {"unwanted_activities": tuple(self.unwanted_activities)})
                else:
                    # All counties case - using standardized schema from rebuild script
                    # Get all counties for this state
                    result = conn.execute(text(f"""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'parcels' 
                        AND table_name LIKE '{self.state}_%'
                        ORDER BY table_name
                    """))
                    
                    counties = []
                    for row in result.fetchall():
                        table_name = row[0]
                        if table_name.startswith(f"{self.state}_"):
                            county = table_name[len(f"{self.state}_"):]
                            counties.append(county)
                    
                    # Create UNION query for all counties (safe with standardized schema)
                    union_queries = []
                    for county in counties:
                        table_name = f"parcels.{self.state}_{county}"
                        union_queries.append(f"""
                        SELECT p.*
                        FROM {table_name} p
                        WHERE p.gisacre > {min_size}
                        AND p.usedesc NOT IN :unwanted_activities
                        """)
                    
                    sql = f"""
                    CREATE TEMP TABLE parcels_filtered_initial AS
                    {' UNION ALL '.join(union_queries)}
                    """
                    conn.execute(text(sql), {"unwanted_activities": tuple(self.unwanted_activities)})
                
                conn.commit()
                
                # Verify the table was created
                result = conn.execute(text("SELECT COUNT(*) FROM parcels_filtered_initial"))
                count = result.scalar()
                logger.info(f"Parcel count in parcels_filtered_initial: {count}")
            
            # Load the filtered parcels
            self.filtered_parcels = gpd.read_postgis(
                "SELECT * FROM parcels_filtered_initial",
                self.db_utils.engine,
                geom_col='geom'
            )
            logger.info(f"Parcel count after initial filtering: {len(self.filtered_parcels)}")
            logger.info(f"Loaded {len(self.filtered_parcels)} parcels after initial filtering")
            
            # Update the filtered parcels table name
            self.filtered_parcels_table = 'parcels_filtered_initial'
            
        except Exception as e:
            logger.error(f"Error loading parcels: {e}")
            raise

    def filter_airports(self) -> None:
        """Filter parcels that intersect with airports using pure PostGIS."""
        logger.info("Filtering parcels that intersect with airports...")
        
        try:
            with self.db_utils.engine.connect() as conn:
                # Create filtered parcels table using PostGIS with a unique name
                temp_table = f"parcels_filtered_airports_{int(time.time())}"
                sql = f"""
                CREATE TEMP TABLE {temp_table} AS
                SELECT p.*
                FROM {self.filtered_parcels_table} p
                WHERE NOT EXISTS (
                    SELECT 1 
                    FROM other_gis.us_civil_airports a 
                    WHERE ST_Intersects(p.geom, a.geom)
                    AND a.FEATTYPE = 'Airport ground'
                );
                """
                
                # Execute the SQL
                conn.execute(text(sql))
                conn.commit()
                
                # Get count of filtered parcels
                result = conn.execute(text(f"SELECT COUNT(*) FROM {temp_table}"))
                count = result.scalar()
                logger.info(f"Created temporary {temp_table} with {count} rows")
                
                # Update the filtered parcels table name
                self.filtered_parcels_table = temp_table
                logger.info(f"Parcel count after airport filtering: {count}")
                logger.info(f"After airport filtering: {count} parcels")
                
        except Exception as e:
            logger.error(f"Error filtering parcels by airports: {e}")
            raise

    def filter_transmission_lines(self) -> None:
        """Filter parcels based on transmission line distance using pure PostGIS."""
        logger.info("Filtering parcels based on transmission line distance...")
        
        try:
            with self.db_utils.engine.connect() as conn:
                # Create filtered parcels table using PostGIS with a unique name
                temp_table = f"parcels_filtered_transmission_{int(time.time())}"
                sql = f"""
                CREATE TEMP TABLE {temp_table} AS
                SELECT p.*
                FROM {self.filtered_parcels_table} p
                WHERE p.et_distance <= {self.transmission_distance};
                """
                
                # Execute the SQL
                conn.execute(text(sql))
                conn.commit()
                
                # Get count of filtered parcels
                result = conn.execute(text(f"SELECT COUNT(*) FROM {temp_table}"))
                count = result.scalar()
                logger.info(f"Created temporary {temp_table} with {count} rows")
                
                # Update the filtered parcels table name
                self.filtered_parcels_table = temp_table
                logger.info(f"Parcel count after transmission line filtering: {count}")
                
        except Exception as e:
            logger.error(f"Error filtering parcels by transmission lines: {e}")
            raise

    def filter_power_provider(self, provider: Optional[str] = None) -> None:
        """Filter parcels by power provider using pure PostGIS.
        
        Args:
            provider: Name of the power provider to filter by. If None, no filtering is applied.
        """
        if provider is None:
            logger.info("No power provider filter selected, skipping provider filtering")
            return
            
        logger.info(f"Filtering parcels by power provider: {provider}")
        self.power_provider = provider
        
        try:
            with self.db_utils.engine.connect() as conn:
                # Create filtered parcels table using PostGIS with a unique name
                temp_table = f"filtered_parcels_{int(time.time())}"
                sql = f"""
                CREATE TEMP TABLE {temp_table} AS
                SELECT p.*
                FROM {self.filtered_parcels_table} p
                WHERE p.et_operator = '{provider}'
                OR p.est_operator = '{provider}';
                """
                
                # Execute the SQL
                conn.execute(text(sql))
                conn.commit()
                
                # Get count of filtered parcels
                result = conn.execute(text(f"SELECT COUNT(*) FROM {temp_table}"))
                count = result.scalar()
                logger.info(f"Created temporary {temp_table} with {count} rows")
                
                # Update the filtered parcels table name
                self.filtered_parcels_table = temp_table
                logger.info(f"Parcel count after power operator filtering: {count}")
                
        except Exception as e:
            logger.error(f"Error filtering parcels by power operator: {e}")
            raise

    def filter_roadway_distance(self, max_distance: Optional[float] = None) -> None:
        """Filter parcels based on roadway distance using the hwy_distance column.
        
        Args:
            max_distance: Maximum distance to roadway in meters. If None, will prompt user for input.
        """
        # First check if hwy_distance column exists
        try:
            with self.db_utils.engine.connect() as conn:
                # Check if hwy_distance column exists in the current filtered table
                result = conn.execute(text(f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = '{self.filtered_parcels_table}' 
                    AND column_name = 'hwy_distance'
                """))
                
                if not result.fetchone():
                    logger.warning("hwy_distance column not found in the data. Skipping roadway distance filter.")
                    return
                    
        except Exception as e:
            logger.error(f"Error checking for hwy_distance column: {e}")
            return
        
        if max_distance is None:
            # Prompt user for roadway distance filter
            print("\nRoadway Distance Filter")
            print("Do you want to filter parcels by distance to nearest roadway?")
            while True:
                choice = input("Enter 'y' for yes or 'n' for no: ").lower().strip()
                if choice in ['y', 'n']:
                    break
                print("Please enter 'y' or 'n'.")
            
            if choice == 'n':
                logger.info("Roadway distance filter skipped by user")
                return
            
            # Prompt for distance
            while True:
                try:
                    distance_input = input("Enter maximum distance to roadway in meters: ")
                    max_distance = float(distance_input)
                    if max_distance > 0:
                        break
                    print("Distance must be greater than 0.")
                except ValueError:
                    print("Please enter a valid number.")
        
        logger.info(f"Filtering parcels based on roadway distance: {max_distance} meters")
        
        try:
            with self.db_utils.engine.connect() as conn:
                # Create filtered parcels table using PostGIS with a unique name
                temp_table = f"parcels_filtered_roadway_{int(time.time())}"
                sql = f"""
                CREATE TEMP TABLE {temp_table} AS
                SELECT p.*
                FROM {self.filtered_parcels_table} p
                WHERE p.hwy_distance <= {max_distance};
                """
                
                # Execute the SQL
                conn.execute(text(sql))
                conn.commit()
                
                # Get count of filtered parcels
                result = conn.execute(text(f"SELECT COUNT(*) FROM {temp_table}"))
                count = result.scalar()
                logger.info(f"Created temporary {temp_table} with {count} rows")
                
                # Update the filtered parcels table name
                self.filtered_parcels_table = temp_table
                logger.info(f"Parcel count after roadway distance filtering: {count}")
                
        except Exception as e:
            logger.error(f"Error filtering parcels by roadway distance: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up temporary and filtered tables from the database."""
        try:
            with self.db_utils.engine.connect() as conn:
                # Drop the filtered parcels table if it exists
                if self.filtered_parcels_table:
                    conn.execute(text(f"DROP TABLE IF EXISTS {self.filtered_parcels_table}"))
                    logger.info(f"Cleaned up temporary table: {self.filtered_parcels_table}")
                conn.commit()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise

    def get_power_providers(self) -> List[str]:
        """Get list of unique power providers from the database."""
        try:
            if self.county:
                # Single county case
                table_name = f"parcels.{self.state}_{self.county}"
                with self.db_utils.engine.connect() as conn:
                    result = conn.execute(text(f"""
                        SELECT DISTINCT et_operator
                        FROM {table_name}
                        WHERE et_operator IS NOT NULL 
                        AND et_operator != ''
                        ORDER BY et_operator
                    """))
                    providers = [row[0].strip() for row in result.fetchall() if row[0]]
            else:
                # All counties case
                with self.db_utils.engine.connect() as conn:
                    # Get all counties for this state
                    result = conn.execute(text(f"""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'parcels' 
                        AND table_name LIKE '{self.state}_%'
                        ORDER BY table_name
                    """))
                    
                    counties = []
                    for row in result.fetchall():
                        table_name = row[0]
                        if table_name.startswith(f"{self.state}_"):
                            county = table_name[len(f"{self.state}_"):]
                            counties.append(county)
                    
                    # Query power providers from all counties
                    providers = set()
                    for county in counties:
                        table_name = f"parcels.{self.state}_{county}"
                        result = conn.execute(text(f"""
                            SELECT DISTINCT et_operator
                            FROM {table_name}
                            WHERE et_operator IS NOT NULL 
                            AND et_operator != ''
                        """))
                        county_providers = [row[0].strip() for row in result.fetchall() if row[0]]
                        providers.update(county_providers)
                    
                    providers = list(providers)
            
            # Log the providers we found
            logger.info(f"Found {len(providers)} unique power operators")
            logger.info(f"Available operators in data: {providers}")
            
            return sorted(providers)
        except Exception as e:
            logger.error(f"Error getting power operators: {e}")
            raise

    def get_available_states(self) -> List[str]:
        """Get list of available states from the database."""
        try:
            with self.db_utils.engine.connect() as conn:
                # Query to get all table names in the parcels schema
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'parcels' 
                    AND table_name LIKE '%_%'
                    ORDER BY table_name
                """))
                
                # Extract state codes from table names (format: state_county)
                states = set()
                for row in result.fetchall():
                    table_name = row[0]
                    if '_' in table_name:
                        state = table_name.split('_')[0]
                        states.add(state)
                
                states_list = sorted(list(states))
                logger.info(f"Found {len(states_list)} available states: {states_list}")
                return states_list
                
        except Exception as e:
            logger.error(f"Error getting available states: {e}")
            raise

    def get_available_counties(self, state: str) -> List[str]:
        """Get list of available counties for a given state from the database."""
        try:
            with self.db_utils.engine.connect() as conn:
                # Query to get all table names in the parcels schema for the given state
                result = conn.execute(text(f"""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'parcels' 
                    AND table_name LIKE '{state}_%'
                    ORDER BY table_name
                """))
                
                # Extract county names from table names (format: state_county)
                counties = []
                for row in result.fetchall():
                    table_name = row[0]
                    if table_name.startswith(f"{state}_"):
                        county = table_name[len(f"{state}_"):]
                        counties.append(county)
                
                counties_list = sorted(counties)
                logger.info(f"Found {len(counties_list)} available counties for state {state}: {counties_list}")
                return counties_list
                
        except Exception as e:
            logger.error(f"Error getting available counties for state {state}: {e}")
            raise

    def set_utility_filter(self, utility: str) -> None:
        """Set the utility filter and update the ranker."""
        self.utility_filter = utility
        if self.ranking_url and self.ranker:
            # Update the existing ranker's utility provider (sanitized for file paths)
            if utility:
                # Convert to lowercase and replace spaces with underscores
                sanitized = utility.lower().replace(' ', '_')
                # Remove or replace invalid characters for file paths
                sanitized = re.sub(r'[^a-z0-9_]', '', sanitized)
                # Ensure it doesn't start with a number
                if sanitized and sanitized[0].isdigit():
                    sanitized = 'provider_' + sanitized
                # Ensure it's not empty
                if not sanitized:
                    sanitized = 'unknown_provider'
                self.ranker.utility_provider = sanitized
            else:
                self.ranker.utility_provider = 'all_providers'
        elif self.ranking_url and not self.ranker:
            # Create a new ranker if one doesn't exist
            self.ranker = MCDARanker(self.ranking_url, self.state, self.county, utility)

    def save_results_to_db(self, table_name: str = None) -> None:
        """Save the filtered and ranked parcels to the results schema in PostgreSQL.
        
        Args:
            table_name: Optional name for the results table. If not provided, a name will be generated
                      based on state, county, and timestamp.
        """
        if hasattr(self, 'final_results_table') and self.final_results_table:
            logger.info(f"Results already saved to permanent table: {self.final_results_table}")
            return
        elif self.filtered_parcels is None:
            raise ValueError("No filtered parcels to save. Run the filtering pipeline first.")
        else:
            # Fallback to the old method if no permanent table was created
            self.db_utils.save_results_to_db(self.filtered_parcels, table_name, self.utility_filter)

    def rank_parcels_in_postgis(self) -> None:
        """Rank parcels using MCDA (Multi-Criteria Decision Analysis) in PostGIS."""
        try:
            # First, ensure we have the ranking data
            if not self.ranker:
                raise ValueError("No ranker initialized. Please set up the ranker with ranking URL.")
            
            # Load ranking data if not already loaded
            if self.ranker.ranking_data is None:
                self.ranker.load_ranking_data()
            
            # Ensure the results schema exists
            with self.db_utils.engine.connect() as conn:
                conn.execute(text("CREATE SCHEMA IF NOT EXISTS results"))
                conn.commit()
            
            # Calculate MCDA rankings using the new system
            final_table_name, ranking_stats = self.ranker.calculate_mcda_rankings(
                self.db_utils.engine, 
                self.filtered_parcels_table
            )
            
            # Read the ranked results from the permanent table
            self.filtered_parcels = gpd.read_postgis(
                f"SELECT * FROM {final_table_name}",
                self.db_utils.engine,
                geom_col='geom'
            )
            
            # Store the final table name for reference
            self.final_results_table = final_table_name
            
            # Log ranking statistics
            logger.info("MCDA Ranking Statistics:")
            logger.info(f"  Total parcels: {ranking_stats['total_parcels']}")
            logger.info(f"  Min score: {ranking_stats['min_score']:.2f}")
            logger.info(f"  Max score: {ranking_stats['max_score']:.2f}")
            logger.info(f"  Mean score: {ranking_stats['mean_score']:.2f}")
            logger.info(f"  Median score: {ranking_stats['median_score']:.2f}")
            logger.info(f"  Standard deviation: {ranking_stats['std_dev']:.2f}")
            
            logger.info("Top 5 parcels by MCDA ranking:")
            for i, parcel in enumerate(ranking_stats['top_parcels'], 1):
                logger.info(f"  {i}. Parcel {parcel['parcelnumb']}: Score {parcel['parcel_rank_normalized']:.2f}")
                logger.info(f"     Zoning: {parcel['zoning_type']}")
                logger.info(f"     Activity: {parcel['lbcs_activity_desc']}")
            
            # Clean up temporary tables
            with self.db_utils.engine.connect() as conn:
                conn.execute(text("""
                    DROP TABLE IF EXISTS zoning_rankings;
                    DROP TABLE IF EXISTS activity_rankings;
                    DROP TABLE IF EXISTS fema_rankings;
                """))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error ranking parcels in PostGIS: {e}")
            raise

    def calculate_drive_times(self) -> None:
        """Calculate drive times for parcels using isochrone data."""
        try:
            logger.info("Loading isochrone data from database")
            isochrone_query = """
                SELECT 
                    time,
                    ST_Transform(geom, 4326) as geom
                FROM other_gis.iso_50
            """
            try:
                isochrones = gpd.read_postgis(
                    isochrone_query,
                    self.db_utils.engine,
                    geom_col='geom'
                )
                logger.info(f"Loaded {len(isochrones)} isochrone polygons")
                
                logger.info("Calculating drive times...")
                # Create a temporary table for the filtered parcels
                self.filtered_parcels.to_postgis(
                    'temp_filtered_parcels',
                    self.db_utils.engine,
                    if_exists='replace',
                    index=False
                )
                # Ensure drive_time column exists
                with self.db_utils.engine.connect() as conn:
                    conn.execute(text("ALTER TABLE temp_filtered_parcels ADD COLUMN IF NOT EXISTS drive_time DOUBLE PRECISION;"))
                    conn.commit()
                # Calculate drive times using spatial join
                drive_time_query = """
                    WITH parcel_isochrone_join AS (
                        SELECT 
                            p.parcelnumb,
                            MIN(i.time) as drive_time
                        FROM temp_filtered_parcels p
                        JOIN other_gis.iso_50 i
                        ON ST_Intersects(p.geom, i.geom)
                        GROUP BY p.parcelnumb
                    )
                    UPDATE temp_filtered_parcels p
                    SET drive_time = j.drive_time
                    FROM parcel_isochrone_join j
                    WHERE p.parcelnumb = j.parcelnumb;
                """
                with self.db_utils.engine.connect() as conn:
                    conn.execute(text(drive_time_query))
                    # Set drive_time to 9999 for parcels that did not get a value (i.e., are outside isochrone areas)
                    conn.execute(text("UPDATE temp_filtered_parcels SET drive_time = 9999 WHERE drive_time IS NULL;"))
                    conn.commit()
                # Load the updated parcels
                self.filtered_parcels = gpd.read_postgis(
                    "SELECT * FROM temp_filtered_parcels",
                    self.db_utils.engine,
                    geom_col='geom'
                )
                # Clean up temporary table
                with self.db_utils.engine.connect() as conn:
                    conn.execute(text("DROP TABLE IF EXISTS temp_filtered_parcels"))
                    conn.commit()
            except Exception as e:
                logger.warning(f"Could not load isochrone data: {str(e)}. Using default drive time value.")
                # Add a default drive time column with a special value (9999)
                self.filtered_parcels['drive_time'] = 9999
                # Ensure the column is added to the database table
                with self.db_utils.engine.connect() as conn:
                    conn.execute(text(f"""
                        ALTER TABLE {self.filtered_parcels_table} 
                        ADD COLUMN IF NOT EXISTS drive_time DOUBLE PRECISION DEFAULT 9999;
                    """))
                    conn.commit()
                
            logger.info("Drive time calculation completed")
            
        except Exception as e:
            logger.error(f"Failed to calculate drive times: {str(e)}")
            raise

    def get_final_results_table(self) -> Optional[str]:
        """Get the name of the final results table if it exists.
        
        Returns:
            The full table name (including schema) if a permanent results table was created,
            None otherwise.
        """
        return getattr(self, 'final_results_table', None)

    def get_final_parcel_count(self) -> int:
        """Get the count of final parcels after all filtering and ranking.
        
        Returns:
            Number of parcels in the final results, or 0 if no parcels available.
        """
        if self.filtered_parcels is not None:
            return len(self.filtered_parcels)
        elif self.final_results_table:
            # Try to get count from the final results table
            try:
                with self.db_utils.engine.connect() as conn:
                    result = conn.execute(text(f"SELECT COUNT(*) FROM {self.final_results_table}"))
                    return result.scalar()
            except Exception as e:
                logger.error(f"Error getting final parcel count: {e}")
                return 0
        else:
            return 0

    def get_filtering_summary(self) -> Dict[str, Any]:
        """Get a summary of the filtering process and final results.
        
        Returns:
            Dictionary containing filtering statistics and final results info.
        """
        summary = {
            'final_parcel_count': self.get_final_parcel_count(),
            'final_results_table': self.final_results_table,
            'state': self.state,
            'county': self.county,
            'transmission_distance': self.transmission_distance,
            'power_provider': self.power_provider,
            'utility_filter': self.utility_filter
        }
        
        # Add ranking statistics if available
        if hasattr(self, 'ranking_data') and self.ranking_data is not None:
            summary['ranking_data_loaded'] = True
        else:
            summary['ranking_data_loaded'] = False
            
        return summary