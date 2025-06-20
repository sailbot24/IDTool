import psycopg2
import logging
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from .ranking import ParcelRanker
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
        self.ranker = ParcelRanker(ranking_url, self.state, self.county) if ranking_url else None
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
                # Get the correct table name with schema
                table_name = f"parcels.{self.state}_{self.county}"
                
                # Create the filtered table
                conn.execute(text(f"""
                    CREATE TEMP TABLE parcels_filtered_initial AS
                    SELECT p.*
                    FROM {table_name} p
                    WHERE p.gisacre > {min_size}
                    AND p.usedesc NOT IN :unwanted_activities
                """), {"unwanted_activities": tuple(self.unwanted_activities)})
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
            # Get the source table name
            table_name = f"parcels.{self.state}_{self.county}" if self.county else None
            if not table_name:
                raise ValueError("County must be specified to get power operators")
                
            with self.db_utils.engine.connect() as conn:
                result = conn.execute(text(f"""
                    SELECT DISTINCT et_operator
                    FROM {table_name}
                    WHERE et_operator IS NOT NULL 
                    AND et_operator != ''
                    ORDER BY et_operator
                """))
                providers = [row[0].strip() for row in result.fetchall() if row[0]]
                
                # Log the providers we found
                logger.info(f"Found {len(providers)} unique power operators")
                logger.info(f"Available operators in data: {providers}")
                
                return sorted(providers)
        except Exception as e:
            logger.error(f"Error getting power operators: {e}")
            raise

    def set_utility_filter(self, utility: str) -> None:
        """Set the utility filter and initialize the ranker."""
        self.utility_filter = utility
        if self.ranking_url:
            self.ranker = ParcelRanker(self.ranking_url, self.state, self.county, self.utility_filter)

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
        """Rank parcels in PostGIS using various criteria."""
        try:
            # First, ensure we have the ranking data
            if not self.ranker:
                raise ValueError("No ranker initialized. Please set up the ranker with ranking URL.")
            
            # Load ranking data if not already loaded
            if not self.ranker.ranking_data is not None:
                self.ranker.load_ranking_data()
            
            # Create a temporary table for the ranking mappings
            with self.db_utils.engine.connect() as conn:
                # Ensure the results schema exists
                conn.execute(text("CREATE SCHEMA IF NOT EXISTS results"))
                conn.commit()
                
                # Create ranking mapping tables from Google Sheets data
                ranking_data = self.ranker.ranking_data
                
                # Create zoning rankings table
                zoning_values = []
                for _, row in ranking_data.iterrows():
                    if pd.notna(row['zoning_type']):
                        zoning_values.append(f"('{row['zoning_type']}', '{row['zoning_subtype']}', {row['Zoning Ranking']}, {row['Zoning Subtype Ranking']})")
                
                conn.execute(text(f"""
                    CREATE TEMP TABLE zoning_rankings AS
                    SELECT 
                        zoning_type,
                        zoning_subtype,
                        zoning_ranking,
                        zoning_subtype_ranking
                    FROM (VALUES {','.join(zoning_values)}) 
                    AS t(zoning_type, zoning_subtype, zoning_ranking, zoning_subtype_ranking);
                """))
                
                # Create activity rankings table
                activity_values = []
                for _, row in ranking_data.iterrows():
                    if pd.notna(row['lbcs_activity_desc']):
                        # Handle NULL values properly in PostgreSQL
                        site_desc = f"'{row['lbcs_site_desc']}'" if pd.notna(row['lbcs_site_desc']) else 'NULL'
                        ownership_desc = f"'{row['lbcs_ownership_desc']}'" if pd.notna(row['lbcs_ownership_desc']) else 'NULL'
                        activity_rank = row['Land Activity Ranking'] if pd.notna(row['Land Activity Ranking']) else 'NULL'
                        site_rank = row['Site Descirption Ranking'] if pd.notna(row['Site Descirption Ranking']) else 'NULL'
                        ownership_rank = row['Ownership Ranking'] if pd.notna(row['Ownership Ranking']) else 'NULL'
                        
                        # Escape single quotes in string values
                        activity_desc = row['lbcs_activity_desc'].replace("'", "''")
                        
                        activity_values.append(f"('{activity_desc}', {site_desc}, {ownership_desc}, {activity_rank}, {site_rank}, {ownership_rank})")
                
                conn.execute(text(f"""
                    CREATE TEMP TABLE activity_rankings AS
                    SELECT 
                        lbcs_activity_desc,
                        lbcs_site_desc,
                        lbcs_ownership_desc,
                        activity_ranking,
                        site_ranking,
                        ownership_ranking
                    FROM (VALUES {','.join(activity_values)}) 
                    AS t(lbcs_activity_desc, lbcs_site_desc, lbcs_ownership_desc, 
                         activity_ranking, site_ranking, ownership_ranking);
                """))
                
                # Create FEMA rankings table
                fema_values = []
                for _, row in ranking_data.iterrows():
                    if pd.notna(row['fema_nri_risk_ranking']):
                        fema_values.append(f"('{row['fema_nri_risk_ranking']}', '{row['fema_flood_zone']}', '{row['fema_flood_zone_subtype']}', {row['Fema NRI Flood Risk Ranking']}, {row['Fema Flood Zone Ranking']}, {row['Fema Flood Zone Subtype Ranking']})")
                
                conn.execute(text(f"""
                    CREATE TEMP TABLE fema_rankings AS
                    SELECT 
                        fema_nri_risk_rating,
                        fema_flood_zone,
                        fema_flood_zone_subtype,
                        fema_nri_ranking,
                        flood_zone_ranking,
                        flood_subtype_ranking
                    FROM (VALUES {','.join(fema_values)}) 
                    AS t(fema_nri_risk_rating, fema_flood_zone, fema_flood_zone_subtype,
                         fema_nri_ranking, flood_zone_ranking, flood_subtype_ranking);
                """))
                
                # Create the ranked parcels table with all calculations
                conn.execute(text(f"""
                    CREATE TEMP TABLE ranked_parcels AS
                    WITH normalized_values AS (
                        SELECT 
                            p.*,
                            -- Normalize gisacre (0-10 range)
                            CASE 
                                WHEN p.gisacre IS NULL THEN 0
                                ELSE (p.gisacre - MIN(p.gisacre) OVER ()) / 
                                     (MAX(p.gisacre) OVER () - MIN(p.gisacre) OVER ()) * 10
                            END as gisacre_norm,
                            
                            -- Normalize transmission line distance (0-10 range, inverted)
                            CASE 
                                WHEN p.et_distance IS NULL THEN 0
                                ELSE 10 - ((p.et_distance - MIN(p.et_distance) OVER ()) / 
                                         (MAX(p.et_distance) OVER () - MIN(p.et_distance) OVER ()) * 10)
                            END as trans_line_distance_norm,
                            
                            -- Normalize drive time (0-10 range)
                            CASE 
                                WHEN p.drive_time IS NULL THEN 0
                                ELSE (p.drive_time - MIN(p.drive_time) OVER ()) / 
                                     (MAX(p.drive_time) OVER () - MIN(p.drive_time) OVER ()) * 10
                            END as norm_dt,
                            
                            -- Calculate building cover percentage
                            CASE 
                                WHEN p.ll_gissqft = 0 OR p.ll_gissqft IS NULL THEN 0
                                ELSE p.ll_bldg_footprint_sqft / p.ll_gissqft
                            END as building_cover_prec
                        FROM {self.filtered_parcels_table} p
                    ),
                    building_cover_normalized AS (
                        SELECT 
                            *,
                            -- Normalize building cover (1-10 range, inverted)
                            CASE 
                                WHEN building_cover_prec IS NULL THEN 5
                                WHEN MIN(building_cover_prec) OVER () = MAX(building_cover_prec) OVER () THEN 1
                                ELSE 10 - ((building_cover_prec - MIN(building_cover_prec) OVER ()) / 
                                         (MAX(building_cover_prec) OVER () - MIN(building_cover_prec) OVER ()) * 9 + 1)
                            END as building_cover_prec_invert
                        FROM normalized_values
                    ),
                    mapped_rankings AS (
                        SELECT 
                            b.*,
                            COALESCE(z.zoning_ranking, 0) as zoning_ranking,
                            COALESCE(z.zoning_subtype_ranking, 0) as zoning_subtype_ranking,
                            COALESCE(a.activity_ranking, 0) as activity_ranking,
                            COALESCE(a.site_ranking, 0) as site_ranking,
                            COALESCE(a.ownership_ranking, 0) as ownership_ranking,
                            COALESCE(f.fema_nri_ranking, 0) as fema_nri_ranking,
                            COALESCE(f.flood_zone_ranking, 0) as flood_zone_ranking,
                            COALESCE(f.flood_subtype_ranking, 0) as flood_subtype_ranking
                        FROM building_cover_normalized b
                        LEFT JOIN zoning_rankings z 
                            ON b.zoning_type = z.zoning_type 
                            AND b.zoning_subtype = z.zoning_subtype
                        LEFT JOIN activity_rankings a 
                            ON b.lbcs_activity_desc = a.lbcs_activity_desc
                            AND b.lbcs_site_desc = a.lbcs_site_desc
                            AND b.lbcs_ownership_desc = a.lbcs_ownership_desc
                        LEFT JOIN fema_rankings f 
                            ON b.fema_nri_risk_rating = f.fema_nri_risk_rating
                            AND b.fema_flood_zone = f.fema_flood_zone
                            AND b.fema_flood_zone_subtype = f.fema_flood_zone_subtype
                    )
                    SELECT 
                        *,
                        -- Calculate final ranking using weights from ranking_data
                        (
                            {self.ranker.weights['zoning_subtype']} * (zoning_ranking + zoning_subtype_ranking) +
                            {self.ranker.weights['activity']} * (
                                site_ranking * 0.3 + 
                                activity_ranking * 0.3 + 
                                ownership_ranking * 0.2 + 
                                gisacre_norm * 0.1 + 
                                building_cover_prec_invert * 0.1
                            ) +
                            {self.ranker.weights['site']} * norm_dt +
                            {self.ranker.weights['fema_nri']} * (
                                fema_nri_ranking + 
                                flood_zone_ranking + 
                                flood_subtype_ranking
                            ) +
                            {self.ranker.weights['flood_zone']} * trans_line_distance_norm
                        ) as parcel_rank
                    FROM mapped_rankings
                """))
                
                # Generate table name for the final results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                final_table_name = f"results.{self.state}_{self.county}_{self.utility_filter}_{timestamp}"
                
                # Create the final ranked parcels table as a permanent table
                conn.execute(text(f"""
                    CREATE TABLE {final_table_name} AS
                    SELECT 
                        p.*,
                        CASE 
                            WHEN MIN(parcel_rank) OVER () = MAX(parcel_rank) OVER () THEN 10
                            ELSE (parcel_rank - MIN(parcel_rank) OVER ()) / 
                                 (MAX(parcel_rank) OVER () - MIN(parcel_rank) OVER ()) * 10
                        END as parcel_rank_normalized
                    FROM ranked_parcels p
                    ORDER BY parcel_rank_normalized DESC
                """))
                
                # Add spatial index to the permanent table
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS {final_table_name.split('.')[-1]}_geom_idx 
                    ON {final_table_name} 
                    USING GIST (geom)
                """))
                
                # Verify the table was created
                result = conn.execute(text(f"SELECT COUNT(*) FROM {final_table_name}"))
                count = result.scalar()
                logger.info(f"Created permanent table {final_table_name} with {count} rows")
                
                # Log the table structure
                columns_result = conn.execute(text(f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = '{final_table_name.split('.')[-1]}'
                    AND table_schema = 'results'
                    ORDER BY ordinal_position;
                """))
                columns = columns_result.fetchall()
                logger.info("Table structure:")
                for col in columns:
                    logger.info(f"Column: {col[0]}, Type: {col[1]}")
                
                # Log a sample row
                sample = conn.execute(text(f"SELECT * FROM {final_table_name} LIMIT 1")).fetchone()
                if sample:
                    logger.info("Sample row columns:")
                    for i, col in enumerate(columns):
                        logger.info(f"{col[0]}: {sample[i]}")
                
                conn.commit()
            
            # Read the ranked results from the permanent table
            self.filtered_parcels = gpd.read_postgis(
                f"SELECT * FROM {final_table_name}",
                self.db_utils.engine,
                geom_col='geom'
            )
            
            # Store the final table name for reference
            self.final_results_table = final_table_name
            
            # Clean up temporary tables
            with self.db_utils.engine.connect() as conn:
                conn.execute(text("""
                    DROP TABLE IF EXISTS zoning_rankings;
                    DROP TABLE IF EXISTS activity_rankings;
                    DROP TABLE IF EXISTS fema_rankings;
                    DROP TABLE IF EXISTS ranked_parcels;
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
                    ST_Transform(wkb_geometry, 4326) as geom
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
                        ON ST_Intersects(p.geom, i.wkb_geometry)
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