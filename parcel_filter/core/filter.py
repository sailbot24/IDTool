import duckdb
import logging
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from .ranking import ParcelRanker
import os
import geopandas as gpd
import numpy as np
from shapely.wkb import loads as wkb_loads
from shapely import wkt

logger = logging.getLogger(__name__)

class ParcelFilter:
    def __init__(self, state: str, county: str, data_dir: Optional[Union[str, Path]] = None, ranking_url: Optional[str] = None):
        """
        Initialize ParcelFilter.
        
        Args:
            state: Two-letter state code (e.g., 'az')
            county: County name (e.g., 'maricopa')
            data_dir: Base directory containing data. If None, assumes 'data'
            ranking_url: URL to Google Sheets document containing ranking data
        """
        self.state = state.lower()
        self.county = county.lower()
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.con = None
        self.ranker = ParcelRanker(ranking_url) if ranking_url else None
        self.isochrone_data = None
        self.filtered_parcels = None
        
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures connection is closed."""
        if self.con is not None:
            logger.debug("Closing database connection")
            self.con.close()
            self.con = None

    def _ensure_geometry_type(self, table_name: str) -> None:
        """
        Ensures that the geometry column in the given table is of type GEOMETRY.
        If it's WKB_BLOB, converts it to GEOMETRY type using DuckDB's spatial functions.
        
        Args:
            table_name: Fully qualified table name (e.g., 'schema.table')
        """
        try:
            # Ensure spatial extension is loaded
            self.con.execute("LOAD spatial;")
            
            # Check current column type
            result = self.con.execute(f"SELECT typeof(geom) as geom_type FROM {table_name} LIMIT 1").fetchone()
            if result is None:
                raise ValueError(f"No rows found in table {table_name}")
                
            current_type = result[0].upper()
            
            if current_type == 'WKB_BLOB':
                logger.info(f"Converting geometry column in {table_name} from WKB_BLOB to GEOMETRY type")
                self.con.execute(f"""
                    ALTER TABLE {table_name}
                    ALTER COLUMN geom TYPE GEOMETRY 
                    USING ST_GeomFromWKB(geom)
                """)
                
                # Verify conversion
                result = self.con.execute(f"SELECT typeof(geom) as geom_type FROM {table_name} LIMIT 1").fetchone()
                if result[0].upper() != 'GEOMETRY':
                    raise RuntimeError(f"Geometry column conversion failed for {table_name}. Type is {result[0]}, expected GEOMETRY")
                
                logger.debug(f"Successfully converted geometry column in {table_name} to GEOMETRY type")
            elif current_type != 'GEOMETRY':
                raise ValueError(f"Unexpected geometry column type in {table_name}: {current_type}")
                
        except Exception as e:
            logger.error(f"Error converting geometry column in {table_name}: {e}")
            raise

    def connect(self):
        """Establish connection to DuckDB and configure it."""
        if self.con is None:
            db_path = self.data_dir / "LandAI.ddb"
            logger.debug(f"Connecting to DuckDB at {db_path}")
            self.con = duckdb.connect(str(db_path))
            self.con.execute("SET memory_limit='30GB'")
            self.con.execute("SET threads TO 8")
            self.con.execute("SET enable_progress_bar=true")
            
            # Load spatial extension
            logger.debug("Loading spatial extension")
            try:
                self.con.execute("INSTALL spatial;")
            except Exception as e:
                logger.debug(f"Spatial extension already installed: {e}")
                
            self.con.execute("LOAD spatial;")
            
            # Create checkpoints schema if it doesn't exist
            self.con.execute("CREATE SCHEMA IF NOT EXISTS checkpoints;")
        else:
            logger.debug("Using existing database connection")
    
    def get_checkpoint_name(self, checkpoint_name: str) -> str:
        """Get the full checkpoint table name in the checkpoints schema."""
        return f"checkpoints.{self.state}_{self.county}_{checkpoint_name}"
    
    def save_checkpoint(self, table_name: str, checkpoint_name: str) -> None:
        """Save a table to a checkpoint in the DuckDB schema."""
        full_checkpoint_name = self.get_checkpoint_name(checkpoint_name)
        logger.info(f"Saving checkpoint: {checkpoint_name}")
        self.con.execute(f"""
            CREATE OR REPLACE TABLE {full_checkpoint_name} AS
            SELECT * FROM {table_name}
        """)
    
    def load_checkpoint(self, checkpoint_name: str, table_name: str) -> bool:
        """
        Load a checkpoint from the DuckDB schema.
        Returns True if checkpoint was loaded, False if it doesn't exist.
        """
        full_checkpoint_name = self.get_checkpoint_name(checkpoint_name)
        
        # Check if checkpoint exists
        result = self.con.execute(f"""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_schema = 'checkpoints' 
            AND table_name = '{self.state}_{self.county}_{checkpoint_name}'
        """).fetchone()[0]
        
        if result == 0:
            return False
            
        logger.info(f"Loading checkpoint: {checkpoint_name}")
        self.con.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT * FROM {full_checkpoint_name}
        """)
        return True
    
    def _table_to_gdf(self, query: str) -> gpd.GeoDataFrame:
        """Convert a DuckDB table query result to a GeoDataFrame.
        
        Args:
            query: SQL query that includes a geometry column
            
        Returns:
            GeoDataFrame with geometry column
            
        Raises:
            ValueError: If geometry column is missing
        """
        # Convert GEOMETRY to WKT in DuckDB
        wkt_query = f"""
            SELECT 
                * EXCLUDE (geom),
                ST_AsText(geom) AS geom_wkt
            FROM ({query}) as q
        """
        df = self.con.execute(wkt_query).df()
        if 'geom_wkt' not in df.columns:
            raise ValueError("Missing geometry column in query result")
        # Create geometry column from WKT
        df['geometry'] = gpd.GeoSeries.from_wkt(df['geom_wkt'])
        # Drop the original geom_wkt column
        df = df.drop(columns=['geom_wkt'])
        # Create GeoDataFrame with the geometry column
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=4326)
        gdf = gdf.to_crs(5070)
        return gdf

    def calculate_drive_times(self) -> None:
        """Calculate drive times for each parcel based on isochrone data."""
        if self.isochrone_data is None:
            self.load_isochrone_data(self.data_dir)
            
        if self.filtered_parcels is None:
            raise ValueError("No parcels loaded. Call load_parcels first.")
            
        logger.info("Calculating drive times...")
        
        try:
            # Ensure we have the correct input table
            input_table = "parcels_filtered_transmission"
            if not self.con.execute(f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{input_table}'").fetchone()[0]:
                logger.warning(f"Table {input_table} not found, using parcels_filtered_airports instead")
                input_table = "parcels_filtered_airports"
            
            # Create a single temporary table for the final result
            self.con.execute(f"""
                CREATE TEMP TABLE parcels_with_drive_times AS
                WITH sorted_isochrones AS (
                    SELECT 
                        value,
                        ST_Force2D(geom) as geom
                    FROM other_gis.iso_50
                    ORDER BY value
                ),
                parcel_geoms AS (
                    SELECT 
                        p.*,
                        p.geom as geom_2d
                    FROM {input_table} p
                )
                SELECT 
                    p.* EXCLUDE (geom_2d),
                    COALESCE(
                        (SELECT MIN(i.value)
                         FROM sorted_isochrones i
                         WHERE ST_Intersects(p.geom_2d, i.geom)),
                        0
                    ) as drive_time
                FROM parcel_geoms p
            """)
            
            # Save checkpoint
            self.save_checkpoint("parcels_with_drive_times", "drive_times")
            
            # Read into geopandas
            self.filtered_parcels = self._table_to_gdf("SELECT * FROM parcels_with_drive_times")
            
            # Clean up temporary tables
            self.con.execute("DROP TABLE IF EXISTS parcels_with_drive_times")
            self.con.execute("DROP TABLE IF EXISTS parcels_filtered_transmission")
            
        except Exception as e:
            # Clean up temporary tables in case of error
            self.con.execute("DROP TABLE IF EXISTS parcels_with_drive_times")
            self.con.execute("DROP TABLE IF EXISTS parcels_filtered_transmission")
            logger.error(f"Error calculating drive times: {str(e)}")
            raise

    def load_parcels(self, min_size: float = 50.0, unwanted: Optional[List[str]] = None) -> int:
        """
        Load and filter parcels based on size and unwanted activities.
        
        Args:
            min_size: Minimum parcel size in acres
            unwanted: List of unwanted activity descriptions to exclude
            
        Returns:
            Number of parcels after filtering
            
        Raises:
            ValueError: If geometry column is missing
        """
        if unwanted is None:
            unwanted = [
                "Military base",
                "Emergency response or public-safety-related",
                "School or library",
                "Social, cultural, or religious assembly",
                "Power generation, control, monitor, or distribution",
                "Trains or other rail movement",
                "Activities associated with utilities (water, sewer, power, etc.)", 
                "Promenading and other activities in parks",
                "Health care, medical, or treatment"
            ]
        
        # Format unwanted activities for SQL query
        unwanted_string = ", ".join([f"'{activity}'" for activity in unwanted])
        
        logger.info(f"Loading parcels with size > {min_size} acres and excluding unwanted activities...")
        
        # Create initial filtered table
        logger.debug("Creating filtered parcels table")
        table_name = f"parcels.{self.state}_{self.county}"
        try:
            # Drop any existing temporary tables
            self.con.execute("DROP TABLE IF EXISTS parcels_filtered_initial")
            
            # Step 1: Initial filter by size and landuse, using geometry as is
            # Use COALESCE to handle missing acreage columns
            self.con.execute(f"""
                CREATE TEMP TABLE parcels_filtered_initial AS 
                SELECT 
                    * EXCLUDE (geom),
                    geom as geom,
                    COALESCE(gisacre, ll_gisacre, deeded_acres) as gisacre,
                    lbcs_activity_desc as lbcs_activity_desc
                FROM {table_name}
                WHERE COALESCE(gisacre, ll_gisacre, deeded_acres) > {min_size}
                AND (lbcs_activity_desc IS NULL OR lbcs_activity_desc NOT IN ({unwanted_string}))
            """)
            
            # Ensure geometry type is correct
            self._ensure_geometry_type("parcels_filtered_initial")
            
            # Read into geopandas using helper
            self.filtered_parcels = self._table_to_gdf("SELECT * FROM parcels_filtered_initial")
            
            count = len(self.filtered_parcels)
            logger.info(f"Loaded {count} parcels after initial filtering")
            return count
            
        except Exception as e:
            # Clean up temporary table in case of error
            self.con.execute("DROP TABLE IF EXISTS parcels_filtered_initial")
            logger.error(f"Error loading parcels: {str(e)}")
            raise

    def filter_airports(self) -> int:
        """
        Filter out parcels that intersect with airports.
        
        Returns:
            Number of parcels after airport filtering
            
        Raises:
            ValueError: If geometry column is missing
        """
        logger.info("Filtering parcels that intersect with airports...")
        
        try:
            # Drop any existing temporary tables
            self.con.execute("DROP TABLE IF EXISTS parcels_filtered_airports")
            
            # Step 2: Filter by airport intersection using LEFT JOIN anti-join
            self.con.execute("""
                CREATE TEMP TABLE parcels_filtered_airports AS
                SELECT 
                    p.* EXCLUDE (geom),
                    p.geom as geom
                FROM parcels_filtered_initial p
                LEFT JOIN other_gis.us_civil_airports a
                    ON ST_Intersects(p.geom, a.geom)
                    AND a.FEATTYPE = 'Airport ground'
                WHERE a.geom IS NULL
            """)
            
            # Read into geopandas using helper
            self.filtered_parcels = self._table_to_gdf("SELECT * FROM parcels_filtered_airports")
            
            count = len(self.filtered_parcels)
            logger.info(f"After airport filtering: {count} parcels")
            return count
            
        except Exception as e:
            # Clean up temporary table in case of error
            self.con.execute("DROP TABLE IF EXISTS parcels_filtered_airports")
            logger.error(f"Error filtering airports: {str(e)}")
            raise

    def filter_transmission_lines(self, distance_meters: float) -> None:
        """Filter parcels based on distance to transmission lines.

        Args:
            distance_meters: Maximum distance in meters from transmission lines.
            
        Raises:
            ValueError: If no parcels are loaded or if geometry column is missing
        """
        if self.filtered_parcels is None or self.filtered_parcels.empty:
            logger.warning("No parcels to filter")
            return

        logger.info("Filtering parcels by distance to transmission lines...")
        
        try:
            # Drop any existing temporary tables
            self.con.execute("DROP TABLE IF EXISTS parcels_filtered_transmission")
            
            # Filter using the pre-calculated et_distance column
            self.con.execute(f"""
                CREATE TEMP TABLE parcels_filtered_transmission AS
                SELECT 
                    p.* EXCLUDE (geom),
                    p.geom as geom
                FROM parcels_filtered_airports p
                WHERE p.et_distance <= {distance_meters}
            """)
            
            # Verify we actually filtered some parcels
            result = self.con.execute("SELECT COUNT(*) FROM parcels_filtered_transmission").fetchone()[0]
            if result == 0:
                logger.warning("No parcels found near transmission lines. This might indicate an issue with the data or query.")
            
            # Save checkpoint
            checkpoint_name = f"transmission_filtered_{int(distance_meters)}m"
            self.save_checkpoint("parcels_filtered_transmission", checkpoint_name)
            
            # Read into geopandas
            self.filtered_parcels = self._table_to_gdf("SELECT * FROM parcels_filtered_transmission")
            
            count = len(self.filtered_parcels)
            logger.info(f"After transmission line filtering: {count} parcels")
            
        except Exception as e:
            # Clean up temporary table in case of error
            self.con.execute("DROP TABLE IF EXISTS parcels_filtered_transmission")
            logger.error(f"Error filtering transmission lines: {str(e)}")
            raise

    def filter_power_provider(self, provider: Optional[str] = None) -> int:
        """Filter parcels based on power utility provider.

        Args:
            provider: Name of the power utility provider to filter by. If None, no filtering is applied.
            
        Returns:
            Number of parcels after provider filtering
            
        Raises:
            ValueError: If no parcels are loaded or if geometry column is missing
        """
        if self.filtered_parcels is None or self.filtered_parcels.empty:
            logger.warning("No parcels to filter")
            return 0

        if provider is None:
            logger.info("No power provider specified, skipping provider filter")
            return len(self.filtered_parcels)

        logger.info(f"Filtering parcels by power provider: {provider}")
        
        try:
            # Drop any existing temporary tables
            self.con.execute("DROP TABLE IF EXISTS parcels_filtered_provider")
            
            # Filter by power provider using either est_name or et_name
            self.con.execute(f"""
                CREATE TEMP TABLE parcels_filtered_provider AS
                SELECT 
                    p.* EXCLUDE (geom),
                    p.geom as geom
                FROM parcels_filtered_transmission p
                WHERE COALESCE(p.est_name, p.et_name) ILIKE '{provider}'
            """)
            
            # Verify we actually filtered some parcels
            result = self.con.execute("SELECT COUNT(*) FROM parcels_filtered_provider").fetchone()[0]
            if result == 0:
                logger.warning(f"No parcels found with provider '{provider}'. This might indicate an issue with the data or query.")
            
            # Save checkpoint
            checkpoint_name = f"provider_filtered_{provider.lower().replace(' ', '_')}"
            self.save_checkpoint("parcels_filtered_provider", checkpoint_name)
            
            # Read into geopandas
            self.filtered_parcels = self._table_to_gdf("SELECT * FROM parcels_filtered_provider")
            
            count = len(self.filtered_parcels)
            logger.info(f"After provider filtering: {count} parcels")
            return count
            
        except Exception as e:
            # Clean up temporary table in case of error
            self.con.execute("DROP TABLE IF EXISTS parcels_filtered_provider")
            logger.error(f"Error filtering power provider: {str(e)}")
            raise
    
    def load_isochrone_data(self, data_dir: Union[str, Path]) -> None:
        """Load isochrone data from the database."""
        logger.info("Loading isochrone data from database")
        logger.debug("Querying isochrone data from DuckDB")
        df = self.con.execute("""
            SELECT *, ST_AsText(geom) as geom_wkt FROM other_gis.iso_50
        """).df()
        logger.debug(f"Retrieved {len(df)} isochrone records")
        logger.debug("Converting WKT to geometry objects")
        df['geom'] = df['geom_wkt'].apply(wkt.loads)
        df = df.drop(columns=['geom_wkt'])
        # Always assign a default CRS (WGS84) and reproject to 5070
        self.isochrone_data = gpd.GeoDataFrame(df, geometry='geom', crs=4326).to_crs(5070)
        logger.info(f"Loaded {len(self.isochrone_data)} isochrone polygons")
        logger.debug(f"Isochrone columns: {self.isochrone_data.columns.tolist()}")

    def get_power_providers(self) -> List[str]:
        """Get a list of all unique power providers in the county.
        
        Returns:
            List of unique power provider names, sorted alphabetically
        """
        try:
            # Query for unique providers from both est_name and et_name columns
            providers = self.con.execute("""
                SELECT DISTINCT provider
                FROM (
                    SELECT est_name as provider FROM parcels.co_adams WHERE est_name IS NOT NULL
                    UNION
                    SELECT et_name as provider FROM parcels.co_adams WHERE et_name IS NOT NULL
                )
                WHERE provider IS NOT NULL AND provider != ''
                ORDER BY provider
            """).fetchall()
            
            # Extract provider names from result
            provider_list = [p[0] for p in providers]
            
            if not provider_list:
                logger.warning("No power providers found in the county data")
                return []
                
            logger.info(f"Found {len(provider_list)} unique power providers")
            return provider_list
            
        except Exception as e:
            logger.error(f"Error getting power providers: {str(e)}")
            raise