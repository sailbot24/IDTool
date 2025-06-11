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
    def __init__(self, state: str, county: Optional[str] = None, data_dir: Optional[Union[str, Path]] = None, ranking_url: Optional[str] = None):
        """
        Initialize ParcelFilter.
        
        Args:
            state: Two-letter state code (e.g., 'az')
            county: County name (e.g., 'maricopa'). If None, will analyze all counties in the state.
            data_dir: Base directory containing data. If None, assumes 'data'
            ranking_url: URL to Google Sheets document containing ranking data
        """
        self.state = state.lower()
        self.county = county.lower() if county else None
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.con = None
        self.ranking_url = ranking_url
        self.ranker = ParcelRanker(ranking_url, self.state, self.county) if ranking_url else None
        self.isochrone_data = None
        self.filtered_parcels = None
        self.unwanted_activities = None
        self.utility_filter = None  # Will store the selected utility filter
        
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
        if self.county:
            return f"checkpoints.{self.state}_{self.county}_{checkpoint_name}"
        else:
            return f"checkpoints.{self.state}_all_{checkpoint_name}"
    
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
        # First check the geometry type
        type_query = f"""
            SELECT typeof(geom) as geom_type 
            FROM ({query}) as q 
            LIMIT 1
        """
        geom_type = self.con.execute(type_query).fetchone()[0].upper()
        
        # Convert to WKT based on the geometry type
        if geom_type == 'WKB_BLOB':
            wkt_query = f"""
                SELECT 
                    * EXCLUDE (geom),
                    ST_AsText(ST_GeomFromWKB(geom)) AS geom_wkt
                FROM ({query}) as q
            """
        else:  # GEOMETRY type
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
                
                # Filter for activities marked as unwanted (TRUE)
                unwanted_activities = unwanted_df[unwanted_df['Unwanted'] == 'TRUE']['Land Use'].tolist()
                
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

    def load_parcels(self, min_size: float = 50.0, unwanted: Optional[List[str]] = None) -> int:
        """Load parcels from the database with initial filtering."""
        if unwanted is None:
            if self.unwanted_activities is None:
                self.unwanted_activities = self.load_unwanted_activities()
            unwanted = self.unwanted_activities
        
        unwanted_string = ", ".join([f"'{w}'" for w in unwanted])
        
        logger.info(f"Loading parcels with size > {min_size} acres and excluding unwanted activities...")
        
        try:
            # Drop any existing temporary tables
            self.con.execute("DROP TABLE IF EXISTS parcels_filtered_initial")
            
            if self.county:
                # Single county analysis
                table_name = f"parcels.{self.state}_{self.county}"
                self.con.execute(f"""
                    CREATE TEMP TABLE parcels_filtered_initial AS 
                    SELECT 
                        * EXCLUDE (geom),
                        geom as geom,
                        COALESCE(sgisacre, ll_gisacre, deeded_acres) as gisacre
                    FROM {table_name}
                    WHERE COALESCE(gisacre, ll_gisacre, deeded_acres) > {min_size}
                    AND (lbcs_activity_desc IS NULL OR lbcs_activity_desc NOT IN ({unwanted_string}))
                """)
                
                # Read into geopandas using helper
                self.filtered_parcels = self._table_to_gdf("SELECT * FROM parcels_filtered_initial")
                
            else:
                # State-wide analysis - process each county separately
                # Get list of counties from the database
                counties = self.con.execute(f"""
                    SELECT DISTINCT 
                        REPLACE(table_name, '{self.state}_', '') as county
                    FROM information_schema.tables 
                    WHERE table_schema = 'parcels' 
                    AND table_name LIKE '{self.state}_%'
                """).fetchall()
                
                # Process each county and store results
                county_results = []
                for county in counties:
                    county_name = county[0]
                    logger.info(f"Processing county: {county_name}")
                    
                    # Create temporary table for this county
                    table_name = f"parcels.{self.state}_{county_name}"
                    self.con.execute(f"""
                        CREATE TEMP TABLE county_parcels AS 
                        SELECT 
                            * EXCLUDE (geom),
                            geom as geom,
                            COALESCE(gisacre, ll_gisacre, deeded_acres) as gisacre
                        FROM {table_name}
                        WHERE COALESCE(gisacre, ll_gisacre, deeded_acres) > {min_size}
                        AND (lbcs_activity_desc IS NULL OR lbcs_activity_desc NOT IN ({unwanted_string}))
                    """)
                    
                    # Read into geopandas
                    county_gdf = self._table_to_gdf("SELECT * FROM county_parcels")
                    if not county_gdf.empty:
                        county_results.append(county_gdf)
                    
                    # Clean up temporary table
                    self.con.execute("DROP TABLE IF EXISTS county_parcels")
                
                # Merge all county results
                if county_results:
                    merged = pd.concat(county_results, ignore_index=True)
                    # Check for geometry column
                    if 'geometry' not in merged.columns:
                        logger.error("Merged state GeoDataFrame is missing 'geometry' column after concat. Columns: %s", merged.columns.tolist())
                        logger.error("Sample of merged DataFrame:\n%s", merged.head())
                    else:
                        # Check for valid geometry
                        if merged['geometry'].isna().any() or merged['geometry'].is_empty.any():
                            logger.error("Merged state GeoDataFrame has missing or empty geometries. Sample:\n%s", merged[merged['geometry'].isna() | merged['geometry'].is_empty].head())
                        # Ensure GeoDataFrame with correct CRS
                        merged = gpd.GeoDataFrame(merged, geometry='geometry', crs=4326)
                        logger.info("Merged state GeoDataFrame has %d features. Sample:\n%s", len(merged), merged.head())
                    self.filtered_parcels = merged

                    # Write to DuckDB as parcels_filtered_initial using WKT
                    merged['geom_wkt'] = merged.geometry.apply(lambda x: x.wkt)
                    merged['geom_wkt'] = merged['geom_wkt'].astype(str)  # Ensure string dtype
                    df = merged.drop(columns=['geometry'])
                    self.con.execute("DROP TABLE IF EXISTS parcels_filtered_initial")
                    self.con.register('temp_parcels_df', df)
                    self.con.execute("""
                        CREATE TEMP TABLE parcels_filtered_initial AS
                        SELECT 
                            * EXCLUDE (geom_wkt),
                            ST_GeomFromText(geom_wkt) as geom
                        FROM temp_parcels_df
                    """)
                    self.con.unregister('temp_parcels_df')

                    # Debug: Read back from DuckDB and log geometry info
                    gdf_check = self._table_to_gdf("SELECT * FROM parcels_filtered_initial")
                    logger.info("After round-trip to DuckDB (WKT), number of features: %d", len(gdf_check))
                    logger.info("Sample after round-trip:\n%s", gdf_check.head())
                    logger.info("Bounds after round-trip: %s", gdf_check.total_bounds)
                    logger.info("Geometry types: %s", gdf_check.geometry.geom_type.value_counts())
                    logger.info("Number of empty geometries: %d", gdf_check.geometry.is_empty.sum())
                    logger.info("Number of NaN geometries: %d", gdf_check.geometry.isna().sum())
            
            count = len(self.filtered_parcels)
            logger.info(f"Loaded {count} parcels after initial filtering")
            return count
            
        except Exception as e:
            # Clean up temporary tables in case of error
            self.con.execute("DROP TABLE IF EXISTS parcels_filtered_initial")
            self.con.execute("DROP TABLE IF EXISTS county_parcels")
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
            
            # Update the ranker with the selected provider
            if self.ranker:
                self.ranker.utility_provider = provider.lower().replace(' ', '_')
            
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
        """Get list of unique power providers from the database."""
        try:
            # Get list of counties to check
            if self.county:
                counties = [self.county]
            else:
                counties = self.con.execute(f"""
                    SELECT DISTINCT 
                        REPLACE(table_name, '{self.state}_', '') as county
                    FROM information_schema.tables 
                    WHERE table_schema = 'parcels' 
                    AND table_name LIKE '{self.state}_%'
                """).fetchall()
                counties = [c[0] for c in counties]
            
            # Get unique power providers across all counties
            providers = set()
            for county in counties:
                table_name = f"parcels.{self.state}_{county}"
                county_providers = self.con.execute(f"""
                    SELECT DISTINCT et_operator 
                    FROM {table_name}
                    WHERE et_operator IS NOT NULL
                """).fetchall()
                providers.update(p[0] for p in county_providers if p[0])
            
            return sorted(list(providers))
        except Exception as e:
            logger.error(f"Error getting power providers: {str(e)}")
            raise

    def set_utility_filter(self, utility: str) -> None:
        """Set the utility filter and initialize the ranker."""
        self.utility_filter = utility
        if self.ranking_url:
            self.ranker = ParcelRanker(self.ranking_url, self.state, self.county, self.utility_filter)