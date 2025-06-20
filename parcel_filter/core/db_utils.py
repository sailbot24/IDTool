import logging
import geopandas as gpd
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from geoalchemy2 import Geometry
from datetime import datetime
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class DatabaseUtils:
    def __init__(self, state: str, county: Optional[str] = None, 
                 db_config: Optional[Dict[str, str]] = None):
        """
        Initialize DatabaseUtils with PostgreSQL connection.
        
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
        """
        self.state = state.lower()
        self.county = county.lower() if county else None
        self.db_config = db_config or {
            'host': 'localhost',
            'port': '5432',
            'database': 'LandAI',
            'user': 'postgres', 
            'password': 'postgres'
        }
        self.engine = None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures connection is closed."""
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None

    def connect(self):
        """Connect to PostgreSQL database."""
        if self.engine is None:
            self.engine = create_engine(
                f"postgresql://{self.db_config['user']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            )
            logger.info("Connected to PostgreSQL database")

    def save_results_to_db(self, gdf: gpd.GeoDataFrame, table_name: Optional[str] = None, utility_filter: Optional[str] = None) -> None:
        """Save the filtered and ranked parcels to the results schema in PostgreSQL.
        
        Args:
            gdf: GeoDataFrame containing the results to save
            table_name: Optional name for the results table. If not provided, a name will be generated
                      based on state, county, and timestamp.
            utility_filter: Optional utility provider name to include in the table name
        """
        if gdf is None or gdf.empty:
            raise ValueError("No data to save")
            
        try:
            # Generate table name if not provided
            if table_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                table_name = f"results.{self.state}_{self.county}_{utility_filter}_{timestamp}"
            
            # Ensure the results schema exists
            with self.engine.connect() as conn:
                conn.execute(text("CREATE SCHEMA IF NOT EXISTS results"))
                conn.commit()
            
            # Write to PostGIS
            gdf.to_postgis(
                table_name,
                self.engine,
                schema='results',
                if_exists='replace',
                index=False
            )
            
            logger.info(f"Successfully saved results to {table_name}")
            
            # Add spatial index
            with self.engine.connect() as conn:
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS {table_name.split('.')[-1]}_geom_idx 
                    ON {table_name} 
                    USING GIST (geometry)
                """))
                conn.commit()
                logger.info(f"Created spatial index on {table_name}")
                
        except Exception as e:
            logger.error(f"Error saving results to database: {str(e)}")
            raise 