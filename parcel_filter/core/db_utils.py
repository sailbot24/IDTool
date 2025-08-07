import logging
import geopandas as gpd
import re
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from geoalchemy2 import Geometry
from datetime import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Global database connection manager that maintains a single connection
    throughout the application lifecycle.
    """
    _instance = None
    _engine = None
    _db_config = None
    
    def __new__(cls, db_config: Optional[Dict[str, str]] = None):
        """Singleton pattern to ensure only one database manager exists."""
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._db_config = db_config or {
                'host': 'localhost',
                'port': '5432',
                'database': 'LandAI',
                'user': 'postgres',
                'password': 'postgres'
            }
        return cls._instance
    
    def connect(self):
        """Establish database connection if not already connected."""
        if self._engine is None:
            try:
                self._engine = create_engine(
                    f"postgresql://{self._db_config['user']}@{self._db_config['host']}:{self._db_config['port']}/{self._db_config['database']}"
                )
                logger.info("Database connection established successfully")
            except Exception as e:
                logger.error(f"Failed to establish database connection: {e}")
                raise
    
    def get_engine(self):
        """Get the database engine, connecting if necessary."""
        if self._engine is None:
            self.connect()
        return self._engine
    
    def is_connected(self):
        """Check if database is connected."""
        return self._engine is not None
    
    def dispose(self):
        """Close the database connection."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures connection is closed."""
        self.dispose()

def sanitize_utility_name(utility_name: str) -> str:
    """
    Sanitize utility provider name for use in PostgreSQL table names.
    
    Args:
        utility_name: The original utility provider name
        
    Returns:
        Sanitized name safe for PostgreSQL identifiers
    """
    if not utility_name:
        return 'all_providers'
    
    # Convert to lowercase and replace spaces with underscores
    sanitized = utility_name.lower().replace(' ', '_')
    
    # Remove or replace invalid PostgreSQL identifier characters
    # Keep only letters, numbers, and underscores
    sanitized = re.sub(r'[^a-z0-9_]', '', sanitized)
    
    # Ensure it doesn't start with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = 'provider_' + sanitized
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = 'unknown_provider'
    
    return sanitized

class DatabaseUtils:
    def __init__(self, state: str, county: Optional[str] = None, 
                 db_config: Optional[Dict[str, str]] = None,
                 db_manager: Optional[DatabaseManager] = None):
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
            db_manager: Optional DatabaseManager instance. If not provided, will create one.
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
        self.db_manager = db_manager or DatabaseManager(self.db_config)
        self.engine = None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures connection is closed."""
        # Note: We don't dispose here anymore since the manager handles the connection lifecycle
        pass

    def connect(self):
        """Connect to PostgreSQL database using the manager."""
        if self.engine is None:
            self.engine = self.db_manager.get_engine()
            logger.info("Connected to PostgreSQL database via DatabaseManager")

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
                # Sanitize utility filter name for PostgreSQL table name
                if utility_filter:
                    # Convert to lowercase and replace spaces with underscores
                    utility_name = utility_filter.lower().replace(' ', '_')
                    # Remove or replace invalid PostgreSQL identifier characters
                    utility_name = re.sub(r'[^a-z0-9_]', '', utility_name)
                    # Ensure it doesn't start with a number
                    if utility_name and utility_name[0].isdigit():
                        utility_name = 'provider_' + utility_name
                    # Ensure it's not empty
                    if not utility_name:
                        utility_name = 'unknown_provider'
                else:
                    utility_name = 'all_providers'
                table_name = f"results.{self.state}_{self.county}_{utility_name}_{timestamp}"
            
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
                
            # Clean up old results tables after saving new one
            self.cleanup_old_results_tables(utility_filter)
                
        except Exception as e:
            logger.error(f"Error saving results to database: {str(e)}")
            raise

    def cleanup_old_results_tables(self, utility_filter: Optional[str] = None, keep_count: int = 2) -> None:
        """Clean up old results tables, keeping only the most recent ones.
        
        Args:
            utility_filter: Optional utility provider name to filter tables by
            keep_count: Number of most recent tables to keep (default: 2)
        """
        try:
            # Sanitize utility filter name for table name matching
            if utility_filter:
                utility_name = utility_filter.lower().replace(' ', '_')
                utility_name = re.sub(r'[^a-z0-9_]', '', utility_name)
                if utility_name and utility_name[0].isdigit():
                    utility_name = 'provider_' + utility_name
                if not utility_name:
                    utility_name = 'unknown_provider'
            else:
                utility_name = 'all_providers'
            
            # Build the table name pattern to match
            table_pattern = f"results.{self.state}_{self.county}_{utility_name}_%"
            
            with self.engine.connect() as conn:
                # Get all matching tables ordered by creation time (timestamp)
                query = text(f"""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'results' 
                    AND table_name LIKE :pattern
                    ORDER BY table_name DESC
                """)
                
                result = conn.execute(query, {"pattern": table_pattern.replace("results.", "")})
                tables = [row[0] for row in result.fetchall()]
                
                if len(tables) > keep_count:
                    # Get tables to delete (all except the most recent keep_count)
                    tables_to_delete = tables[keep_count:]
                    
                    logger.info(f"Found {len(tables)} results tables for {self.state}_{self.county}_{utility_name}")
                    logger.info(f"Keeping {keep_count} most recent tables, deleting {len(tables_to_delete)} old tables")
                    
                    # Delete old tables
                    for table_name in tables_to_delete:
                        try:
                            # Drop the table
                            conn.execute(text(f"DROP TABLE IF EXISTS results.{table_name}"))
                            logger.info(f"Deleted old results table: results.{table_name}")
                        except Exception as e:
                            logger.warning(f"Failed to delete table results.{table_name}: {e}")
                    
                    conn.commit()
                    logger.info(f"Cleanup completed. Kept {keep_count} most recent tables.")
                else:
                    logger.info(f"No cleanup needed. Only {len(tables)} tables found (keeping up to {keep_count})")
                    
        except Exception as e:
            logger.error(f"Error during results table cleanup: {str(e)}")
            # Don't raise the exception - cleanup failure shouldn't stop the main process
            pass

    def get_results_tables_info(self, utility_filter: Optional[str] = None) -> Dict[str, Any]:
        """Get information about existing results tables.
        
        Args:
            utility_filter: Optional utility provider name to filter tables by
            
        Returns:
            Dictionary containing table information
        """
        try:
            # Sanitize utility filter name for table name matching
            if utility_filter:
                utility_name = utility_filter.lower().replace(' ', '_')
                utility_name = re.sub(r'[^a-z0-9_]', '', utility_name)
                if utility_name and utility_name[0].isdigit():
                    utility_name = 'provider_' + utility_name
                if not utility_name:
                    utility_name = 'unknown_provider'
            else:
                utility_name = 'all_providers'
            
            # Build the table name pattern to match
            table_pattern = f"results.{self.state}_{self.county}_{utility_name}_%"
            
            with self.engine.connect() as conn:
                # Get all matching tables with their creation times
                query = text(f"""
                    SELECT 
                        table_name,
                        table_name as full_table_name
                    FROM information_schema.tables 
                    WHERE table_schema = 'results' 
                    AND table_name LIKE :pattern
                    ORDER BY table_name DESC
                """)
                
                result = conn.execute(query, {"pattern": table_pattern.replace("results.", "")})
                tables = []
                
                for row in result.fetchall():
                    table_name = row[0]
                    full_table_name = f"results.{table_name}"
                    
                    # Extract timestamp from table name
                    timestamp_part = table_name.split('_')[-1]
                    try:
                        # Parse the timestamp (format: YYYYMMDD_HHMMSS)
                        creation_time = datetime.strptime(timestamp_part, "%Y%m%d_%H%M%S")
                    except ValueError:
                        creation_time = None
                    
                    tables.append({
                        'table_name': table_name,
                        'full_table_name': full_table_name,
                        'creation_time': creation_time,
                        'timestamp': timestamp_part
                    })
                
                return {
                    'state': self.state,
                    'county': self.county,
                    'utility_provider': utility_name,
                    'total_tables': len(tables),
                    'tables': tables
                }
                
        except Exception as e:
            logger.error(f"Error getting results tables info: {str(e)}")
            return {
                'state': self.state,
                'county': self.county,
                'utility_provider': utility_name if 'utility_name' in locals() else 'unknown',
                'total_tables': 0,
                'tables': [],
                'error': str(e)
            }

    def get_power_providers(self) -> List[str]:
        """Get list of available power providers from the database.
        
        Returns:
            List of power provider names
        """
        try:
            with self.engine.connect() as conn:
                # Query to get unique power providers
                query = text("""
                    SELECT DISTINCT utility_provider 
                    FROM parcels 
                    WHERE utility_provider IS NOT NULL 
                    AND utility_provider != ''
                    ORDER BY utility_provider
                """)
                
                result = conn.execute(query)
                providers = [row[0] for row in result.fetchall()]
                
                logger.info(f"Found {len(providers)} power providers")
                return providers
                
        except Exception as e:
            logger.error(f"Error getting power providers: {str(e)}")
            return [] 