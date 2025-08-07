import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import requests
from sqlalchemy import text
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class MCDARanker:
    """
    Multi-Criteria Decision Analysis (MCDA) ranking system for parcels.
    
    This class implements a proper MCDA approach using PostGIS for efficient
    spatial data processing and ranking calculations.
    """
    
    def __init__(self, ranking_url: str, state: str, county: str, utility_provider: Optional[str] = None):
        """
        Initialize the MCDA ranker with ranking data from Google Sheets.
        
        Args:
            ranking_url: URL to the Google Sheets document containing ranking data
            state: Two-letter state code (e.g., 'az')
            county: County name (e.g., 'maricopa')
            utility_provider: Name of the utility provider filter (optional)
        """
        self.ranking_url = ranking_url
        self.ranking_data = None
        self.weights = None
        self.state = state.lower()
        self.county = county.lower() if county else None
        
        # Sanitize utility provider name for PostgreSQL table names
        if utility_provider:
            sanitized = self._sanitize_utility_name(utility_provider)
            self.utility_provider = sanitized
        else:
            self.utility_provider = 'all_providers'
    
    def _sanitize_utility_name(self, utility_name: str) -> str:
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
        sanitized = re.sub(r'[^a-z0-9_]', '', sanitized)
        
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = 'provider_' + sanitized
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = 'unknown_provider'
        
        return sanitized
    
    def load_ranking_data(self) -> None:
        """
        Load ranking data and weights from Google Sheets.
        
        Raises:
            ValueError: If the Google Sheets URL is invalid
            Exception: If there's an error loading the data
        """
        try:
            # Extract the document ID from the URL
            if 'docs.google.com/spreadsheets' in self.ranking_url:
                doc_id = self.ranking_url.split('/d/')[1].split('/')[0]
                
                # Load ranking data (sheet gid=843285247)
                ranking_url = f"https://docs.google.com/spreadsheets/d/{doc_id}/gviz/tq?tqx=out:csv&gid=843285247"
                self.ranking_data = pd.read_csv(ranking_url)
                
                # Load weights (sheet gid=1862912750)
                weights_url = f"https://docs.google.com/spreadsheets/d/{doc_id}/gviz/tq?tqx=out:csv&gid=1862912750"
                weights_df = pd.read_csv(weights_url)
                
                # Create weights dictionary from the first two columns
                self.weights = dict(zip(weights_df.iloc[:, 0], weights_df.iloc[:, 1]))
                
                # Validate the loaded data
                self._validate_ranking_data()
                
                logger.info(f"Successfully loaded ranking data with {len(self.ranking_data)} rows")
                logger.info(f"Loaded weights: {self.weights}")
            else:
                raise ValueError("Invalid Google Sheets URL")
        except Exception as e:
            logger.error(f"Failed to load ranking data: {str(e)}")
            raise
    
    def _validate_ranking_data(self) -> None:
        """
        Validate that the ranking data contains all required columns and reasonable values.
        
        Raises:
            ValueError: If validation fails
        """
        # Required columns for ranking data (updated for new structure)
        required_columns = [
            'zoning_type', 'zoning_subtype', 'lbcs_site_desc', 
            'fema_nri_risk_rating', 'fema_flood_zone', 'fema_flood_zone_subtype',
            'Zoning Ranking', 'Zoning Subtype Ranking', 'Site Descirption Ranking', 
            'Fema NRI Risk Ranking', 'Fema Flood Zone Ranking', 
            'Fema Flood Zone Subtype Ranking'
        ]
        
        missing_cols = [col for col in required_columns if col not in self.ranking_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in ranking data: {missing_cols}")
        
        # Validate weights (updated for new structure)
        required_weights = ['zoning', 'description', 'fema_nri', 'flood']
        missing_weights = [w for w in required_weights if w not in self.weights]
        if missing_weights:
            raise ValueError(f"Missing required weights: {missing_weights}")
        
        # Validate weight values are positive
        for weight_name, weight_value in self.weights.items():
            if not isinstance(weight_value, (int, float)) or weight_value < 0:
                raise ValueError(f"Weight '{weight_name}' must be a positive number, got: {weight_value}")
    
    def create_ranking_tables(self, conn) -> None:
        """
        Create temporary ranking lookup tables in PostgreSQL from Google Sheets data.
        
        Args:
            conn: Database connection
        """
        if self.ranking_data is None:
            raise ValueError("Ranking data not loaded. Call load_ranking_data() first.")
        
        # Create zoning rankings table
        self._create_zoning_rankings_table(conn)
        
        # Create site description rankings table
        self._create_site_rankings_table(conn)
        
        # Create FEMA rankings table
        self._create_fema_rankings_table(conn)
        
        logger.info("Created all ranking lookup tables")
    
    def _create_zoning_rankings_table(self, conn) -> None:
        """Create temporary table for zoning rankings."""
        zoning_values = []
        for _, row in self.ranking_data.iterrows():
            if pd.notna(row['zoning_type']):
                # Escape single quotes in string values
                zoning_type = row['zoning_type'].replace("'", "''")
                zoning_subtype = row['zoning_subtype'].replace("'", "''") if pd.notna(row['zoning_subtype']) else ''
                
                zoning_values.append(f"('{zoning_type}', '{zoning_subtype}', {row['Zoning Ranking']}, {row['Zoning Subtype Ranking']})")
        
        if zoning_values:
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
    
    def _create_site_rankings_table(self, conn) -> None:
        """Create temporary table for site description rankings."""
        site_values = []
        for _, row in self.ranking_data.iterrows():
            if pd.notna(row['lbcs_site_desc']):
                # Handle NULL values properly in PostgreSQL
                site_desc = f"'{row['lbcs_site_desc']}'" if pd.notna(row['lbcs_site_desc']) else 'NULL'
                site_rank = row['Site Descirption Ranking'] if pd.notna(row['Site Descirption Ranking']) else 'NULL'
                
                # Escape single quotes in string values
                site_desc_clean = row['lbcs_site_desc'].replace("'", "''") if pd.notna(row['lbcs_site_desc']) else ''
                
                site_values.append(f"('{site_desc_clean}', {site_rank})")
        
        if site_values:
            conn.execute(text(f"""
                CREATE TEMP TABLE site_rankings AS
                SELECT 
                    lbcs_site_desc,
                    site_ranking
                FROM (VALUES {','.join(site_values)}) 
                AS t(lbcs_site_desc, site_ranking);
            """))
    
    def _create_fema_rankings_table(self, conn) -> None:
        """Create temporary table for FEMA rankings."""
        fema_values = []
        for _, row in self.ranking_data.iterrows():
            if pd.notna(row['fema_nri_risk_rating']):
                # Escape single quotes in string values
                fema_nri = row['fema_nri_risk_rating'].replace("'", "''")
                flood_zone = row['fema_flood_zone'].replace("'", "''") if pd.notna(row['fema_flood_zone']) else ''
                flood_subtype = row['fema_flood_zone_subtype'].replace("'", "''") if pd.notna(row['fema_flood_zone_subtype']) else ''
                
                fema_values.append(f"('{fema_nri}', '{flood_zone}', '{flood_subtype}', {row['Fema NRI Risk Ranking']}, {row['Fema Flood Zone Ranking']}, {row['Fema Flood Zone Subtype Ranking']})")
        
        if fema_values:
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
    
    def calculate_mcda_rankings(self, engine, filtered_parcels_table: str) -> Tuple[str, Dict]:
        """
        Calculate MCDA rankings for parcels using PostGIS.
        
        Args:
            engine: SQLAlchemy engine for database connection
            filtered_parcels_table: Name of the table containing filtered parcels
            
        Returns:
            Tuple of (final_table_name, ranking_stats)
        """
        with engine.connect() as conn:
            # Create ranking lookup tables in the same connection
            self.create_ranking_tables(conn)
            
            # Generate table name for the final results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_table_name = f"results.{self.state}_{self.county}_{self.utility_provider}_{timestamp}"
            
            # Execute the MCDA ranking calculation
            self._execute_mcda_calculation(conn, filtered_parcels_table, final_table_name)
            
            # Get ranking statistics
            ranking_stats = self._get_ranking_statistics(conn, final_table_name)
            
            conn.commit()
            
            logger.info(f"Created MCDA ranking table: {final_table_name}")
            return final_table_name, ranking_stats
    
    def _execute_mcda_calculation(self, conn, filtered_parcels_table: str, final_table_name: str) -> None:
        """
        Execute the MCDA calculation using proper normalization and weighting.
        
        Args:
            conn: Database connection
            filtered_parcels_table: Name of the filtered parcels table
            final_table_name: Name for the final results table
        """
        # First check if we have enough parcels to rank
        count_result = conn.execute(text(f"SELECT COUNT(*) FROM {filtered_parcels_table}"))
        parcel_count = count_result.scalar()
        
        if parcel_count == 0:
            raise ValueError("No parcels to rank. All parcels were filtered out.")
        elif parcel_count == 1:
            # Handle single parcel case - assign a default score
            conn.execute(text(f"""
                CREATE TABLE {final_table_name} AS
                SELECT 
                    p.*,
                    5.0 as mcda_score,
                    5.0 as parcel_rank_normalized
                FROM {filtered_parcels_table} p
            """))
            logger.warning(f"Only 1 parcel found. Assigned default score of 5.0.")
            return
        
        # Create the MCDA calculation with proper normalization (for multiple parcels)
        conn.execute(text(f"""
            CREATE TABLE {final_table_name} AS
            WITH normalized_criteria AS (
                SELECT 
                    p.*,
                    -- Normalize parcel size (0-10 range)
                    CASE 
                        WHEN p.gisacre IS NULL THEN 5
                        WHEN MIN(p.gisacre) OVER () = MAX(p.gisacre) OVER () THEN 5
                        ELSE (p.gisacre - MIN(p.gisacre) OVER ()) / 
                             (MAX(p.gisacre) OVER () - MIN(p.gisacre) OVER ()) * 10
                    END as gisacre_norm,
                    
                    -- Normalize transmission line distance (0-10 range, inverted - closer is better)
                    CASE 
                        WHEN p.et_distance IS NULL THEN 5
                        WHEN MIN(p.et_distance) OVER () = MAX(p.et_distance) OVER () THEN 5
                        ELSE 10 - ((p.et_distance - MIN(p.et_distance) OVER ()) / 
                                 (MAX(p.et_distance) OVER () - MIN(p.et_distance) OVER ()) * 10)
                    END as transmission_distance_norm,
                    
                    -- Normalize drive time (0-10 range, shorter is better)
                    CASE 
                        WHEN p.drive_time IS NULL THEN 5
                        WHEN MIN(p.drive_time) OVER () = MAX(p.drive_time) OVER () THEN 5
                        ELSE 10 - ((p.drive_time - MIN(p.drive_time) OVER ()) / 
                                 (MAX(p.drive_time) OVER () - MIN(p.drive_time) OVER ()) * 10)
                    END as drive_time_norm,
                    
                    -- Calculate building coverage percentage (inverted - less building is better)
                    CASE 
                        WHEN p.ll_gissqft = 0 OR p.ll_gissqft IS NULL THEN 5
                        ELSE LEAST(10, GREATEST(0, 10 - (p.ll_bldg_footprint_sqft / p.ll_gissqft * 10)))
                    END as building_coverage_norm
                FROM {filtered_parcels_table} p
            ),
            mapped_rankings AS (
                SELECT 
                    n.*,
                    -- Map zoning rankings with proper NULL handling
                    COALESCE(z.zoning_ranking, 5) as zoning_ranking,
                    COALESCE(z.zoning_subtype_ranking, 5) as zoning_subtype_ranking,
                    
                    -- Map site description rankings with proper NULL handling
                    COALESCE(s.site_ranking, 5) as site_ranking,
                    
                    -- Map FEMA rankings with proper NULL handling
                    COALESCE(f.fema_nri_ranking, 5) as fema_nri_ranking,
                    COALESCE(f.flood_zone_ranking, 5) as flood_zone_ranking,
                    COALESCE(f.flood_subtype_ranking, 5) as flood_subtype_ranking
                FROM normalized_criteria n
                LEFT JOIN zoning_rankings z 
                    ON n.zoning_type = z.zoning_type 
                    AND n.zoning_subtype = z.zoning_subtype
                LEFT JOIN site_rankings s 
                    ON n.lbcs_site_desc = s.lbcs_site_desc
                LEFT JOIN fema_rankings f 
                    ON n.fema_nri_risk_rating = f.fema_nri_risk_rating
                    AND n.fema_flood_zone = f.fema_flood_zone
                    AND n.fema_flood_zone_subtype = f.fema_flood_zone_subtype
            ),
            mcda_scores AS (
                SELECT 
                    *,
                    -- Calculate MCDA composite scores for each criterion group
                    -- Zoning criterion (mean of type and subtype)
                    {self.weights['zoning']} * (zoning_ranking + zoning_subtype_ranking) / 2 as zoning_score,
                    
                    -- Site Description criterion (description weight)
                    {self.weights['description']} * site_ranking as site_score,
                    
                    -- NRI Risk criterion (fema_nri weight)
                    {self.weights['fema_nri']} * fema_nri_ranking as nri_score,
                    
                    -- Flood criterion (flood weight)
                    {self.weights['flood']} * (flood_zone_ranking + flood_subtype_ranking) as flood_score
                FROM mapped_rankings
            )
            SELECT 
                *,
                -- Calculate final MCDA score
                zoning_score + site_score + nri_score + flood_score as mcda_score,
                
                -- Normalize final score to 0-10 range
                CASE 
                    WHEN MIN(zoning_score + site_score + nri_score + flood_score) OVER () = 
                         MAX(zoning_score + site_score + nri_score + flood_score) OVER () THEN 5
                    ELSE (zoning_score + site_score + nri_score + flood_score - 
                          MIN(zoning_score + site_score + nri_score + flood_score) OVER ()) / 
                         (MAX(zoning_score + site_score + nri_score + flood_score) OVER () - 
                          MIN(zoning_score + site_score + nri_score + flood_score) OVER ()) * 10
                END as parcel_rank_normalized
            FROM mcda_scores
            ORDER BY parcel_rank_normalized DESC
        """))
        
        # Add spatial index
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS {final_table_name.split('.')[-1]}_geom_idx 
            ON {final_table_name} 
            USING GIST (geom)
        """))
    
    def _get_ranking_statistics(self, conn, final_table_name: str) -> Dict:
        """
        Get statistics about the ranking results.
        
        Args:
            conn: Database connection
            final_table_name: Name of the final results table
            
        Returns:
            Dictionary containing ranking statistics
        """
        # Get basic statistics
        stats_query = f"""
            SELECT 
                COUNT(*) as total_parcels,
                MIN(parcel_rank_normalized) as min_score,
                MAX(parcel_rank_normalized) as max_score,
                AVG(parcel_rank_normalized) as mean_score,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY parcel_rank_normalized) as median_score,
                STDDEV(parcel_rank_normalized) as std_dev
            FROM {final_table_name}
        """
        
        result = conn.execute(text(stats_query))
        stats = result.fetchone()
        
        # Get top 5 parcels
        top_query = f"""
            SELECT parcelnumb, parcel_rank_normalized, zoning_type, lbcs_site_desc
            FROM {final_table_name}
            ORDER BY parcel_rank_normalized DESC
            LIMIT 5
        """
        
        top_result = conn.execute(text(top_query))
        top_parcels = []
        for row in top_result.fetchall():
            top_parcels.append({
                'parcelnumb': row[0],
                'parcel_rank_normalized': row[1],
                'zoning_type': row[2],
                'lbcs_site_desc': row[3]
            })
        
        return {
            'total_parcels': stats[0],
            'min_score': float(stats[1]),
            'max_score': float(stats[2]),
            'mean_score': float(stats[3]),
            'median_score': float(stats[4]),
            'std_dev': float(stats[5]),
            'top_parcels': top_parcels
        }
    
    def get_ranking_explanation(self, parcel_id: str, engine) -> Dict:
        """
        Get a detailed explanation of why a specific parcel received its ranking.
        
        Args:
            parcel_id: The parcel number to explain
            engine: SQLAlchemy engine for database connection
            
        Returns:
            Dictionary containing ranking explanation
        """
        # This would query the final results table to get detailed breakdown
        # Implementation depends on the specific table structure
        # For now, return a placeholder
        return {
            'parcel_id': parcel_id,
            'explanation': 'Detailed ranking explanation not yet implemented'
        } 