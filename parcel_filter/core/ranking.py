import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from io import StringIO
import geopandas as gpd
import json
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib

logger = logging.getLogger(__name__)

class ParcelRanker:
    def __init__(self, ranking_url: str):
        """
        Initialize the ParcelRanker with ranking data from Google Sheets.
        
        Args:
            ranking_url: URL to the Google Sheets document containing ranking data
        """
        self.ranking_url = ranking_url
        self.ranking_data = None
        self.weights = None
        
    def load_ranking_data(self) -> None:
        """Load ranking data and weights from Google Sheets."""
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
                
                # Log the loaded data for debugging
                logger.info(f"Successfully loaded ranking data with columns: {self.ranking_data.columns.tolist()}")
                logger.info(f"Loaded weights: {self.weights}")
            else:
                raise ValueError("Invalid Google Sheets URL")
        except Exception as e:
            logger.error(f"Failed to load ranking data: {str(e)}")
            raise
    
    def save_weights_and_rankings(self, parcels: gpd.GeoDataFrame, output_dir: str) -> str:
        """
        Save weights and rankings to a text file and return the timestamped filename.
        
        Args:
            parcels: GeoDataFrame containing the ranked parcels
            output_dir: Directory to save the weights file
            
        Returns:
            Timestamp string for use in output filenames
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create weights and rankings summary
        summary = {
            "timestamp": timestamp,
            "weights": self.weights,
            "ranking_data": self.ranking_data.to_dict('records'),
            "ranking_stats": {
                "min_score": float(parcels['parcel_rank_normalized'].min()),
                "max_score": float(parcels['parcel_rank_normalized'].max()),
                "mean_score": float(parcels['parcel_rank_normalized'].mean())
            },
            "top_parcels": parcels.nlargest(5, 'parcel_rank_normalized')[['parcelnumb', 'parcel_rank_normalized']].to_dict('records')
        }
        
        # Save to JSON file
        weights_file = f"{output_dir}/weights_{timestamp}.json"
        with open(weights_file, 'w') as f:
            json.dump(summary, f, indent=4)
            
        logger.info(f"Saved weights and rankings to {weights_file}")
        return timestamp
    
    def min_max_normalize(self, series: pd.Series) -> pd.Series:
        """
        Normalize a series using min-max normalization to 0-10 range.
        Matches the R implementation.
        
        Args:
            series: Pandas Series to normalize
            
        Returns:
            Normalized series scaled to 0-10 range
        """
        # If all values are NA, return a vector of zeros
        if series.isna().all():
            return pd.Series(0, index=series.index)
        
        # Get min and max, ignoring NA values
        min_val = series.min(skipna=True)
        max_val = series.max(skipna=True)
        
        if min_val == max_val:
            # For non-NA values, set to 10; for NA values, set to 0
            return series.notna().astype(float) * 10
        else:
            # Scale to 0-10 range
            normalized = (series - min_val) / (max_val - min_val) * 10
            # Replace NA values with 0
            return normalized.fillna(0)
    
    def calculate_rankings(self, parcels: gpd.GeoDataFrame, output_dir: str) -> gpd.GeoDataFrame:
        """Calculate rankings for parcels based on various criteria."""
        if self.ranking_data is None or self.weights is None:
            self.load_ranking_data()
            
        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a run-specific directory
        run_dir = f"{output_dir}/run_{timestamp}"
        os.makedirs(run_dir, exist_ok=True)
        logger.info(f"Created run directory: {run_dir}")
            
        # Create a mapping dictionary for each ranking type
        zoning_ranking = pd.Series(self.ranking_data['Zoning Ranking'].values, index=self.ranking_data['zoning_type']).to_dict()
        zoning_subtype_ranking = pd.Series(self.ranking_data['Zoning Subtype Ranking'].values, index=self.ranking_data['zoning_subtype']).to_dict()
        activity_ranking = pd.Series(self.ranking_data['Land Activity Ranking'].values, index=self.ranking_data['lbcs_activity_desc']).to_dict()
        site_ranking = pd.Series(self.ranking_data['Site Descirption Ranking'].values, index=self.ranking_data['lbcs_site_desc']).to_dict()
        ownership_ranking = pd.Series(self.ranking_data['Ownership Ranking'].values, index=self.ranking_data['lbcs_ownership_desc']).to_dict()
        fema_nri_ranking = pd.Series(self.ranking_data['Fema NRI Flood Risk Ranking'].values, index=self.ranking_data['fema_nri_risk_ranking']).to_dict()
        flood_zone_ranking = pd.Series(self.ranking_data['Fema Flood Zone Ranking'].values, index=self.ranking_data['fema_flood_zone']).to_dict()
        flood_subtype_ranking = pd.Series(self.ranking_data['Fema Flood Zone Subtype Ranking'].values, index=self.ranking_data['fema_flood_zone_subtype']).to_dict()
        
        # Map rankings to parcels using the dictionaries
        parcels['Zoning Ranking'] = parcels['zoning_type'].map(zoning_ranking).fillna(0)
        parcels['Zoning Subtype Ranking'] = parcels['zoning_subtype'].map(zoning_subtype_ranking).fillna(0)
        parcels['Land Activity Ranking'] = parcels['lbcs_activity_desc'].map(activity_ranking).fillna(0)
        parcels['Site Descirption Ranking'] = parcels['lbcs_site_desc'].map(site_ranking).fillna(0)
        parcels['Ownership Ranking'] = parcels['lbcs_ownership_desc'].map(ownership_ranking).fillna(0)
        parcels['Fema NRI Flood Risk Ranking'] = parcels['fema_nri_risk_rating'].map(fema_nri_ranking).fillna(0)
        parcels['Fema Flood Zone Ranking'] = parcels['fema_flood_zone'].map(flood_zone_ranking).fillna(0)
        parcels['Fema Flood Zone Subtype Ranking'] = parcels['fema_flood_zone_subtype'].map(flood_subtype_ranking).fillna(0)
        
        # Calculate normalized values
        parcels['gisacre_norm'] = self.min_max_normalize(parcels['gisacre'])
        parcels['trans_line_distance_norm'] = 10 - self.min_max_normalize(parcels['transmission_line_distance'])
        parcels['norm_dt'] = self.min_max_normalize(parcels['drive_time'])
        
        # Calculate building cover percentage
        parcels['building_cover_prec'] = parcels['ll_bldg_footprint_sqft'] / parcels['ll_gissqft']
        parcels['building_cover_prec'] = parcels['building_cover_prec'].fillna(0)
        
        # Normalize building cover percentage
        min_bc = parcels['building_cover_prec'].min()
        max_bc = parcels['building_cover_prec'].max()
        if min_bc == max_bc:
            parcels['building_cover_prec'] = 1
        else:
            parcels['building_cover_prec'] = ((parcels['building_cover_prec'] - min_bc) / 
                                            (max_bc - min_bc) * 9 + 1).fillna(0)
        
        parcels['building_cover_prec_invert'] = 10 - parcels['building_cover_prec'].fillna(5)
        
        # Calculate final ranking score using weights from Google Sheets
        parcels['parcel_rank'] = (
            self.weights['zoning_subtype'] * (parcels['Zoning Ranking'] + parcels['Zoning Subtype Ranking']) +
            self.weights['activity'] * (parcels['Site Descirption Ranking'] * 0.3 + 
                                      parcels['Land Activity Ranking'] * 0.3 + 
                                      parcels['Ownership Ranking'] * 0.2 + 
                                      parcels['gisacre_norm'] * 0.1 + 
                                      parcels['building_cover_prec_invert'] * 0.1) +
            self.weights['site'] * parcels['norm_dt'] +
            self.weights['fema_nri'] * (parcels['Fema NRI Flood Risk Ranking'] + 
                                      parcels['Fema Flood Zone Ranking'] + 
                                      parcels['Fema Flood Zone Subtype Ranking']) +
            self.weights['flood_zone'] * parcels['trans_line_distance_norm']
        )
        
        # Normalize final ranking score
        parcels['parcel_rank_normalized'] = self.min_max_normalize(parcels['parcel_rank'])
        
        # Save weights and rankings to the run directory
        self.save_weights_and_rankings(parcels, run_dir)
        
        # Convert all numeric columns to standard numpy types
        numeric_cols = parcels.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            parcels[col] = parcels[col].astype('float64')
        
        # Ensure geometry column is properly set
        if 'geometry' not in parcels.columns:
            raise ValueError("No geometry column found in parcels DataFrame")
        
        # Save results as GeoPackage in the run directory
        gpkg_file = f"{run_dir}/{timestamp}_ranked_parcels.gpkg"
        parcels.to_file(gpkg_file, driver="GPKG")
        logger.info(f"Saved ranked parcels to {gpkg_file}")
        

        # Suppress matplotlib font manager messages
        matplotlib.set_loglevel('error')

        # Plot a distribution of the parcel_rank_normalized values 
        plt.figure(figsize=(10, 6))
        plt.hist(parcels['parcel_rank_normalized'], bins=20, edgecolor='black')
        plt.title('Distribution of Parcel Rankings')
        plt.xlabel('Normalized Ranking Score')
        plt.ylabel('Frequency')
        mean_value = parcels['parcel_rank_normalized'].mean()
        plt.axvline(x=mean_value, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.2f}')
        plt.legend()
        plt.show()
    
        

        # Log ranking statistics
        logger.info("Ranking statistics:")
        logger.info(f"  Min score: {parcels['parcel_rank_normalized'].min():.2f}")
        logger.info(f"  Max score: {parcels['parcel_rank_normalized'].max():.2f}")
        logger.info(f"  Mean score: {parcels['parcel_rank_normalized'].mean():.2f}")
        
        # Log top 5 parcels by ranking
        top_parcels = parcels.nlargest(5, 'parcel_rank_normalized')
        logger.info("Top 5 parcels by ranking:")
        for idx, row in top_parcels.iterrows():
            logger.info(f"  Parcel {row['parcelnumb']}: Score {row['parcel_rank_normalized']:.2f}")
            
        return parcels, timestamp 