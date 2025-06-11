#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys
from parcel_filter.core.filter import ParcelFilter
from parcel_filter.core.ranking import ParcelRanker
import geopandas as gpd
from parcel_filter.core.map_viewer import create_parcel_map
from parcel_filter.core.version import VERSION, GITHUB_REPO, check_for_updates, setup_environment, update_application, ensure_environment
from datetime import datetime

def setup_logging():
    """Configure logging settings."""
    # Set PIL logging to WARNING level to suppress debug messages
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    logging.basicConfig(
        level=logging.INFO,  # Changed from DEBUG to INFO
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('parcel_processing.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main function to run the parcel filtering and ranking process."""
    parser = argparse.ArgumentParser(description="Parcel Filtering and Ranking Tool")
    parser.add_argument("--state", required=True, help="Two-letter state code (e.g., az)")
    parser.add_argument("--county", help="County name (e.g., maricopa). If not provided, will analyze all counties in the state.")
    parser.add_argument("--data-dir", help="Base directory containing data. If not provided, assumes 'data'")
    parser.add_argument("--ranking-url", 
                       default='https://docs.google.com/spreadsheets/d/1nzLgafXoqqMcpherLi5jm348cX7rPtOgIDfvTVEp6us/edit?gid=843285247#gid=843285247',
                       help="URL to Google Sheets document containing ranking data")
    parser.add_argument("--force", action="store_true", help="Force re-run of all steps, ignoring checkpoints")
    parser.add_argument("--quick-view", action="store_true", help="Create and display an interactive map of the results")
    parser.add_argument("--update", action="store_true", help="Check for and install updates")
    parser.add_argument("--setup-env", action="store_true", help="Set up or repair the Python environment and exit")
    parser.add_argument("--show-graph", action="store_true", help="Show the distribution graph of parcel rankings")
    args = parser.parse_args()
    
    try:
        # Set up logging
        setup_logging()
        logger = logging.getLogger(__name__)

        # Handle update argument
        if args.update:
            if update_application():
                logger.info("Update completed successfully")
            else:
                logger.error("Update failed")
            return

        # Handle setup-env argument
        if args.setup_env:
            if setup_environment():
                logger.info("Environment setup completed successfully.")
            else:
                logger.error("Environment setup failed.")
            return

        # Ensure environment is set up (fast check, only create if missing)
        if not ensure_environment():
            logger.error("Failed to ensure environment. Please check the logs for details.")
            return

        # Initialize ParcelFilter
        parcel_filter = ParcelFilter(
            state=args.state,
            county=args.county,
            data_dir=args.data_dir,
            ranking_url=args.ranking_url
        )
        
        with parcel_filter:
            # Get list of power providers
            providers = parcel_filter.get_power_providers()
            
            if not providers:
                logger.error("No power providers found in the data")
                return
                
            # Display available providers
            print("\nAvailable power providers:")
            for i, provider in enumerate(providers, 1):
                print(f"{i}. {provider}")
            print("0. No provider filter (include all parcels)")
            
            # Get user selection
            while True:
                try:
                    choice = int(input("\nSelect a power provider (enter number): "))
                    if 0 <= choice <= len(providers):
                        break
                    print("Invalid selection. Please try again.")
                except ValueError:
                    print("Please enter a number.")
            
            if choice == 0:
                logger.info("No power provider filter selected")
                selected_provider = None
            else:
                selected_provider = providers[choice - 1]
                logger.info(f"Selected power provider: {selected_provider}")
            
            if args.force:
                logger.info("Force flag set, running full pipeline")
                parcel_filter.load_parcels(min_size=50)
                parcel_filter.filter_airports()
                parcel_filter.filter_transmission_lines(100.0)
                parcel_filter.filter_power_provider(selected_provider)
                parcel_filter.calculate_drive_times()
            else:
                # Try to load from checkpoints
                if parcel_filter.load_checkpoint("drive_times", "parcels_with_drive_times"):
                    logger.info("Loaded drive time checkpoint, skipping to ranking")
                    parcel_filter.filtered_parcels = parcel_filter._table_to_gdf("SELECT * FROM parcels_with_drive_times")
                else:
                    # Check for transmission line checkpoint
                    checkpoint_name = f"transmission_filtered_{int(100.0)}m"
                    if parcel_filter.load_checkpoint(checkpoint_name, "parcels_filtered_transmission"):
                        logger.info("Loaded transmission line checkpoint, skipping to provider filter")
                        parcel_filter.filtered_parcels = parcel_filter._table_to_gdf("SELECT * FROM parcels_filtered_transmission")
                        parcel_filter.filter_power_provider(selected_provider)
                        parcel_filter.calculate_drive_times()
                    else:
                        # No checkpoints found, run full pipeline
                        logger.info("No checkpoints found, running full pipeline")
                        parcel_filter.load_parcels(min_size=50)
                        parcel_filter.filter_airports()
                        parcel_filter.filter_transmission_lines(100.0)
                        parcel_filter.filter_power_provider(selected_provider)
                        parcel_filter.calculate_drive_times()
            
            # Calculate rankings
            ranked_parcels = parcel_filter.ranker.calculate_rankings(parcel_filter.filtered_parcels, "results", show_graph=args.show_graph)
            
            # Log completion
            logger.info(f"Results saved in {parcel_filter.ranker.state}_{parcel_filter.ranker.county}_{parcel_filter.ranker.utility_provider}_{datetime.now().strftime('%Y%m%d_%H%M%S')} directory")
            
            # If quick-view is enabled, create and display the interactive map
            if args.quick_view:
                create_parcel_map(
                    ranked_parcels,
                    args.state,
                    args.county if args.county else "all_counties",
                    Path("results") / f"{parcel_filter.ranker.state}_{parcel_filter.ranker.county}_{parcel_filter.ranker.utility_provider}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",  # Save map in the same directory as ranked results
                    db_connection=parcel_filter.con  # Pass the database connection
                )
                
    except Exception as e:
        logger.error(f"Error processing parcels: {str(e)}")
        raise

if __name__ == '__main__':
    main() 