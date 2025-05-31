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
    parser = argparse.ArgumentParser(description='Filter and rank parcels based on various criteria.')
    parser.add_argument('--state', required=False, help='State code (e.g., az)')
    parser.add_argument('--county', required=False, help='County name (e.g., maricopa)')
    parser.add_argument('--min-size', type=float, default=50, help='Minimum parcel size in acres')
    parser.add_argument('--data-dir', required=False, default='data', help='Directory containing input data')
    parser.add_argument('--output', default='results', help='Output directory for results')
    parser.add_argument('--transmission-distance', type=float, default=100.0,
                      help='Distance in meters from transmission lines to filter (default: 100.0)')
    parser.add_argument('--power-provider', type=str,
                      help='Filter parcels by power utility provider name (deprecated, use interactive selection)')
    parser.add_argument('--list-providers', action='store_true',
                      help='List all available power providers in the county')
    parser.add_argument('--ranking-url', type=str,
                      default='https://docs.google.com/spreadsheets/d/1nzLgafXoqqMcpherLi5jm348cX7rPtOgIDfvTVEp6us/edit?gid=843285247#gid=843285247',
                      help='Google Sheets URL containing ranking data (default: provided URL)')
    parser.add_argument('--force', action='store_true',
                      help='Force a full run, ignoring any checkpoints')
    parser.add_argument('--quick-view', action='store_true',
                      help='Generate and display an interactive map view of results')
    parser.add_argument('--update', action='store_true',
                      help='Check for and install updates')
    parser.add_argument('--setup-env', action='store_true',
                      help='Set up or repair the Python environment and exit')
    args = parser.parse_args()

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

    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Initialize parcel filter with context manager
        with ParcelFilter(
            state=args.state,
            county=args.county,
            data_dir=args.data_dir,
            ranking_url=args.ranking_url
        ) as parcel_filter:
            
            # Get list of available providers
            providers = parcel_filter.get_power_providers()
            if not providers:
                logger.error("No power providers found in the county data")
                return
            
            # Display available providers and get user selection
            print("\nAvailable power providers:")
            for i, provider in enumerate(providers, 1):
                print(f"{i}. {provider}")
            print("0. No provider filter (include all parcels)")
            
            while True:
                try:
                    selection = input("\nSelect a power provider (enter number): ")
                    if selection == "0":
                        selected_provider = None
                        break
                    index = int(selection) - 1
                    if 0 <= index < len(providers):
                        selected_provider = providers[index]
                        break
                    else:
                        print("Invalid selection. Please enter a number from the list.")
                except ValueError:
                    print("Please enter a valid number.")
            
            if selected_provider:
                logger.info(f"Selected power provider: {selected_provider}")
            else:
                logger.info("No power provider filter selected")
            
            if args.force:
                logger.info("Force flag set, running full pipeline")
                parcel_filter.load_parcels(min_size=args.min_size)
                parcel_filter.filter_airports()
                parcel_filter.filter_transmission_lines(args.transmission_distance)
                parcel_filter.filter_power_provider(selected_provider)
                parcel_filter.calculate_drive_times()
            else:
                # Check for checkpoints in reverse order
                if parcel_filter.load_checkpoint("drive_times", "parcels_with_drive_times"):
                    logger.info("Loaded drive time checkpoint, skipping to ranking")
                    parcel_filter.filtered_parcels = parcel_filter._table_to_gdf("SELECT * FROM parcels_with_drive_times")
                else:
                    # Check for provider checkpoint if specified
                    if selected_provider:
                        provider_checkpoint = f"provider_filtered_{selected_provider.lower().replace(' ', '_')}"
                        if parcel_filter.load_checkpoint(provider_checkpoint, "parcels_filtered_provider"):
                            logger.info("Loaded provider checkpoint, skipping to drive time calculation")
                            parcel_filter.filtered_parcels = parcel_filter._table_to_gdf("SELECT * FROM parcels_filtered_provider")
                            parcel_filter.calculate_drive_times()
                    
                    # Check for transmission line checkpoint
                    checkpoint_name = f"transmission_filtered_{int(args.transmission_distance)}m"
                    if parcel_filter.load_checkpoint(checkpoint_name, "parcels_filtered_transmission"):
                        logger.info("Loaded transmission line checkpoint, skipping to provider filter")
                        parcel_filter.filtered_parcels = parcel_filter._table_to_gdf("SELECT * FROM parcels_filtered_transmission")
                        parcel_filter.filter_power_provider(selected_provider)
                        parcel_filter.calculate_drive_times()
                    else:
                        # No checkpoints found, run full pipeline
                        logger.info("No checkpoints found, running full pipeline")
                        parcel_filter.load_parcels(min_size=args.min_size)
                        parcel_filter.filter_airports()
                        parcel_filter.filter_transmission_lines(args.transmission_distance)
                        parcel_filter.filter_power_provider(selected_provider)
                        parcel_filter.calculate_drive_times()
            
            # Initialize parcel ranker
            parcel_ranker = ParcelRanker(ranking_url=args.ranking_url)
            
            # Calculate rankings
            ranked_parcels, timestamp = parcel_ranker.calculate_rankings(parcel_filter.filtered_parcels, str(output_dir))
            
            logger.info(f"Results saved in run_{timestamp} directory")

            # If quick-view is enabled, create and display the interactive map
            if args.quick_view:
                logger.info("Generating interactive map...")
                map_path = create_parcel_map(
                    ranked_parcels,
                    args.state,
                    args.county,
                    output_dir / f"run_{timestamp}"  # Save map in the same directory as ranked results
                )
                if map_path:
                    logger.info(f"Interactive map saved to {map_path}")

    except Exception as e:
        logger.error(f"Error processing parcels: {str(e)}")
        raise

if __name__ == '__main__':
    main() 