#! source venv/bin/activate
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
import json
from sqlalchemy import text

def setup_logging():
    """Configure logging settings."""
    # Clear the log file at the start of each run
    with open('parcel_processing.log', 'w'):
        pass
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

def read_db_config(config_path):
    """Read database configuration from a file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing database configuration
    """
    config = {}
    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
        return config
    except Exception as e:
        logging.error(f"Error reading database configuration: {e}")
        raise

def run_parcel_pipeline(parcel_filter, selected_provider=None):
    """Run the complete parcel filtering and ranking pipeline in the correct order.
    
    Args:
        parcel_filter: ParcelFilter instance
        selected_provider: Optional power provider to filter by
        
    Returns:
        bool: True if pipeline completed successfully, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Step 1: Load parcels
        logger.info("Step 1: Loading parcels...")
        parcel_filter.load_parcels()
        
        # Step 2: Apply filters in correct order
        logger.info("Step 2: Applying filters...")
        parcel_filter.filter_airports()
        parcel_filter.filter_transmission_lines()
        if selected_provider:
            parcel_filter.filter_power_provider(selected_provider)
            # Set the utility filter for table naming
            parcel_filter.set_utility_filter(selected_provider)
            
        # Step 3: Calculate drive times (must be done before ranking)
        logger.info("Step 3: Calculating drive times...")
        parcel_filter.calculate_drive_times()
        
        # Step 4: Rank parcels
        logger.info("Step 4: Ranking parcels...")
        parcel_filter.rank_parcels_in_postgis()
        
        # Step 5: Save results
        logger.info("Step 5: Saving results...")
        parcel_filter.save_results_to_db()
        
        # Step 6: Cleanup
        logger.info("Step 6: Cleaning up temporary data...")
        parcel_filter.cleanup()
        
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return False

def main():
    """Main function to run the parcel filtering and ranking process."""
    parser = argparse.ArgumentParser(description="Parcel Filtering and Ranking Tool")
    parser.add_argument("--state", help="Two-letter state code (e.g., az). If not provided, will show interactive selection.")
    parser.add_argument("--county", help="County name (e.g., maricopa). If not provided, will show interactive selection.")
    parser.add_argument("--db-config", default="db_config.txt", help="Path to database configuration file (default: db_config.txt)")
    parser.add_argument("--min-size", type=float, default=50.0,
                      help="Minimum parcel size in acres")
    parser.add_argument("--transmission-distance", type=float, default=100.0,
                      help="Maximum distance to transmission lines in meters")
    parser.add_argument("--provider", type=str,
                      help="Power utility provider to filter by")
    parser.add_argument("--ranking-url", 
                      default='https://docs.google.com/spreadsheets/d/1nzLgafXoqqMcpherLi5jm348cX7rPtOgIDfvTVEp6us/edit?gid=843285247#gid=843285247',
                      help="URL to Google Sheets document containing ranking data")
    parser.add_argument("--force", action="store_true", help="Force re-run of all steps")
    parser.add_argument("--quick-view", action="store_true", help="Create and display an interactive map of the results")
    parser.add_argument("--update", action="store_true", help="Check for and install updates")
    parser.add_argument("--setup-env", action="store_true", help="Set up or repair the Python environment and exit")
    parser.add_argument("--show-graph", action="store_true", help="Show the distribution graph of parcel rankings")
    args = parser.parse_args()
    
    try:
        # Set up logging
        setup_logging()
        logger = logging.getLogger(__name__)

        # Handle update and setup-env arguments
        if args.update:
            if update_application():
                logger.info("Update completed successfully")
            else:
                logger.error("Update failed")
            return

        if args.setup_env:
            if setup_environment():
                logger.info("Environment setup completed successfully.")
            else:
                logger.error("Environment setup failed.")
            return

        # Ensure environment is set up
        if not ensure_environment():
            logger.error("Failed to ensure environment. Please check the logs for details.")
            return

        # Load database configuration
        try:
            with open(args.db_config, 'r') as f:
                db_config = {}
                for line in f:
                    key, value = line.strip().split('=')
                    db_config[key.strip()] = value.strip()
            logger.info(f"Using database configuration from {args.db_config}")
            
            # Create a temporary ParcelFilter instance to get available states and counties
            temp_filter = ParcelFilter(
                state="temp",  # Temporary state for connection
                db_config=db_config,
                ranking_url=args.ranking_url,
                transmission_distance=args.transmission_distance
            )
            
            with temp_filter:
                # Get available states
                available_states = temp_filter.get_available_states()
                
                if not available_states:
                    logger.error("No states found in the database")
                    return
                
                # State selection
                selected_state = args.state
                if not selected_state:
                    print("\nAvailable states:")
                    for i, state in enumerate(available_states, 1):
                        print(f"{i}. {state.upper()}")
                    
                    while True:
                        try:
                            choice = int(input("\nSelect a state (enter number): "))
                            if 1 <= choice <= len(available_states):
                                selected_state = available_states[choice - 1]
                                break
                            print("Invalid selection. Please try again.")
                        except ValueError:
                            print("Please enter a number.")
                else:
                    selected_state = selected_state.lower()
                    if selected_state not in available_states:
                        logger.error(f"State '{selected_state}' not found in available states: {available_states}")
                        return
                
                logger.info(f"Selected state: {selected_state}")
                
                # Get available counties for the selected state
                available_counties = temp_filter.get_available_counties(selected_state)
                
                if not available_counties:
                    logger.error(f"No counties found for state {selected_state}")
                    return
                
                # County selection
                selected_county = args.county
                if not selected_county:
                    print(f"\nAvailable counties for {selected_state.upper()}:")
                    for i, county in enumerate(available_counties, 1):
                        print(f"{i}. {county.title()}")
                    print("0. All counties in state")
                    
                    while True:
                        try:
                            choice = int(input("\nSelect a county (enter number): "))
                            if 0 <= choice <= len(available_counties):
                                selected_county = None if choice == 0 else available_counties[choice - 1]
                                break
                            print("Invalid selection. Please try again.")
                        except ValueError:
                            print("Please enter a number.")
                else:
                    selected_county = selected_county.lower()
                    if selected_county not in available_counties:
                        logger.error(f"County '{selected_county}' not found in available counties for state {selected_state}: {available_counties}")
                        return
                
                logger.info(f"Selected county: {selected_county if selected_county else 'All counties'}")
            
            # Now create the actual ParcelFilter instance with selected state and county
            parcel_filter = ParcelFilter(
                state=selected_state,
                county=selected_county,
                db_config=db_config,
                ranking_url=args.ranking_url,
                transmission_distance=args.transmission_distance
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
                
                selected_provider = None if choice == 0 else providers[choice - 1]
                logger.info(f"Selected power provider: {selected_provider if selected_provider else 'None'}")
                
                # Run the pipeline
                if run_parcel_pipeline(parcel_filter, selected_provider):
                    # Show quick view if requested
                    if args.quick_view:
                        # Create results directory
                        results_dir = Path("results")
                        results_dir.mkdir(exist_ok=True)
                        
                        # Create timestamp for the output directory
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        # Sanitize provider name for file system
                        if selected_provider:
                            # Convert to lowercase and replace spaces with underscores
                            provider_name = selected_provider.lower().replace(' ', '_')
                            # Remove or replace invalid file system characters
                            import re
                            provider_name = re.sub(r'[^a-z0-9_]', '', provider_name)
                            # Ensure it doesn't start with a number
                            if provider_name and provider_name[0].isdigit():
                                provider_name = 'provider_' + provider_name
                            # Ensure it's not empty
                            if not provider_name:
                                provider_name = 'unknown_provider'
                        else:
                            provider_name = 'all_providers'
                        output_dir = results_dir / f"{parcel_filter.state}_{parcel_filter.county}_{provider_name}_{timestamp}"
                        output_dir.mkdir(exist_ok=True)
                        
                        # Load ranked parcels directly from the database
                        final_table = parcel_filter.get_final_results_table()
                        if final_table:
                            logger.info(f"Loading ranked parcels from database table: {final_table}")
                            ranked_parcels = gpd.read_postgis(
                                f"SELECT * FROM {final_table}",
                                parcel_filter.db_utils.engine,
                                geom_col='geom'
                            )
                            
                            # Create and display the interactive map
                            create_parcel_map(
                                ranked_parcels,
                                selected_state,
                                selected_county if selected_county else "all_counties",
                                output_dir,
                                db_connection=parcel_filter.db_utils.engine
                            )
                        else:
                            logger.error("No final results table found. Cannot create quick view.")
                else:
                    logger.error("Pipeline failed. Check the logs for details.")
                
        except Exception as e:
            logger.error(f"Error processing parcels: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")
        raise
    finally:
        # Ensure connection is closed
        if 'parcel_filter' in locals() and parcel_filter.db_utils.engine is not None:
            parcel_filter.db_utils.engine.dispose()

if __name__ == '__main__':
    main() 