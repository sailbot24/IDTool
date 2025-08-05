#!/usr/bin/env python3
"""
Test script for the final parcel count functionality.
This script tests the get_final_parcel_count method in the ParcelFilter class.
"""

import sys
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from parcel_filter.core.filter import ParcelFilter

def test_final_count():
    """Test the final parcel count functionality."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Database configuration (you may need to adjust this)
    db_config = {
        'host': 'localhost',
        'port': '5432',
        'database': 'LandAI',
        'user': 'postgres',
        'password': 'your_password'  # Update this
    }
    
    try:
        # Create a ParcelFilter instance
        parcel_filter = ParcelFilter(
            state="az",  # Change to your state
            county="maricopa",  # Change to your county
            db_config=db_config
        )
        
        with parcel_filter:
            # Test the final count functionality
            logger.info("Testing final parcel count functionality...")
            
            # Initially, there should be no parcels
            initial_count = parcel_filter.get_final_parcel_count()
            logger.info(f"Initial parcel count: {initial_count}")
            
            # Load parcels
            parcel_filter.load_parcels(min_size=50.0)
            
            # After loading, check count
            loaded_count = parcel_filter.get_final_parcel_count()
            logger.info(f"After loading parcels: {loaded_count}")
            
            # Test the summary functionality
            summary = parcel_filter.get_filtering_summary()
            logger.info("Filtering summary:")
            for key, value in summary.items():
                logger.info(f"  {key}: {value}")
            
            logger.info("Final parcel count test completed successfully!")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_final_count()
    if success:
        print("✅ Final parcel count test passed!")
    else:
        print("❌ Final parcel count test failed!")
        sys.exit(1) 