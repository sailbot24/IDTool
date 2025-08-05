import pytest
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point, Polygon
from parcel_filter.core.filter import ParcelFilter

def test_parcel_filter_initialization(test_data_dir):
    """Test ParcelFilter initialization."""
    filter = ParcelFilter(
        state="az",
        county="maricopa",
        data_dir=test_data_dir
    )
    assert filter.state == "az"
    assert filter.county == "maricopa"
    assert filter.data_dir == test_data_dir
    assert filter.con is None
    assert filter.filtered_parcels is None

def test_parcel_filter_context_manager(test_db, test_data_dir):
    """Test ParcelFilter context manager functionality."""
    with ParcelFilter(state="az", county="maricopa", data_dir=test_data_dir) as filter:
        assert filter.con is not None
    assert filter.con is None  # Connection should be closed

def test_load_parcels(test_db, test_data_dir):
    """Test parcel loading functionality."""
    with ParcelFilter(state="az", county="maricopa", data_dir=test_data_dir) as filter:
        count = filter.load_parcels(min_size=50.0)
        assert count > 0
        assert filter.filtered_parcels is not None

def test_filter_airports(test_db, test_data_dir):
    """Test airport filtering functionality."""
    with ParcelFilter(state="az", county="maricopa", data_dir=test_data_dir) as filter:
        filter.load_parcels(min_size=50.0)
        count = filter.filter_airports()
        assert count >= 0  # Could be 0 if no parcels near airports

def test_filter_transmission_lines(test_db, test_data_dir):
    """Test transmission line filtering functionality."""
    with ParcelFilter(state="az", county="maricopa", data_dir=test_data_dir) as filter:
        filter.load_parcels(min_size=50.0)
        filter.filter_transmission_lines(distance_meters=100.0)
        assert filter.filtered_parcels is not None

def test_filter_roadway_distance(test_db, test_data_dir):
    """Test roadway distance filtering functionality."""
    with ParcelFilter(state="az", county="maricopa", data_dir=test_data_dir) as filter:
        filter.load_parcels(min_size=50.0)
        # Test with a specific distance
        filter.filter_roadway_distance(max_distance=1000.0)
        assert filter.filtered_parcels is not None

def test_calculate_drive_times(test_db, test_data_dir):
    """Test drive time calculation functionality."""
    with ParcelFilter(state="az", county="maricopa", data_dir=test_data_dir) as filter:
        filter.load_parcels(min_size=50.0)
        filter.filter_airports()
        filter.filter_transmission_lines(distance_meters=100.0)
        filter.calculate_drive_times()
        assert 'drive_time' in filter.filtered_parcels.columns

def test_checkpoint_operations(test_db, test_data_dir):
    """Test checkpoint save and load operations."""
    with ParcelFilter(state="az", county="maricopa", data_dir=test_data_dir) as filter:
        # Test saving checkpoint
        filter.load_parcels(min_size=50.0)
        filter.save_checkpoint("parcels", "test_checkpoint")
        
        # Test loading checkpoint
        assert filter.load_checkpoint("test_checkpoint", "parcels_loaded")
        
        # Test loading non-existent checkpoint
        assert not filter.load_checkpoint("non_existent", "parcels_loaded")

def test_geometry_conversion(test_db, test_data_dir):
    """Test geometry conversion functionality."""
    with ParcelFilter(state="az", county="maricopa", data_dir=test_data_dir) as filter:
        # Test _table_to_gdf method
        gdf = filter._table_to_gdf("SELECT * FROM parcels")
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert 'geometry' in gdf.columns
        assert gdf.crs == "EPSG:5070"

def test_error_handling(test_data_dir):
    """Test error handling in ParcelFilter."""
    with pytest.raises(ValueError):
        with ParcelFilter(state="invalid", county="maricopa", data_dir=test_data_dir) as filter:
            filter.load_parcels(min_size=-1)  # Invalid min_size

    with pytest.raises(RuntimeError):
        with ParcelFilter(state="az", county="maricopa", data_dir=test_data_dir) as filter:
            filter.calculate_drive_times()  # No parcels loaded 