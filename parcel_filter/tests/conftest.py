import pytest
import os
from pathlib import Path
import tempfile
import shutil
import duckdb
import geopandas as gpd
from shapely.geometry import Point, Polygon

@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def test_data_dir(temp_dir):
    """Create a test data directory with required structure."""
    data_dir = Path(temp_dir) / "data"
    data_dir.mkdir(parents=True)
    return data_dir

@pytest.fixture(scope="session")
def test_results_dir(temp_dir):
    """Create a test results directory."""
    results_dir = Path(temp_dir) / "results"
    results_dir.mkdir(parents=True)
    return results_dir

@pytest.fixture(scope="session")
def test_db(test_data_dir):
    """Create a test DuckDB database with sample data."""
    db_path = test_data_dir / "test_landai.ddb"
    con = duckdb.connect(str(db_path))
    
    # Create test tables
    con.execute("""
        CREATE TABLE parcels AS
        SELECT 
            1 as id,
            'Test Parcel' as name,
            ST_GeomFromText('POLYGON((0 0, 0 1, 1 1, 1 0, 0 0))') as geometry,
            100.0 as acres
    """)
    
    con.execute("""
        CREATE TABLE airports AS
        SELECT 
            1 as id,
            'Test Airport' as name,
            ST_GeomFromText('POINT(0.5 0.5)') as geometry
    """)
    
    con.execute("""
        CREATE TABLE transmission_lines AS
        SELECT 
            1 as id,
            'Test Line' as name,
            ST_GeomFromText('LINESTRING(0 0.5, 1 0.5)') as geometry
    """)
    
    yield con
    con.close()
    if db_path.exists():
        db_path.unlink()

@pytest.fixture
def sample_parcel_gdf():
    """Create a sample GeoDataFrame of parcels."""
    return gpd.GeoDataFrame({
        'id': [1, 2],
        'name': ['Parcel 1', 'Parcel 2'],
        'geometry': [
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
            Polygon([(2, 2), (2, 3), (3, 3), (3, 2)])
        ],
        'acres': [100.0, 200.0]
    }, crs="EPSG:5070")

@pytest.fixture
def mock_ranking_data():
    """Create mock ranking data."""
    return {
        'criteria': ['size', 'location'],
        'weights': [0.6, 0.4],
        'scores': {
            'size': {'100.0': 0.8, '200.0': 0.9},
            'location': {'near': 0.7, 'far': 0.3}
        }
    } 