import pytest
import pandas as pd
from pathlib import Path
from parcel_filter.core.ranking import ParcelRanker
from unittest.mock import patch, MagicMock

def test_ranker_initialization():
    """Test ParcelRanker initialization."""
    ranking_url = "https://example.com/ranking"
    ranker = ParcelRanker(ranking_url=ranking_url)
    assert ranker.ranking_url == ranking_url
    assert ranker.ranking_data is None

def test_calculate_rankings(sample_parcel_gdf, mock_ranking_data):
    """Test ranking calculation functionality."""
    with patch('parcel_filter.core.ranking.ParcelRanker._load_ranking_data') as mock_load:
        mock_load.return_value = mock_ranking_data
        
        ranker = ParcelRanker(ranking_url="https://example.com/ranking")
        ranked_parcels, timestamp = ranker.calculate_rankings(sample_parcel_gdf, "results")
        
        assert isinstance(ranked_parcels, pd.DataFrame)
        assert 'rank' in ranked_parcels.columns
        assert 'score' in ranked_parcels.columns
        assert isinstance(timestamp, str)

def test_ranking_data_loading():
    """Test ranking data loading functionality."""
    with patch('pandas.read_csv') as mock_read:
        mock_data = pd.DataFrame({
            'criteria': ['size', 'location'],
            'weight': [0.6, 0.4]
        })
        mock_read.return_value = mock_data
        
        ranker = ParcelRanker(ranking_url="https://example.com/ranking")
        ranking_data = ranker._load_ranking_data()
        
        assert isinstance(ranking_data, dict)
        assert 'criteria' in ranking_data
        assert 'weights' in ranking_data

def test_ranking_validation(sample_parcel_gdf):
    """Test ranking validation functionality."""
    ranker = ParcelRanker(ranking_url="https://example.com/ranking")
    
    # Test with invalid ranking data
    with pytest.raises(ValueError):
        ranker._validate_ranking_data({})
    
    # Test with missing required fields
    with pytest.raises(ValueError):
        ranker._validate_ranking_data({'criteria': []})

def test_score_calculation(sample_parcel_gdf, mock_ranking_data):
    """Test score calculation functionality."""
    with patch('parcel_filter.core.ranking.ParcelRanker._load_ranking_data') as mock_load:
        mock_load.return_value = mock_ranking_data
        
        ranker = ParcelRanker(ranking_url="https://example.com/ranking")
        scores = ranker._calculate_scores(sample_parcel_gdf)
        
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(sample_parcel_gdf)
        assert all(0 <= score <= 1 for score in scores)

def test_ranking_output(sample_parcel_gdf, mock_ranking_data, test_results_dir):
    """Test ranking output functionality."""
    with patch('parcel_filter.core.ranking.ParcelRanker._load_ranking_data') as mock_load:
        mock_load.return_value = mock_ranking_data
        
        ranker = ParcelRanker(ranking_url="https://example.com/ranking")
        ranked_parcels, timestamp = ranker.calculate_rankings(sample_parcel_gdf, str(test_results_dir))
        
        # Check if output files exist
        output_dir = Path(test_results_dir) / f"run_{timestamp}"
        assert output_dir.exists()
        assert (output_dir / "ranked_parcels.csv").exists()
        assert (output_dir / "ranking_summary.txt").exists()

def test_error_handling(sample_parcel_gdf):
    """Test error handling in ranking process."""
    ranker = ParcelRanker(ranking_url="https://example.com/ranking")
    
    # Test with empty DataFrame
    with pytest.raises(ValueError):
        ranker.calculate_rankings(pd.DataFrame(), "results")
    
    # Test with missing required columns
    with pytest.raises(ValueError):
        ranker.calculate_rankings(pd.DataFrame({'id': [1]}), "results") 