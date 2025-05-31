import pytest
from pathlib import Path
import subprocess
import os
from unittest.mock import patch, MagicMock
from parcel_filter.core.version import (
    VERSION,
    GITHUB_REPO,
    setup_environment,
    check_for_updates,
    update_application,
    validate_config
)

def test_version_format():
    """Test that version follows semantic versioning."""
    parts = VERSION.split('.')
    assert len(parts) == 3, "Version should be in format X.Y.Z"
    assert all(part.isdigit() for part in parts), "Version parts should be numbers"

def test_github_repo_format():
    """Test that GitHub repo URL is properly formatted."""
    assert GITHUB_REPO.startswith('https://github.com/'), "Should be a GitHub URL"
    assert 'yourusername' not in GITHUB_REPO, "Should replace placeholder username"

def test_setup_environment(temp_dir):
    """Test environment setup functionality."""
    with patch('pathlib.Path.exists') as mock_exists, \
         patch('subprocess.run') as mock_run:
        
        # Test when venv doesn't exist
        mock_exists.return_value = False
        assert setup_environment() is True
        mock_run.assert_called()

        # Test when venv exists
        mock_exists.return_value = True
        assert setup_environment() is True
        mock_run.assert_called()

def test_setup_environment_failure():
    """Test environment setup failure handling."""
    with patch('subprocess.run') as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(1, 'pip')
        assert setup_environment() is False

def test_check_for_updates():
    """Test update checking functionality."""
    with patch('requests.get') as mock_get:
        # Test when update is available
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"tag_name": "v1.0.0"}
        mock_get.return_value = mock_response
        
        with patch('parcel_filter.core.version.VERSION', '0.1.0'):
            assert check_for_updates() is True

        # Test when no update is available
        with patch('parcel_filter.core.version.VERSION', '1.0.0'):
            assert check_for_updates() is False

        # Test when GitHub API fails
        mock_get.side_effect = Exception("API Error")
        assert check_for_updates() is False

def test_update_application():
    """Test application update functionality."""
    with patch('parcel_filter.core.version.check_for_updates') as mock_check, \
         patch('subprocess.run') as mock_run, \
         patch('parcel_filter.core.version.setup_environment') as mock_setup:
        
        # Test successful update
        mock_check.return_value = True
        mock_setup.return_value = True
        assert update_application() is True
        mock_run.assert_called_with(['git', 'pull', 'origin', 'main'], check=True)

        # Test when no update is available
        mock_check.return_value = False
        assert update_application() is True
        mock_run.assert_not_called()

        # Test when update fails
        mock_check.return_value = True
        mock_run.side_effect = subprocess.CalledProcessError(1, 'git')
        assert update_application() is False

def test_validate_config(temp_dir, test_data_dir, test_results_dir):
    """Test configuration validation."""
    # Test with valid configuration
    assert validate_config() is None

    # Test with missing directory
    with patch('pathlib.Path.exists') as mock_exists:
        mock_exists.return_value = False
        with pytest.raises(RuntimeError):
            validate_config() 