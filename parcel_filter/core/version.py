import requests
import logging
import subprocess
import sys
import os
from pathlib import Path

VERSION = "0.2.1"
GITHUB_REPO = "https://github.com/sailbot24/IDTool.git"  # Updated repository URL

def ensure_environment():
    """Ensure the virtual environment exists. Create it if missing."""
    venv_path = Path("venv")
    if os.name == 'nt':
        activate_script = venv_path / "Scripts" / "activate"
    else:
        activate_script = venv_path / "bin" / "activate"

    if venv_path.exists() and activate_script.exists():
        logging.info("Virtual environment already exists. Skipping setup.")
        return True  # Environment is ready

    logging.info("Virtual environment not found. Creating one now...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        # Optionally, install requirements here if needed
        logging.info("Virtual environment created successfully.")
        return True
    except Exception as e:
        logging.error(f"Failed to create virtual environment: {e}")
        return False

def setup_environment():
    """Set up the required environment for the application."""
    venv_created = False
    try:
        # Ensure the virtual environment exists (create if missing)
        venv_created = not ensure_environment()
        venv_path = Path("venv")
        if os.name == 'nt':  # Windows
            pip_path = venv_path / "Scripts" / "pip"
        else:  # Unix/Linux/MacOS
            pip_path = venv_path / "bin" / "pip"

        # Only install/upgrade pip and requirements if the venv was just created
        if venv_created:
            subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
            logging.info("Installing/updating requirements")
            subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
            logging.info("Environment setup completed successfully!")
        else:
            logging.info("Environment already set up. Skipping requirements installation.")
        return True
    except Exception as e:
        logging.error(f"Error setting up environment: {str(e)}")
        return False

def check_for_updates():
    """Check if a newer version is available on GitHub and show current version."""
    print(f"Current version: {VERSION}")
    try:
        # Get the latest release version from GitHub
        response = requests.get(f"{GITHUB_REPO}/releases/latest")
        if response.status_code == 200:
            latest_version = response.json()["tag_name"].strip("v")
            print(f"Latest version on GitHub: {latest_version}")
            if latest_version > VERSION:
                print(f"\nA new version ({latest_version}) is available!")
                print(f"Please update by running: git pull origin main\n")
                return True
            else:
                print("You are running the latest version.")
        else:
            print("Could not fetch the latest version information from GitHub.")
    except Exception as e:
        logging.warning(f"Could not check for updates: {str(e)}")
    return False

def update_application():
    """Update the application to the latest version."""
    try:
        # Check for updates first
        if not check_for_updates():
            print("Application is already up to date!")
            return True

        # Pull latest changes
        subprocess.run(["git", "pull", "origin", "main"], check=True)
        
        # Set up environment with new requirements
        if setup_environment():
            print("Application updated successfully!")
            return True
        return False
    except Exception as e:
        logging.error(f"Error updating application: {str(e)}")
        return False

def validate_config():
    """Validate configuration settings."""
    required_dirs = ['data', 'results']
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            raise RuntimeError(f"Required directory {dir_name} does not exist") 