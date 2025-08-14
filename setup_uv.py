#!/usr/bin/env python3
"""
UV Migration Setup Script

This script helps migrate from the old venv/pip setup to uv.
Run this script to set up uv for the IDTool project.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_uv_installed():
    """Check if uv is installed."""
    try:
        result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def install_uv():
    """Install uv if not already installed."""
    print("Installing uv...")
    try:
        subprocess.run([
            'curl', '-LsSf', 'https://astral.sh/uv/install.sh', '|', 'sh'
        ], shell=True, check=True)
        print("✅ uv installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install uv: {e}")
        sys.exit(1)

def setup_project():
    """Set up the project with uv."""
    print("Setting up project with uv...")
    
    # Remove old venv if it exists
    venv_path = Path("venv")
    if venv_path.exists():
        print("Removing old virtual environment...")
        shutil.rmtree(venv_path)
    
    # Sync dependencies with uv
    try:
        subprocess.run(['uv', 'sync'], check=True)
        print("✅ Dependencies synced successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to sync dependencies: {e}")
        sys.exit(1)

def create_run_scripts():
    """Create convenient run scripts."""
    
    # Create a run script for main.py
    run_script = """#!/bin/bash
# Run the main IDTool application
uv run main.py "$@"
"""
    
    with open('run_idtool.sh', 'w') as f:
        f.write(run_script)
    
    # Make it executable
    os.chmod('run_idtool.sh', 0o755)
    
    # Create a run script for database rebuild
    rebuild_script = """#!/bin/bash
# Run the database rebuild script
uv run rebuild_complete_database.py "$@"
"""
    
    with open('rebuild_db.sh', 'w') as f:
        f.write(rebuild_script)
    
    # Make it executable
    os.chmod('rebuild_db.sh', 0o755)
    
    print("✅ Created run scripts: run_idtool.sh, rebuild_db.sh")

def main():
    """Main migration function."""
    print("🚀 Starting UV migration for IDTool...")
    
    # Check if uv is installed
    if not check_uv_installed():
        print("uv not found. Installing...")
        install_uv()
    else:
        print("✅ uv is already installed")
    
    # Set up the project
    setup_project()
    
    # Create convenient run scripts
    create_run_scripts()
    
    print("\n🎉 Migration complete!")
    print("\nNext steps:")
    print("1. Run the application: ./run_idtool.sh")
    print("2. Rebuild database: ./rebuild_db.sh")
    print("3. Add new dependencies: uv add package_name")
    print("4. Run tests: uv run pytest")
    print("\nYou can now delete requirements.txt if everything works correctly.")

if __name__ == "__main__":
    main()
