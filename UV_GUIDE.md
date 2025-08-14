# UV Quick Reference Guide

This guide covers the essential uv commands for working with the IDTool project.

## 🚀 Getting Started

### Install UV
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Set up the project
```bash
# Install dependencies with Python 3.11
uv sync --python 3.11

# Or run the migration script
python3 setup_uv.py
```

## 📦 Package Management

### Install dependencies
```bash
# Install all dependencies from pyproject.toml
uv sync --python 3.11

# Install with development dependencies
uv sync --dev --python 3.11
```

### Add new packages
```bash
# Add a production dependency
uv add requests

# Add a development dependency
uv add --dev pytest

# Add with specific version
uv add "pandas>=2.0.0"
```

### Remove packages
```bash
# Remove a package
uv remove requests
```

## 🏃‍♂️ Running Scripts

### Run Python scripts
```bash
# Run the main application
uv run main.py

# Run with arguments
uv run main.py --state co --county adams

# Run the database rebuild
uv run rebuild_complete_database.py --rebuild-all

# Run tests
uv run pytest
```

### Use the convenience scripts
```bash
# Run the main application
./run_idtool.sh

# Rebuild the database
./rebuild_db.sh
```

## 🔧 Development

### Activate the virtual environment
```bash
# Enter the uv virtual environment
uv shell

# Now you can run Python directly
python main.py
```

### Run tests
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=parcel_filter

# Run specific test file
uv run pytest tests/test_filter.py
```

### Update dependencies
```bash
# Update all dependencies to latest compatible versions
uv sync --upgrade

# Update specific package
uv add --upgrade pandas
```

## 📁 Project Structure

The project now uses:
- `pyproject.toml` - Project configuration and dependencies
- `uv.lock` - Lock file for reproducible builds (auto-generated)
- `run_idtool.sh` - Convenience script to run the main application
- `rebuild_db.sh` - Convenience script to rebuild the database

## 🔄 Migration from pip/venv

If you're migrating from the old setup:

1. **Remove old virtual environment**:
   ```bash
   rm -rf venv/
   ```

2. **Set up with uv**:
   ```bash
   uv sync
   ```

3. **Update your workflow**:
   - Replace `source venv/bin/activate && python script.py` with `uv run script.py`
   - Replace `pip install package` with `uv add package`
   - Replace `pip install -r requirements.txt` with `uv sync`

## 🆚 UV vs pip Comparison

| Task | pip/venv | uv |
|------|----------|-----|
| Install dependencies | `pip install -r requirements.txt` | `uv sync` |
| Add package | `pip install package` | `uv add package` |
| Run script | `source venv/bin/activate && python script.py` | `uv run script.py` |
| Create venv | `python -m venv venv` | Automatic |
| Activate venv | `source venv/bin/activate` | `uv shell` |

## 🎯 Benefits of UV

- **Faster**: 10-100x faster than pip
- **Better dependency resolution**: Fewer conflicts
- **Lock files**: Reproducible builds
- **No manual venv management**: Automatic virtual environments
- **Modern tooling**: Built for modern Python development

## 🐛 Troubleshooting

### If uv is not found
```bash
# Clear shell command cache (if uv was installed after shell started)
hash -r

# Or add uv to your PATH (add to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.local/bin:$PATH"
```

### If dependencies fail to install
```bash
# Clear cache and retry
uv cache clean
uv sync
```

### If you need to go back to pip
```bash
# The old requirements.txt is still available
pip install -r requirements.txt
```
