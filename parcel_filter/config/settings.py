from pathlib import Path
from typing import List, Optional

# Default unwanted activities from filter_parcels.py
DEFAULT_UNWANTED = [
    "Military base",
    "Emergency response or public-safety-related",
    "School or library",
    "Social, cultural, or religious assembly",
    "Power generation, control, monitor, or distribution",
    "Trains or other rail movement",
    "Activities associated with utilities (water, sewer, power, etc.)", 
    "Promenading and other activities in parks",
    "Health care, medical, or treatment"
]

class Settings:
    def __init__(
        self,
        state: str,
        county: str,
        min_size: float = 50,
        data_dir: Optional[str] = None,
        output_file: Optional[str] = None,
        unwanted: Optional[List[str]] = None,
        transmission_distance: float = 100
    ):
        """
        Initialize settings for parcel filtering.
        
        Args:
            state: Two-letter state code (e.g., 'az')
            county: County name (e.g., 'maricopa')
            min_size: Minimum parcel size in acres
            data_dir: Base directory containing data
            output_file: Path to save the filtered results
            unwanted: List of unwanted activity descriptions to exclude. If None, uses DEFAULT_UNWANTED
            transmission_distance: Maximum distance (in meters) from transmission lines
        """
        self.state = state.lower()
        self.county = county.lower()
        self.min_size = min_size
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.output_file = Path(output_file) if output_file else self._get_default_output_path()
        self.unwanted = unwanted if unwanted is not None else DEFAULT_UNWANTED.copy()
        self.transmission_distance = transmission_distance
    
    def _get_default_output_path(self) -> Path:
        """Get the default output path based on state and county."""
        return self.data_dir / "parcels" / self.state / self.county / "filtered" / f"{self.state}_{self.county}_filtered.parquet"
    
    def validate(self) -> None:
        """Validate the settings."""
        if not self.state or len(self.state) != 2:
            raise ValueError("State must be a two-letter code")
        
        if not self.county:
            raise ValueError("County must be specified")
        
        if self.min_size <= 0:
            raise ValueError("Minimum size must be positive")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")
        
        if self.transmission_distance <= 0:
            raise ValueError("Transmission distance must be positive") 