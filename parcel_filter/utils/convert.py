import logging
from pathlib import Path
import duckdb
from typing import Union, Optional

logger = logging.getLogger(__name__)

def convert_gpkg_to_parquet(input_file: Union[str, Path], output_file: Optional[Union[str, Path]] = None) -> Path:
    """
    Convert a GeoPackage file to Parquet format.
    
    Args:
        input_file: Path to the input GeoPackage file
        output_file: Path to save the output Parquet file. If None, uses the same name with .parquet extension
        
    Returns:
        Path to the output Parquet file
    """
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if output_file is None:
        output_file = input_file.with_suffix('.parquet')
    else:
        output_file = Path(output_file)
    
    logger.info(f"Converting {input_file} to {output_file}")
    
    # Connect to DuckDB
    con = duckdb.connect(database=":memory:")
    con.install_extension('spatial')
    con.load_extension('spatial')
    
    try:
        # Read the GeoPackage file
        con.execute(f"""
            CREATE TABLE data AS
            SELECT *
            FROM ST_Read('{input_file}')
        """)
        
        # Write to Parquet
        con.execute(f"""
            COPY data TO '{output_file}' (FORMAT PARQUET)
        """)
        
        return output_file
    finally:
        con.close() 