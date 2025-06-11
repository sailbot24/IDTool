import logging
from pathlib import Path
from typing import Union, Dict, Any, Optional, List, Tuple
import geopandas as gpd
import folium
from folium import plugins
from folium.plugins import MousePosition, Fullscreen
import webbrowser
import os
import branca.colormap as cm
import pandas as pd
import numpy as np
import shapely.wkt as wkt
import json
from shapely.geometry import Point

logger = logging.getLogger(__name__)

# Define formatter for mouse position coordinates
COORDINATE_FORMATTER = "function(num) {return L.Util.formatNum(num, 5) + ' &deg; ';};"

def create_parcel_map(parcels: gpd.GeoDataFrame, state: str, county: str, output_dir: Path, 
                     isochrones: Optional[gpd.GeoDataFrame] = None,
                     poi_data: Optional[Dict[str, gpd.GeoDataFrame]] = None,
                     poi_icons: Optional[Dict[str, str]] = None,
                     poi_colors: Optional[Dict[str, str]] = None,
                     db_connection = None) -> None:
    """
    Create an interactive map of parcels with ranking visualization.
    
    Args:
        parcels: GeoDataFrame containing parcel data with ranking
        state: State code
        county: County name
        output_dir: Directory to save the map
        isochrones: Optional GeoDataFrame containing isochrone data
        poi_data: Optional dictionary of POI GeoDataFrames
        poi_icons: Optional dictionary mapping POI types to icon names
        poi_colors: Optional dictionary mapping POI types to colors
        db_connection: Optional DuckDB connection for loading substations
    """
    # Default center points for each state (approximate)
    default_centers = {
        'co': [39.0, -105.5],  # Colorado
        'az': [34.0, -112.0],  # Arizona
    }
    try:
        # Convert any Timestamp columns to strings
        for col in parcels.columns:
            if pd.api.types.is_datetime64_any_dtype(parcels[col]):
                parcels[col] = parcels[col].astype(str)
                logger.info(f"Converted Timestamp column '{col}' to string")

        logger.info(f"Number of parcels passed to map: {len(parcels)}")
        logger.info(f"Sample of parcels:\n{parcels.head()}\nColumns: {parcels.columns.tolist()}")

        # Check if parcels is empty
        if parcels.empty:
            logger.error("No parcels to display on the map. Skipping map creation.")
            return

        # Ensure parcels are in EPSG:4326 for folium
        if parcels.crs is not None and parcels.crs.to_epsg() != 4326:
            parcels = parcels.to_crs(epsg=4326)
            logger.info("Transformed parcels to WGS84 (EPSG:4326) for mapping")

        # Calculate bounds and center
        bounds = parcels.total_bounds
        logger.info(f"Parcel bounds: {bounds}")
        if not any(pd.isna(bounds)) and not any(np.isinf(bounds)):
            center_lon = (bounds[0] + bounds[2]) / 2
            center_lat = (bounds[1] + bounds[3]) / 2
            logger.info(f"Map center: lat={center_lat}, lon={center_lon}")
        else:
            logger.warning("Invalid bounds, using default center")
            center_lat, center_lon = default_centers.get(state.lower(), [39.0, -105.5])

        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='CartoDB positron'
        )
        
        # Add base maps
        base_maps = {
            'Light': 'CartoDB positron',
            'Dark': 'CartoDB dark_matter',
            'Satellite': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
        }
        
        for name, url in base_maps.items():
            if name == 'Satellite':
                folium.TileLayer(url, name=name, attr='Esri').add_to(m)
            else:
                folium.TileLayer(url, name=name).add_to(m)
        
        # Add mouse position
        MousePosition(
            position="topright",
            separator=" | ",
            empty_string="NaN",
            lng_first=True,
            num_digits=20,
            prefix="Coordinates:",
            lat_formatter=COORDINATE_FORMATTER,
            lng_formatter=COORDINATE_FORMATTER,
        ).add_to(m)
        
        # Add fullscreen button
        Fullscreen(
            position="topright",
            title="Expand me",
            title_cancel="Exit me",
            force_separate_button=True,
        ).add_to(m)
        
        # Create a color scale for normalized rank
        if 'parcel_rank_normalized' in parcels.columns:
            # Create a simple red-to-green color scale for normalized ranks (1-10)
            colormap = cm.LinearColormap(
                colors=['red', 'yellow', 'green'],
                vmin=1,
                vmax=10,
                caption='Normalized Rank (1-10)'
            )
            colormap.add_to(m)
            
            # Create a style function that uses the color scale
            def style_function(feature):
                norm_rank = feature['properties']['parcel_rank_normalized']
                return {
                    'fillColor': colormap(norm_rank),
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.7
                }
            
            # Group parcels by rank ranges to reduce the number of layers
            rank_ranges = [
                (0, 2, 'Rank 0-2'),
                (2, 4, 'Rank 2-4'),
                (4, 6, 'Rank 4-6'),
                (6, 8, 'Rank 6-8'),
                (8, 11, 'Rank 8-10')
            ]
            
            for min_rank, max_rank, name in rank_ranges:
                group_parcels = parcels[
                    (parcels['parcel_rank_normalized'] >= min_rank) & 
                    (parcels['parcel_rank_normalized'] < max_rank)
                ]
                
                if not group_parcels.empty:
                    folium.GeoJson(
                        group_parcels.__geo_interface__,
                        style_function=style_function,
                        popup=folium.GeoJsonPopup(
                            fields=['parcelnumb', 'parcel_rank_normalized', 'zoning_type', 'zoning', 'zoning_description', 'zoning_code_link', 'lbcs_activity_desc'],
                            aliases=['APN:', 'Normalized Rank:', 'Zoning Type - Zoneomics:', 'Zoning Code - County:', 'Zoning Description - County:', 'Zoning Code Link:', 'Land Use - Zoneomics:'],
                            style="background-color: white; padding: 5px;",
                            max_width=275
                        ),
                        name=name,
                        show=(min_rank >= 6)  # Only show high ranks by default
                    ).add_to(m)
        
        # Add isochrones if provided
        if isochrones is not None:
            if isochrones.crs is not None and isochrones.crs.to_epsg() != 4326:
                isochrones = isochrones.to_crs(epsg=4326)
            
            # Add isochrones with different colors for each time
            for idx, row in isochrones.iterrows():
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda x: {
                        'fillColor': 'blue',
                        'color': 'blue',
                        'weight': 2,
                        'fillOpacity': 0.1
                    },
                    name=f'Isochrone {row["time"]} minutes',
                    show=True
                ).add_to(m)
        
        # Add substations from rextag.electricsubstations, cropped to parcel bounds
        if db_connection is not None and not any(pd.isna(bounds)) and not any(np.isinf(bounds)):
            try:
                minx, miny, maxx, maxy = bounds
                substations_query = f"""
                    SELECT *, ST_AsText(geom) as geom_wkt
                    FROM rextag.electricsubstations
                    WHERE ST_Within(geom, ST_MakeEnvelope({minx}, {miny}, {maxx}, {maxy}, 4326))
                """
                substations_df = db_connection.execute(substations_query).df()
                if not substations_df.empty:
                    substations_df['geometry'] = gpd.GeoSeries.from_wkt(substations_df['geom_wkt'])
                    substations_gdf = gpd.GeoDataFrame(substations_df.drop(columns=['geom_wkt']), geometry='geometry', crs=4326)
                    for idx, row in substations_gdf.iterrows():
                        folium.Marker(
                            location=[row.geometry.y, row.geometry.x],
                            popup=folium.Popup(
                                f"<b>Substation</b><br>"
                                f"Name: {row.get('name', 'N/A')}<br>"
                                f"Voltage: {row.get('voltage', 'N/A')}<br>"
                                f"Operator: {row.get('operator', 'N/A')}",
                                max_width=200
                            ),
                            icon=folium.Icon(color='red', icon='bolt', prefix='fa'),
                            name='Substations'
                        ).add_to(m)
            except Exception as e:
                logger.warning(f"Could not load substations: {str(e)}")
        
        # Add POI data if provided
        if poi_data is not None:
            for poi_type, poi_gdf in poi_data.items():
                if poi_gdf.crs is not None and poi_gdf.crs.to_epsg() != 4326:
                    poi_gdf = poi_gdf.to_crs(epsg=4326)
                
                # Get icon and color for this POI type
                icon_name = poi_icons.get(poi_type, 'info-sign') if poi_icons else 'info-sign'
                color = poi_colors.get(poi_type, 'blue') if poi_colors else 'blue'
                
                # Add POIs to map
                for idx, row in poi_gdf.iterrows():
                    folium.Marker(
                        location=[row.geometry.y, row.geometry.x],
                        popup=folium.Popup(
                            f"<b>{poi_type}</b><br>"
                            f"Name: {row.get('name', 'N/A')}<br>"
                            f"Address: {row.get('address', 'N/A')}",
                            max_width=200
                        ),
                        icon=folium.Icon(color=color, icon=icon_name, prefix='glyphicon'),
                        name=poi_type
                    ).add_to(m)
        
        # Add layer control
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Save the map
        output_file = output_dir / f"{state}_{county}_parcel_map.html"
        m.save(str(output_file))
        logger.info(f"Created interactive map at {output_file}")
        
        # Open the map in the default web browser
        webbrowser.open(f'file://{os.path.abspath(output_file)}')
        
    except Exception as e:
        logger.error(f"Error creating parcel map: {str(e)}")
        raise 