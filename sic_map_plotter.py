#!/usr/bin/env python3
"""
Advanced SIC Map Plotter
Creates daily SIC maps with ship trajectory overlay and combines them into a movie.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sic_extractor import SICExtractor


class SICMapPlotter:
    """Advanced SIC map plotter with ship trajectory overlay."""
    
    def __init__(self, data_dir='./data', output_dir='./output'):
        """Initialize the SIC map plotter."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.maps_dir = self.output_dir / 'daily_maps'
        self.maps_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SIC extractor for downloading files
        self.sic_extractor = SICExtractor(data_dir, output_dir)
        
        # Create separate maps data directory to avoid overwriting original files
        self.maps_data_dir = self.output_dir / 'maps_data'
        self.maps_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Radius in kilometers
        self.radius_km = 50
        
    def degrees_to_km(self, lat):
        """Convert degrees to kilometers at given latitude."""
        # Approximate conversion (varies with latitude)
        km_per_degree_lat = 111.32
        km_per_degree_lon = 111.32 * np.cos(np.radians(lat))
        return km_per_degree_lat, km_per_degree_lon
    
    def get_bounding_box(self, lat, lon, radius_km):
        """Get bounding box for given center point and radius."""
        km_per_deg_lat, km_per_deg_lon = self.degrees_to_km(lat)
        
        # Calculate degree offsets
        lat_offset = radius_km / km_per_deg_lat
        lon_offset = radius_km / km_per_deg_lon
        
        # Create bounding box
        min_lon = lon - lon_offset
        max_lon = lon + lon_offset
        min_lat = lat - lat_offset
        max_lat = lat + lat_offset
        
        return min_lon, min_lat, max_lon, max_lat
    
    def extract_sic_region(self, raster_path, center_lat, center_lon, radius_km):
        """Extract SIC data within radius around center point."""
        try:
            with rasterio.open(raster_path) as src:
                # Get coordinate transformer
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                
                # Transform center point to raster CRS
                center_x, center_y = transformer.transform(center_lon, center_lat)
                center_row, center_col = src.index(center_x, center_y)
                
                print(f"Center point: {center_lat:.3f}, {center_lon:.3f}")
                print(f"Center in raster CRS: {center_x:.1f}, {center_y:.1f}")
                print(f"Center pixel: row={center_row}, col={center_col}")
                
                # Check if center is within raster bounds
                print(f"Raster dimensions: {src.width}x{src.height}")
                if (center_row < 0 or center_row >= src.height or 
                    center_col < 0 or center_col >= src.width):
                    print(f"Center point is outside raster bounds (need 0-{src.height-1}, 0-{src.width-1})")
                    return None
                
                # Create a fixed-size window around the center point
                # Use pixel size to approximate 75km radius
                # For 1km resolution, 75km = ~75 pixels
                window_size = 75
                
                row_start = max(0, center_row - window_size)
                row_end = min(src.height, center_row + window_size)
                col_start = max(0, center_col - window_size)
                col_end = min(src.width, center_col + window_size)
                
                print(f"Window: rows {row_start}-{row_end}, cols {col_start}-{col_end}")
                
                # Create window
                window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
                
                # Read the data
                data = src.read(1, window=window)
                
                if data.size == 0:
                    print("No data in window")
                    return None
                
                # Get the transform for the windowed data
                window_transform = src.window_transform(window)
                
                # Create coordinate arrays
                height, width = data.shape
                cols, rows = np.meshgrid(np.arange(width), np.arange(height))
                
                # Transform pixel coordinates to geographic coordinates
                xs, ys = rasterio.transform.xy(window_transform, rows, cols)
                xs = np.array(xs)
                ys = np.array(ys)
                
                # Transform back to WGS84
                transformer_back = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                lons, lats = transformer_back.transform(xs, ys)
                
                # Clean data (clamp to valid range)
                data = np.where(data > 100, 100, data)
                data = np.where(data < 0, 0, data)
                
                # Calculate actual bounds from the data
                min_lon = np.min(lons)
                max_lon = np.max(lons)
                min_lat = np.min(lats)
                max_lat = np.max(lats)
                
                print(f"Extracted region: {data.shape} pixels")
                print(f"Data bounds: {min_lon:.3f}, {min_lat:.3f}, {max_lon:.3f}, {max_lat:.3f}")
                
                return {
                    'data': data,
                    'lons': lons,
                    'lats': lats,
                    'center_lat': center_lat,
                    'center_lon': center_lon,
                    'bounds': (min_lon, min_lat, max_lon, max_lat)
                }
                
        except Exception as e:
            print(f"Error extracting SIC region: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def plot_daily_map(self, sic_data, ship_positions, date, save_path):
        """Create daily SIC map with ship trajectory."""
        if sic_data is None:
            print(f"No SIC data available for {date}")
            return False
            
        # Create figure with regular matplotlib
        plt.figure(figsize=(12, 10))
        
        # Prepare data for plotting with custom colormap
        plot_data = sic_data['data'].copy()
        
        # Create custom colormap with three segments
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm
        
        # Prepare data for three-segment colormap
        plot_data_custom = plot_data.copy()
        
        # Create custom colormap with focus on 80-100% range
        from matplotlib.colors import LinearSegmentedColormap
        
        # Define color segments with more focus on 80-100% range
        # Less variation in 0-80%, more variation in 80-100%
        colors = [
            (0.0, 'lightcoral'),    # 0-80%: single coral color (80% of range)
            (0.8, 'lightcoral'),    # 
            (0.85, 'yellow'),       # 80-85%: transition to yellow
            (0.9, 'lightblue'),     # 85-90%: transition to light blue
            (0.95, 'blue'),         # 90-95%: transition to blue
            (1.0, 'darkblue')       # 95-100%: transition to dark blue
        ]
        
        # Create segmented colormap with fixed color positions
        cmap_custom = LinearSegmentedColormap.from_list(
            'sic_focused', colors, N=256
        )
        
        # Fixed vmin/vmax for all figures
        vmin, vmax = 0, 100
        
        # Plot SIC data as image
        extent = [sic_data['bounds'][0], sic_data['bounds'][2], 
                  sic_data['bounds'][1], sic_data['bounds'][3]]
        
        im = plt.imshow(
            plot_data_custom, 
            extent=extent,
            cmap=cmap_custom, vmin=vmin, vmax=vmax, alpha=0.9,
            origin='upper', aspect='auto'
        )
        
        # Add colorbar with custom ticks
        cbar = plt.colorbar(im, shrink=0.8, pad=0.05)
        cbar.set_label('Sea Ice Concentration (%)', fontsize=12)
        
        # Set fixed colorbar ticks and labels for all figures
        tick_positions = [0, 80, 85, 90, 95, 100]
        tick_labels = ['0%', '80%', '85%', '90%', '95%', '100%']
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels)
        
        # Set fixed colorbar limits to ensure consistency across all figures
        im.set_clim(vmin=0, vmax=100)
        
        # Get map bounds for clipping other days' tracks
        map_bounds = sic_data['bounds']
        min_lon, min_lat, max_lon, max_lat = map_bounds
        
        # Plot historical/future track (yellow line) - draw first so it's under current day
        # Filter ship positions to map bounds and get ±10 days around current date
        date_range_start = date - pd.Timedelta(days=10)
        date_range_end = date + pd.Timedelta(days=10)
        
        # Get all positions within date range and map bounds
        nearby_dates = ship_positions[
            (ship_positions['datetime'].dt.date >= date_range_start) &
            (ship_positions['datetime'].dt.date <= date_range_end) &
            (ship_positions['datetime'].dt.date != date) &  # Exclude current day
            (ship_positions['longitude'] >= min_lon) &
            (ship_positions['longitude'] <= max_lon) &
            (ship_positions['latitude'] >= min_lat) &
            (ship_positions['latitude'] <= max_lat)
        ]
        
        if not nearby_dates.empty:
            # Plot the background track in yellow
            plt.plot(
                nearby_dates['longitude'], nearby_dates['latitude'],
                'y-', linewidth=3, alpha=0.8, zorder=1, label='Ship Track (±10 days)'
            )
            
            # Add day offset markers
            # Group by date and get one point per day
            daily_nearby = nearby_dates.groupby(nearby_dates['datetime'].dt.date).agg({
                'longitude': 'mean',
                'latitude': 'mean'
            }).reset_index()
            daily_nearby.rename(columns={'datetime': 'date'}, inplace=True)
            
            # Calculate day offsets and add markers
            for _, row in daily_nearby.iterrows():
                day_offset = (row['date'] - date).days
                
                # Only show markers for significant offsets (every few days)
                if abs(day_offset) % 2 == 0 or abs(day_offset) <= 3:  # Every 2 days or within 3 days
                    marker_size = 25 if abs(day_offset) <= 3 else 20
                    
                    # Plot marker with day offset
                    plt.scatter(
                        row['longitude'], row['latitude'],
                        c='gold', s=marker_size, marker='o', 
                        edgecolor='darkorange', linewidth=1,
                        zorder=2, alpha=0.9
                    )
                    
                    # Add larger text with day offset
                    offset_text = f"{day_offset:+d}" if day_offset != 0 else "0"
                    plt.text(
                        row['longitude'], row['latitude'],
                        offset_text, fontsize=9, ha='center', va='center',
                        color='black', fontweight='bold', zorder=3
                    )
        
        # Plot current day ship track (red line) - draw on top
        current_day_data = ship_positions[ship_positions['datetime'].dt.date == date]
        if not current_day_data.empty:
            plt.plot(
                current_day_data['longitude'], current_day_data['latitude'],
                'r-', linewidth=3, alpha=0.9, label='Ship Track (Current Day)', zorder=4
            )
            
            # Add start and end points for the day
            if len(current_day_data) > 1:
                start_point = current_day_data.iloc[0]
                end_point = current_day_data.iloc[-1]
                
                plt.scatter(
                    start_point['longitude'], start_point['latitude'],
                    c='green', s=80, marker='o', edgecolor='darkgreen', linewidth=2,
                    label='Day Start', zorder=6
                )
                plt.scatter(
                    end_point['longitude'], end_point['latitude'],
                    c='red', s=80, marker='s', edgecolor='darkred', linewidth=2,
                    label='Day End', zorder=6
                )
        
        # Add center point
        plt.scatter(
            sic_data['center_lon'], sic_data['center_lat'],
            c='yellow', s=100, marker='*', edgecolor='black', linewidth=2,
            label='Daily Center', zorder=7
        )
        
        # Add 10km scale bars (horizontal and vertical)
        # Calculate 10km in degrees (approximate)
        km_per_degree_lat = 111.32
        km_per_degree_lon = 111.32 * np.cos(np.radians(sic_data['center_lat']))
        
        scale_lat = 10 / km_per_degree_lat  # 10km in latitude degrees
        scale_lon = 10 / km_per_degree_lon  # 10km in longitude degrees
        
        # Position scale bars at bottom left of the map
        map_bounds = sic_data['bounds']
        scale_x = map_bounds[0] + 0.1 * (map_bounds[2] - map_bounds[0])
        scale_y = map_bounds[1] + 0.1 * (map_bounds[3] - map_bounds[1])
        
        # Horizontal scale bar (10km)
        plt.plot([scale_x, scale_x + scale_lon], [scale_y, scale_y], 
                'k-', linewidth=3, solid_capstyle='butt')
        plt.text(scale_x + scale_lon/2, scale_y - 0.02 * (map_bounds[3] - map_bounds[1]), 
                '10 km', ha='center', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Vertical scale bar (10km)
        plt.plot([scale_x, scale_x], [scale_y, scale_y + scale_lat], 
                'k-', linewidth=3, solid_capstyle='butt')
        plt.text(scale_x - 0.02 * (map_bounds[2] - map_bounds[0]), scale_y + scale_lat/2, 
                '10 km', ha='right', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                rotation=90)
        
        # Set labels and grid
        plt.xlabel('Longitude', fontsize=12)
        plt.ylabel('Latitude', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add title and legend
        plt.title(f'Sea Ice Concentration - {date.strftime("%Y-%m-%d")}', fontsize=14, fontweight='bold')
        plt.legend(loc='upper left')
        
        # Save plot
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Daily map saved: {save_path}")
        return True
    
    def create_daily_maps(self, position_file, file_type='mosaic'):
        """Create daily SIC maps with ship trajectory overlay."""
        print("Creating daily SIC maps...")
        
        # Read position data
        if file_type == 'mosaic':
            positions_df = self.sic_extractor.parse_position_mosaic(position_file)
        else:
            positions_df = self.sic_extractor.parse_csv_positions(position_file)
        
        # Get daily average positions
        daily_positions = self.sic_extractor.daily_average_positions(positions_df)
        
        # Get required dates
        required_dates = self.sic_extractor.get_required_dates(daily_positions)
        
        created_maps = []
        
        for date in sorted(required_dates):
            print(f"Processing date: {date}")
            
            # Get daily center position
            day_data = daily_positions[daily_positions['date'] == date]
            if day_data.empty:
                continue
                
            center_lat = day_data['latitude'].iloc[0]
            center_lon = day_data['longitude'].iloc[0]
            
            # Download SIC file for this date
            sic_file_path = self.sic_extractor.download_sic_file(date)
            if not sic_file_path:
                print(f"No SIC data available for {date}")
                continue
            
            # Create a safe copy reference (don't actually copy the file)
            # Just use the original path for reading
            print(f"Using SIC file: {sic_file_path}")
            
            # Extract SIC data around ship position
            sic_data = self.extract_sic_region(
                sic_file_path, center_lat, center_lon, self.radius_km
            )
            
            # Create daily map
            map_filename = f"sic_map_{date.strftime('%Y%m%d')}.png"
            map_path = self.maps_dir / map_filename
            
            if self.plot_daily_map(sic_data, positions_df, date, map_path):
                created_maps.append({
                    'date': date,
                    'path': map_path,
                    'filename': map_filename
                })
        
        return created_maps
    
    def create_movie(self, created_maps, output_filename='sic_movie.gif'):
        """Create animated movie from daily maps."""
        if not created_maps:
            print("No maps to create movie from")
            return
        
        print(f"Creating movie from {len(created_maps)} daily maps...")
        
        # Sort maps by date
        created_maps.sort(key=lambda x: x['date'])
        
        # Create figure for animation
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.axis('off')
        
        def animate(frame):
            ax.clear()
            ax.axis('off')
            
            # Load and display image
            img = plt.imread(created_maps[frame]['path'])
            ax.imshow(img)
            
            # Add frame info
            date_str = created_maps[frame]['date'].strftime('%Y-%m-%d')
            ax.text(0.02, 0.98, f"Date: {date_str}", transform=ax.transAxes,
                   fontsize=14, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            return [ax]
        
        # Create animation
        anim = FuncAnimation(
            fig, animate, frames=len(created_maps),
            interval=1000, blit=False, repeat=True
        )
        
        # Save as GIF
        movie_path = self.output_dir / output_filename
        writer = PillowWriter(fps=1)
        anim.save(movie_path, writer=writer)
        
        plt.close()
        print(f"Movie saved: {movie_path}")
        
        return movie_path
    
    def process_full_analysis(self, position_file, file_type='mosaic'):
        """Run complete analysis: create daily maps and movie."""
        print("Starting full SIC map analysis...")
        
        # Create daily maps
        created_maps = self.create_daily_maps(position_file, file_type)
        
        if not created_maps:
            print("No maps were created")
            return
        
        # Create movie
        movie_path = self.create_movie(created_maps)
        
        print(f"\nAnalysis complete!")
        print(f"Daily maps: {len(created_maps)} files in {self.maps_dir}")
        print(f"Movie: {movie_path}")
        
        return created_maps, movie_path


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python sic_map_plotter.py <position_file> [file_type]")
        print("file_type: 'mosaic' (default) or 'csv'")
        sys.exit(1)
    
    position_file = sys.argv[1]
    file_type = sys.argv[2] if len(sys.argv) > 2 else 'mosaic'
    
    if not os.path.exists(position_file):
        print(f"Position file not found: {position_file}")
        sys.exit(1)
    
    # Create map plotter
    plotter = SICMapPlotter()
    
    # Run full analysis
    plotter.process_full_analysis(position_file, file_type)


if __name__ == "__main__":
    main()