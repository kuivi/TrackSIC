#!/usr/bin/env python3
"""
Sea Ice Concentration (SIC) Extractor
Extract, save and plot SIC data along a track from GeoTIFF files.
"""

import os
import sys
import pandas as pd
import numpy as np
import requests
import rasterio
from datetime import datetime
import matplotlib.pyplot as plt
from pyproj import Transformer
import geopandas as gpd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class SICExtractor:
    """Main class for extracting SIC data along a track."""
    
    def __init__(self, data_dir='./data', output_dir='./output'):
        """Initialize the SIC extractor."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Base URLs for SIC data
        self.base_url = 'https://data.seaice.uni-bremen.de/modis_amsr2/geotiff/Arctic'
        self.fallback_url = 'https://data.seaice.uni-bremen.de/modis_amsr2/geotiff/MOSAiC_StartRegion'
        
    def parse_position_mosaic(self, filepath):
        """Parse position_mosaic.dat file format."""
        print(f"Reading position file: {filepath}")
        
        positions = []
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        # Skip header
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 8:
                date_str = parts[0]
                time_str = parts[1]
                lat_deg = float(parts[2])
                lat_min = float(parts[3])
                lat_hem = parts[4]
                lon_deg = float(parts[5])
                lon_min = float(parts[6])
                lon_hem = parts[7]
                
                # Convert to decimal degrees
                lat = lat_deg + lat_min / 60.0
                if lat_hem == 'S':
                    lat = -lat
                    
                lon = lon_deg + lon_min / 60.0
                if lon_hem == 'W':
                    lon = -lon
                
                # Parse datetime
                dt = datetime.strptime(f"{date_str} {time_str}", "%Y/%m/%d %H:%M:%S")
                
                positions.append({
                    'datetime': dt,
                    'latitude': lat,
                    'longitude': lon
                })
        
        return pd.DataFrame(positions)
    
    def parse_csv_positions(self, filepath):
        """Parse CSV-like position files with datetime, latitude, longitude."""
        print(f"Reading CSV position file: {filepath}")
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df[['datetime', 'latitude', 'longitude']]
    
    def daily_average_positions(self, df):
        """Calculate daily average positions."""
        print("Calculating daily averages...")
        df['date'] = df['datetime'].dt.date
        
        daily_avg = df.groupby('date').agg({
            'latitude': 'mean',
            'longitude': 'mean'
        }).reset_index()
        
        print(f"Reduced {len(df)} positions to {len(daily_avg)} daily averages")
        return daily_avg
    
    def get_required_dates(self, df):
        """Get unique dates for which SIC data is needed."""
        dates = df['date'].unique()
        print(f"Need SIC data for {len(dates)} dates")
        return dates
    
    def check_file_exists(self, url):
        """Check if a file exists on the server."""
        try:
            response = requests.head(url, timeout=10)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def download_sic_file(self, date):
        """Download SIC file for a specific date."""
        date_str = date.strftime('%Y%m%d')
        year = date.strftime('%Y')
        filename = f"sic_modis-aqua_amsr2-gcom-w1_merged_nh_1000m_{date_str}.tif"
        local_path = self.data_dir / filename
        
        if local_path.exists():
            print(f"File already exists: {filename}")
            return local_path
        
        # Try primary URL first
        url = f"{self.base_url}/{year}/{filename}"
        print(f"Checking if file exists: {url}")
        
        if self.check_file_exists(url):
            print(f"File found, downloading: {url}")
            return self._download_file(url, local_path, filename)
        
        # Try fallback URL with different filename
        fallback_filename = f"sic_modis-aqua_amsr2-gcom-w1_merged_nh_1000m_{date_str}_mosaic_startregion.tif"
        fallback_url = f"{self.fallback_url}/{year}/{fallback_filename}"
        print(f"Primary URL failed, checking fallback: {fallback_url}")
        
        if self.check_file_exists(fallback_url):
            print(f"File found at fallback, downloading: {fallback_url}")
            return self._download_file(fallback_url, local_path, fallback_filename)
        
        print(f"No SIC data found for {date_str} at either URL")
        return None
    
    def _download_file(self, url, local_path, filename):
        """Download a file from URL to local path."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Downloaded: {filename}")
            return local_path
            
        except requests.RequestException as e:
            print(f"Failed to download {filename}: {e}")
            return None
    
    def extract_sic_value(self, raster_path, lat, lon):
        """Extract SIC value from raster at given coordinates."""
        try:
            with rasterio.open(raster_path) as src:
                # Transform coordinates from WGS84 to raster CRS
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = transformer.transform(lon, lat)
                #print(f"CRS: {src.crs}")
                #print(f"Transformed coordinates: {lon}, {lat} ->  {x}, {y}")
                # Sample the raster at the transformed coordinates
                sampled = list(src.sample([(x, y)]))
                if sampled:
                    value = sampled[0][0]  # Band 1
                    # Clamp to valid range [0, 100]
                    if value > 100:
                        value = 100
                    elif value < 0:
                        value = 0
                    return value
                else:
                    return np.nan
                    
        except Exception as e:
            print(f"Error extracting SIC value: {e}")
            return np.nan
    
    def process_track(self, position_file, file_type='mosaic'):
        """Process the entire track."""
        print("Starting SIC extraction process...")
        
        # Read positions
        if file_type == 'mosaic':
            positions_df = self.parse_position_mosaic(position_file)
        else:
            positions_df = self.parse_csv_positions(position_file)
        
        # Calculate daily averages
        daily_positions = self.daily_average_positions(positions_df)
        
        # Get required dates
        required_dates = self.get_required_dates(daily_positions)
        
        # Download SIC files
        sic_files = {}
        for date in required_dates:
            file_path = self.download_sic_file(date)
            if file_path:
                sic_files[date] = file_path
        
        # Extract SIC values
        results = []
        for _, row in daily_positions.iterrows():
            date = row['date']
            lat = row['latitude']
            lon = row['longitude']
            
            if date in sic_files:
                sic_value = self.extract_sic_value(sic_files[date], lat, lon)
                sic_display = f"{sic_value:.1f}" if not np.isnan(sic_value) else "NaN"
                print(f"Date: {date}, Lat: {lat:.3f}, Lon: {lon:.3f}, SIC: {sic_display}")
            else:
                sic_value = None
                print(f"Date: {date}, Lat: {lat:.3f}, Lon: {lon:.3f}, SIC: No data available")
            
            results.append({
                'date': date,
                'latitude': lat,
                'longitude': lon,
                'sic_value': sic_value
            })
        
        return pd.DataFrame(results)
    
    def save_results(self, results_df, filename='sic_results.csv'):
        """Save results to CSV file."""
        output_path = self.output_dir / filename
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
        return output_path
    
    def plot_results(self, results_df, filename='sic_plot.png'):
        """Plot SIC values over time."""
        output_path = self.output_dir / filename
        
        # Filter out NaN and None values
        valid_data = results_df.dropna(subset=['sic_value'])
        valid_data = valid_data[valid_data['sic_value'].notna()]
        
        if len(valid_data) == 0:
            print("No valid SIC data to plot")
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(valid_data['date'], valid_data['sic_value'], 'b-o', markersize=4)
        plt.title('Sea Ice Concentration Along Track')
        plt.xlabel('Date')
        plt.ylabel('SIC (%)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
        
        # Show statistics
        total_points = len(results_df)
        missing_points = len(results_df[results_df['sic_value'].isna()])
        
        print(f"\nSIC Statistics:")
        print(f"Total data points: {total_points}")
        print(f"Valid data points: {len(valid_data)}")
        print(f"Missing data points: {missing_points}")
        
        if len(valid_data) > 0:
            print(f"Mean SIC: {valid_data['sic_value'].mean():.1f}%")
            print(f"Min SIC: {valid_data['sic_value'].min():.1f}%")
            print(f"Max SIC: {valid_data['sic_value'].max():.1f}%")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python sic_extractor.py <position_file> [file_type]")
        print("file_type: 'mosaic' (default) or 'csv'")
        sys.exit(1)
    
    position_file = sys.argv[1]
    file_type = sys.argv[2] if len(sys.argv) > 2 else 'mosaic'
    
    if not os.path.exists(position_file):
        print(f"Position file not found: {position_file}")
        sys.exit(1)
    
    # Create extractor
    extractor = SICExtractor()
    
    # Process track
    results = extractor.process_track(position_file, file_type)
    
    # Save and plot results
    if not results.empty:
        extractor.save_results(results)
        extractor.plot_results(results)
        print("\nSIC extraction completed successfully!")
    else:
        print("No results to save or plot")


if __name__ == "__main__":
    main()