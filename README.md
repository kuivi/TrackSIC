# TrackSIC

A Python tool for extracting sea ice concentration data along vessel tracks. This tool processes satellite-derived sea ice concentration data from the University of Bremen and correlates it with ship position data to create time series and spatial visualizations.

## Data Sources

### Sea Ice Concentration Data
- **Source**: University of Bremen MODIS-Aqua/AMSR2 merged products
- **URL**: https://data.seaice.uni-bremen.de/modis_amsr2/geotiff/Arctic/
- **Format**: GeoTIFF files with 1km spatial resolution
- **Coverage**: Northern Hemisphere Arctic Ocean
- **Temporal Resolution**: Daily
- **Data Range**: 0-100% sea ice concentration

### Position Data
- **Format**: Custom position files with degrees/minutes notation or standard CSV
- **Required Fields**: Date, time, latitude, longitude
- **Projection**: WGS84 (EPSG:4326)

## Features

### Core Analysis
- Automated download of sea ice concentration data
- Coordinate transformation between WGS84 and polar stereographic projections
- Daily averaging of high-frequency position data
- Extraction of ice concentration values at vessel positions
- Statistical analysis and data quality control

### Visualization
- Time series plots of sea ice concentration along track
- Daily spatial maps showing ice conditions in 75km radius around vessel
- Temporal context with ±10-day historical tracks
- Animated sequences showing ice evolution over time
- Consistent color scaling optimized for high ice concentration detection (80-100%)

## Installation

```bash
git clone https://github.com/kuivi/TrackSIC.git
cd TrackSIC
pip install -r requirements.txt
```

## Usage

### Basic Analysis
Extract sea ice concentration along vessel track:
```bash
python sic_extractor.py position_mosaic.dat
```

### Advanced Visualization
Create daily maps and animation:
```bash
python sic_map_plotter.py position_mosaic.dat
```

### Input File Formats

#### Position Mosaic Format
```
date time    latd    latm    hemlat  lond    lonm    hemlon
2019/09/20 15:20:00 69 40.772868 N 018 59.799306 E
2019/09/20 15:30:00 69 40.772844 N 018 59.799234 E
```

#### CSV Format
```
datetime,latitude,longitude
2019-09-20 15:20:00,69.679547,18.996655
2019-09-20 15:30:00,69.679547,18.996654
```

## Output

### Data Files
- `output/sic_results.csv`: Extracted SIC values with coordinates and dates
- `output/sic_plot.png`: Time series visualization

### Spatial Analysis
- `output/daily_maps/`: Individual daily maps (when using map plotter)
- `output/sic_movie.gif`: Animated sequence of daily ice conditions

## Technical Details

### Coordinate Systems
- Input coordinates: WGS84 (EPSG:4326)
- SIC data projection: NSIDC Sea Ice Polar Stereographic North (EPSG:3411)
- Automatic coordinate transformation using pyproj

### Data Processing
- Temporal aggregation to daily mean positions
- Spatial extraction within 75km radius of vessel positions
- Data validation and quality control (0-100% range clamping)
- Intelligent file caching to avoid redundant downloads

### Dependencies
- pandas: Data manipulation and analysis
- numpy: Numerical computations
- rasterio: Geospatial raster data processing
- pyproj: Coordinate reference system transformations
- matplotlib: Visualization and plotting
- requests: HTTP data downloading

## Example Results

The toolkit generates:
1. **Time series**: ice concentration along the track
2. **Spatial context maps**: Daily ice conditions with vessel position and historical context
3. **Animated visualizations**: Temporal evolution of ice conditions



