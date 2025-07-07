# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Arctic Sea Ice Concentration Analysis is a comprehensive toolkit for extracting, analyzing, and visualizing sea ice concentration data along vessel tracks in polar regions. The system processes satellite-derived sea ice concentration data from the University of Bremen and correlates it with ship position data to create detailed temporal and spatial analyses.

## Project Purpose

This toolkit serves Arctic research applications including:
- Analysis of sea ice conditions along research vessel tracks during polar expeditions
- Temporal monitoring and statistical analysis of ice concentration changes
- Generation of publication-ready visualizations for oceanographic and climate studies
- Support for navigation planning and risk assessment in ice-covered waters
- Educational materials for polar science outreach

## Core Architecture

The project consists of two main components:

### 1. SIC Extractor (`sic_extractor.py`)
Basic extraction and time series analysis:
1. **Position Data Processing**: Parses custom `position_mosaic.dat` (degrees/minutes/hemispheres) and standard CSV formats
2. **Temporal Aggregation**: Converts high-frequency position data to daily averages
3. **Data Acquisition**: Downloads SIC GeoTIFF files from Bremen repository with fallback URLs and intelligent caching
4. **Coordinate Transformation**: WGS84 to NSIDC Sea Ice Polar Stereographic North projection
5. **Value Extraction**: Samples SIC values from Band 1 using rasterio with proper bounds checking
6. **Output Generation**: CSV results and matplotlib time series plots

### 2. Advanced Map Plotter (`sic_map_plotter.py`)
Spatial visualization and animation:
1. **Regional Data Extraction**: 150x150 pixel (~75km radius) windows around daily positions
2. **Multi-temporal Context**: Shows ±10 days of vessel track with day offset markers
3. **Spatial Mapping**: Daily ice concentration maps with consistent color scaling
4. **Animation Generation**: Creates GIF movies showing temporal ice evolution
5. **Enhanced Visualization**: Custom colormap optimized for 80-100% ice concentration detection

## Key Commands

### Installation
```bash
pip install -r requirements.txt
```

### Basic Analysis
```bash
# Extract SIC along track
python sic_extractor.py position_mosaic.dat

# For CSV format
python sic_extractor.py track_data.csv csv
```

### Advanced Spatial Analysis
```bash
# Create daily maps and animation
python sic_map_plotter.py position_mosaic.dat
```

## Data Sources and Processing

### Input Data
- **Position files**: Custom format (degrees/minutes/hemisphere) or standard CSV (decimal degrees)
- **SIC data**: University of Bremen MODIS-Aqua/AMSR2 merged products
- **Primary URL**: https://data.seaice.uni-bremen.de/modis_amsr2/geotiff/Arctic/YYYY/
- **Fallback URL**: https://data.seaice.uni-bremen.de/modis_amsr2/geotiff/MOSAiC_StartRegion/YYYY/
- **File naming**: `sic_modis-aqua_amsr2-gcom-w1_merged_nh_1000m_YYYYMMDD[_mosaic_startregion].tif`

### Output Structure
- **Basic results**: `output/sic_results.csv`, `output/sic_plot.png`
- **Spatial analysis**: `output/daily_maps/sic_map_YYYYMMDD.png`, `output/sic_movie.gif`
- **Data caching**: `./data/` directory for downloaded GeoTIFF files

## Technical Implementation

### Coordinate Systems
- **Input**: WGS84 (EPSG:4326)
- **SIC data**: NSIDC Sea Ice Polar Stereographic North (EPSG:3411)
- **Transformation**: pyproj with proper bounds validation

### Data Processing Features
- **Quality Control**: SIC values clamped to [0, 100] range
- **Error Handling**: Graceful handling of missing data, failed downloads, and out-of-bounds coordinates
- **Caching Strategy**: Intelligent file caching with existence checking before download
- **Temporal Filtering**: Daily averaging with configurable date ranges
- **Spatial Extraction**: Robust window creation with bounds validation

### Visualization Features
- **Consistent Colormapping**: Custom colormap with focused gradient for 80-100% ice concentration
- **Multi-layer Display**: Current day track (red), historical context (yellow), day offset markers
- **Scale References**: 10km scale bars for spatial context
- **Animation**: Temporal sequences showing ice evolution

## Dependencies

Core geospatial and scientific Python libraries:
- `rasterio>=1.2.0`: GeoTIFF processing and spatial windowing
- `pyproj>=3.0.0`: Coordinate reference system transformations
- `pandas>=1.3.0`: Data manipulation and temporal processing
- `matplotlib>=3.3.0`: Visualization and animation
- `numpy>=1.20.0`: Numerical computations
- `requests>=2.25.0`: HTTP data downloading with error handling
- `geopandas>=0.9.0`: Geospatial data handling
- `Pillow>=8.0.0`: Image processing for animations

## Development Notes

### Critical Implementation Details
- SIC data temporal resolution is daily; position data must be aggregated accordingly
- Raster projection is polar stereographic, requiring careful coordinate transformation
- GeoTIFF files contain 8-bit values in Band 1, with valid SIC range [0, 100]
- Some files may have values >100 that should be clamped
- Window extraction requires proper bounds validation to avoid "bounds and transform inconsistent" errors

### Performance Considerations
- Large GeoTIFF files (up to 5560x4750 pixels) require efficient windowing
- Download operations should include existence checking and fallback URLs
- Animation generation processes many daily maps sequentially

### Error Handling Patterns
- Coordinate transformation failures should be caught and logged
- Missing data dates should be marked as None in output
- Network failures should attempt fallback URLs before giving up
- Out-of-bounds positions should be handled gracefully

## File Organization for GitHub

### Included Files
- Core scripts: `sic_extractor.py`, `sic_map_plotter.py`
- Dependencies: `requirements.txt`
- Sample data: `position_mosaic.dat`
- Example outputs: `output/sic_results.csv`, `output/sic_plot.png`, `output/sic_movie.gif`
- Documentation: `README.md`, `LICENSE`

### Excluded Files (.gitignore)
- Development files: `CLAUDE.md`, `GEMINI.md`, `description.txt`
- Data cache: `data/` directory and all `.tif` files
- Large output directories: `output/daily_maps/`, `output/maps_data/`
- Python artifacts: `venv/`, `__pycache__/`, `*.pyc`