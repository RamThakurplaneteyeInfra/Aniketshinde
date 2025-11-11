# CropEye API Suite

This repository contains a suite of FastAPI-based microservices for agricultural analysis, weather forecasting, and crop monitoring using satellite data from Google Earth Engine, Sentinel-1, Sentinel-2, and external weather APIs.

## Overview

The suite consists of three main APIs:

1. **Admin.py** - SAR Index Mapping and Pest Detection API
2. **events.py** - Comprehensive Agriculture Analysis API
3. **forecast_currentweather.py** - Weather Forecast and Current Weather API

Each API is designed to run independently and provides specific functionalities for crop monitoring, analysis, and weather data retrieval.

## 1. Admin.py - SAR Index Mapping and Pest Detection API

### Description
A FastAPI application for Synthetic Aperture Radar (SAR) index analysis and pest detection using Sentinel-1 data. Provides endpoints for growth analysis, water uptake monitoring, soil moisture assessment, and pest detection using various SAR indices.

### Port
- **Port Number**: 3000
- **Protocol**: HTTP
- **Host**: 0.0.0.0

### Associated Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API information |
| `/plots` | GET | List all available plots |
| `/plots/{plot_name}/info` | GET | Get information about a specific plot |
| `/analyze_Growth` | POST | Analyze plot growth using Sentinel-1 VH or Sentinel-2 NDVI |
| `/wateruptake` | POST | Analyze water uptake using Sentinel-1 VH or Sentinel-2 NDMI |
| `/SoilMoisture` | POST | Analyze soil moisture using Sentinel-1 VV or Sentinel-2 NDWI |
| `/pest-detection` | POST | Perform pest detection analysis |
| `/visualization-params` | GET | Get visualization parameters for indices |
| `/health` | GET | Health check endpoint |
| `/plots/{plot_name}/tiles` | GET | Get tile URLs for a plot |
| `/satellite-updates/{plot_name}` | GET | Check for satellite updates for a plot |
| `/distance` | GET | Calculate distances from factory to plots |
| `/refresh-from-django` | POST | Manually refresh plots from Django API |

### Formulas Used for Calculations

#### SAR Indices
- **VV (Vertical-Vertical)**: Backscatter coefficient in VV polarization
- **VH (Vertical-Horizontal)**: Backscatter coefficient in VH polarization
- **VV_VH_ratio**: Ratio of VV to VH polarizations
  - Formula: `VV_VH_ratio = VV / VH`
- **SWI (Soil Wetness Index)**: 
  - Formula: `SWI = (VV - VH) / (VV + VH)`
- **RVI (Radar Vegetation Index)**:
  - Formula: `RVI = (VH * 4) / (VV + VH)`

#### Classification Thresholds

**VV Classification (Vegetation Health)**:
- Healthy >= -9 dB
- Healthy -10 to -9 dB
- Moderate -11 to -10 dB
- Stress -12 to -11 dB
- Weak -13 to -12 dB
- Weak < -13 dB

**VH Classification**:
- High VH (>= -5 dB)
- Medium VH (-10 to -5 dB)
- Low VH (-15 to -10 dB)
- Very Low VH (< -15 dB)

**VV_VH_ratio Classification**:
- Very High Ratio (>= 6)
- High Ratio (3-6)
- Medium Ratio (1-3)
- Low Ratio (< 1)

**SWI Classification (Soil Moisture)**:
- Water Bodies: 0.5-0.6
- Water Bodies: 0.2-0.5
- Water Bodies: 0-0.2
- Shallow Water: -0.1-0
- Moist Ground: -0.15--0.1
- Moist Ground: -0.25--0.15
- Moist Ground: -0.3--0.25
- Water Stress: -0.4--0.3
- Water Stress: -0.5--0.4
- Dry: -0.6--0.5
- Dry: -0.65--0.6
- Dry: -0.7--0.65
- Dry: -0.75--0.7
- Dry < -0.75

**RVI Classification**:
- Excess: 0.90-1.00
- Excess: 0.80-0.90
- Adequate: 0.70-0.80
- Adequate: 0.60-0.70
- Adequate: 0.50-0.60
- Adequate: 0.40-0.50
- Sufficient Uptake: 0.30-0.40
- Sufficient Uptake: 0.20-0.30
- Less uptake: 0.10-0.20
- Less uptake: 0.00-0.10
- Less uptake: -0.10-0.00
- Less uptake: -0.20--0.10
- Dry: -0.30--0.20
- Dry: -0.40--0.30
- Dry: -0.50--0.40
- Dry: -0.60--0.50
- Dry: -0.70--0.60
- Dry: -0.80--0.70
- Dry: -0.90--0.80
- Dry: -1.00--0.90

#### Pest Detection
- **NDVI (Normalized Difference Vegetation Index)**: `(B8 - B4) / (B8 + B4)`
- **NDWI (Normalized Difference Water Index)**: `(B3 - B8) / (B3 + B8)`
- **Pest Threshold**: 0.3 (NDVI)
- **NIR Threshold**: 0.15
- **NDWI Threshold**: 0.4

#### Soil Moisture (NDWI-based)
- **NDWI Classification**:
  - Less: ≤ -0.4
  - Adequate: -0.4 to 0
  - Excellent: 0 to 0.2
  - Excess: > 0.2

### Protocols
- **HTTP/HTTPS**: Standard web protocol
- **CORS**: Enabled for cross-origin requests
- **Data Source**: Google Earth Engine (Sentinel-1), Django API for plot data

---

## 2. events.py - Comprehensive Agriculture Analysis API

### Description
A comprehensive FastAPI application for agriculture analysis including Brix/Recovery/Sugar Yield calculations, vegetation indices time series, biomass estimation, stress detection, irrigation monitoring, harvest planning, and weather data integration.

### Port
- **Port Number**: 9000
- **Protocol**: HTTP
- **Host**: 0.0.0.0

### Associated Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API information |
| `/plots` | GET | List all available plots |
| `/plots/debug` | GET | Get detailed plot information for debugging |
| `/analyze` | POST | Analyze Brix, Recovery, and Sugar Yield |
| `/plots/agroStats` | GET | Get comprehensive agricultural statistics for all plots |
| `/plots/{plot_name}/indices` | GET | Get vegetation indices time series |
| `/plots/{plot_name}/rvi` | GET | Get RVI time series |
| `/plots/{plot_name}/biomass` | GET | Get biomass statistics |
| `/plots/{plot_name}/stress` | GET | Analyze stress events |
| `/plots/{plot_name}/stress/all` | GET | Analyze stress events for all indices |
| `/plots/{plot_name}/irrigation` | GET | Detect irrigation events |
| `/irrigation/plan` | GET | Plan irrigation events |
| `/irrigation/count` | GET | Count irrigation events |
| `/harvest/timing` | GET | Calculate days to harvest |
| `/harvest/optimal-window` | GET | Get optimal harvest window |
| `/harvest/growth-stages` | GET | Get growth stage information |
| `/harvest/planning` | GET | Get comprehensive harvest planning |
| `/plots/{plot_name}/harvest-analysis` | GET | Get plot-specific harvest analysis |
| `/monthwise-weather-summary` | GET | Get monthly weather summary |
| `/weekly-weather-summary` | GET | Get weekly weather summary |
| `/daily-weather-summary` | GET | Get daily weather summary |
| `/sync/plot` | POST | Sync a single plot from Django |
| `/sync/plots` | POST | Sync multiple plots from Django |
| `/sync/plot/{plot_id}` | DELETE | Delete a plot |
| `/sync/status` | GET | Get sync status |
| `/refresh-from-django` | POST | Refresh plots from Django |
| `/health` | GET | Health check endpoint |

### Formulas Used for Calculations

#### Brix, Recovery, and Sugar Yield
- **GNDVI (Green Normalized Difference Vegetation Index)**: `(B8 - B3) / (B8 + B3)`
- **Brix Calculation**:
  - Formula: `Brix = (B4 * 0.0122621) - (B6 * 0.00706689) + (GNDVI * 21.5251) + 34.784`
  - Then normalized: `Brix = Brix / 2.0`
- **Recovery**: `Recovery = Brix * 0.44`
- **Sugar Yield**: `SugarYield = Brix * Recovery`

#### Vegetation Indices
- **MSAVI (Modified Soil Adjusted Vegetation Index)**: `2 * B8 + 1 - sqrt((2 * B8 + 1)^2 - 8 * (B8 - B4))`
- **NDRE (Normalized Difference Red Edge)**: `(B8 - B5) / (B8 + B5)`
- **NDVI (Normalized Difference Vegetation Index)**: `(B8 - B4) / (B8 + B4)`
- **NDMI (Normalized Difference Moisture Index)**: `(B8 - B11) / (B8 + B11)`
- **NDWI (Normalized Difference Water Index)**: `(B8 - B12) / (B8 + B12)`

#### RVI (Radar Vegetation Index)
- **RVI Calculation**: `(4 * VH) / (VV + VH)`

#### Biomass Estimation
- **Biomass**: `Biomass = RVI * 50` (absolute value)

#### Harvest Timing
- **Growth Days by Variety**:
  - Suru: 365 days
  - Adsali: 548 days
  - Ratoon: 365 days
  - Pre-seasonal: 425 days

#### Soil Parameters
- **Organic Carbon Stock**: Retrieved from ISRIC SoilGrids, scaled by 2.47
- **pH (H2O)**: Retrieved from ISRIC SoilGrids, scaled by 10.0

### Protocols
- **HTTP/HTTPS**: Standard web protocol
- **CORS**: Enabled for cross-origin requests
- **Data Sources**: Google Earth Engine (Sentinel-1, Sentinel-2), ISRIC SoilGrids, Open-Meteo API, Django API for plot data
- **Caching**: TTLCache with 4-hour TTL for weather data

---

## 3. forecast_currentweather.py - Weather Forecast and Current Weather API

### Description
A FastAPI application providing weather forecast and current weather data using Open-Meteo and WeatherAPI services. Includes caching for improved performance.

### Port
- **Port Number**: 8007
- **Protocol**: HTTP
- **Host**: 0.0.0.0

### Associated Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/forecast` | GET | Get 8-day weather forecast |
| `/current-weather` | GET | Get current weather conditions |

### Formulas Used for Calculations
- No complex calculations - primarily API data aggregation
- Temperature, precipitation, wind speed, and humidity values are retrieved directly from external APIs

### Protocols
- **HTTP/HTTPS**: Standard web protocol
- **CORS**: Enabled for cross-origin requests
- **Data Sources**: Open-Meteo API (forecast), WeatherAPI (current weather)
- **Caching**: TTLCache with 2-hour TTL for forecast, 30-minute TTL for current weather
- **Rate Limiting**: 100 requests per minute per IP (commented out in code)

---

## Installation and Setup

### Prerequisites
- Python 3.8+
- Google Earth Engine account and authentication
- API keys for WeatherAPI (for current weather)

### Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up Earth Engine authentication: `earthengine authenticate`
4. Configure API keys in environment variables

### Running the Services
- **Admin.py**: `uvicorn Admin:app --host 0.0.0.0 --port 3000 --reload`
- **events.py**: `uvicorn events:app --host 0.0.0.0 --port 9000 --reload`
- **forecast_currentweather.py**: `uvicorn forecast_currentweather:app --host 0.0.0.0 --port 8007 --reload`

### Environment Variables
- `WEATHERAPI_KEY`: API key for WeatherAPI (current weather service)

## Contributing
Please follow standard Python development practices and ensure all new endpoints include proper documentation and error handling.

## License
[Specify your license here]
