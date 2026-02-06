# Groundwater Vulnerability Mapping Under Climate Uncertainty (Code Sample)
# Applicant: Iman Mahato
# PhD Applicant | IISER Berhampur

# Code Sample Overview
This repository contains a part of the computational framework used to preprocess and downscale GRACE groundwater anomalies from 1° to 0.1° resolution.

# Folder Structure

1. "GRACE_JPL_GLDAS_NOAH_PPS" (Jupyter Notebook)
    Function: Handles the ingestion of raw NetCDF data (GLDAS, GRACE), performs unit standardization (kg/m² to cm), and creates the spatiotemporal tensor for the model.
    Tools: 'xarray', 'pandas', 'geopandas', 'rasterio'

2. "GTNNWR_Model" (Python Modules)
    My Contribution: Adaptation & Configuration.
    Source: Core architecture adapted from the original GTNNWR implementation (Liang et al. 2025; Wu et al. 2020).
    Key Modifications Till Now:
        'setting_data_gtnnwr_simulation_st.cfg': Customized the input feature vector to include hydrological drivers (Precipitation, Evapotranspiration, Soil Moisture, NDVI etc) specific to the study area. Calibrated spatiotemporal bandwidths and initial test runs. 

