image compositing code (s2 gits)
sl2p repos by djamainajib i assume you already know about / worked in
s2gvm gits:

https://github.com/ollinevalainen/satellitetools
https://github.com/DPIRD-DMA/S2Mosaic
https://github.com/senbox-org/sen2like
https://github.com/nasa/HLS-Data-Resources
https://github.com/Ha-eunYu/Sentinel-2_optical_pipeline

 SL2P / Biophysical Variable Retrieval
1. djamainajib/SL2P-PYTHON
The closest sibling to LEAF_runner. A Python implementation of the SL2P processor — the same algorithm used in the LEAF-Toolbox — that estimates LAI, FCOVER, FAPAR, CCC, and CWC from Sentinel-2 L2A images at 10 or 20 m resolution. GitHub Essentially the core algorithm that LEAF_runner wraps.
2. djamainajib/SL2P_DASF
A modified SL2P processor that incorporates the Directional Area Scattering Function (DASF) for constraining LAI/FCOVER/FAPAR/CCC/CWC estimates, and generates uncertainty outputs and quality flags for input/output data.
3. djamainajib/ALR_SL2P
An Active Learning Regularization approach that regularizes biophysical variables (LAI, FCOVER, FAPAR, CCC, CWC) derived from SL2P, using vegetation indices and LASSO-based feature selection to fill in flagged/invalid SL2P estimates.
4. ollinevalainen/satellitetools
Retrieves Sentinel-2 data from Google Earth Engine or AWS cloud-optimized GeoTIFFs, then computes biophysical parameters (LAI, FAPAR, FCOVER) using a Python implementation of ESA's SNAP Biophysical processor.
 
 
 
 S2/HLS Image Compositing & Mosaicking

5. DPIRD-DMA/S2Mosaic
A Python package for creating cloud-free mosaics from Sentinel-2 imagery, with flexible scene selection methods (by valid data %, oldest, or newest), multiple compositing methods (mean, percentile, median, first valid pixel), and cloud masking via OmniCloudMask. Uses the Planetary Computer STAC API. 
6. senbox-org/sen2like
Generates Sentinel-2-like harmonized surface reflectances by fusing Sentinel-2 and Landsat-8/9 data, producing Harmonized (Level 2H, 30 m) and Fused (Level 2F, 10–20 m) ARD products — boosting revisit frequency to ~95 products/year vs. 73 for S2 alone. 
7. nasa/HLS-Data-Resources
NASA's official guide and tutorial repository for accessing and working with Harmonized Landsat Sentinel-2 (HLS) data, including notebooks using ODC and CMR-STAC to create EVI time series. GitHub Directly relevant since LEAF_runner supports HLS_SR as a sensor.
8. Ha-eunYu/Sentinel-2_optical_pipeline
A Python pipeline that generates RGB composites and spectral indices (NDVI, NDWI, MNDWI) from Sentinel-2 L2A imagery using the Copernicus Data Space STAC API and S3 access, automating cloud filtering, resampling, and GeoTIFF export.
relevent label location



E:\S2_mosaics_2025
# Leaf Stac testing before Runner



E:\S2_mosaics_2026
# Buffer tests, Negative buffer tests, processing logs, afforestation runs testing LeafWrapper



E:\S2_mosaics_runner_2026
# Many files to look at here, not many are useful, as they could have incorrect output or be empty
    E:\S2_mosaics_runner_2026\benchmark
    E:\S2_mosaics_runner_2026\benchmark\S2_pipeline3_august_32w_16gb_10m
    E:\S2_mosaics_runner_2026\benchmark\S2_pipelineregion1_10w_50b_10m
    E:\S2_mosaics_runner_2026\old\afforestation #32/16
    E:\S2_mosaics_runner_2026\old\afforestation\S2_may_2020_1000_regions_10m
    E:\S2_mosaics_runner_2026\sample outputs\S2_regions 1000_10m # 32/16 regions mode
    E:\S2_mosaics_runner_2026\csv\S2_regions1000_10m # 10/50 regions mode
    # Benchmarking comparing 32W/16G to 10W/50G and 1000 regions to 1000 tiles (10/50 was better)

E:\S2_mosaics_runner_2026\csv\S2_regions_10m
    # Contains other files that test clipping with pipelines mostly
E:\S2_mosaics_runner_2026\csv
    # testing the Scrapped delayed computing with 1000 regions in tiles mode
E:\S2_mosaics_runner_2026\benchmark\tiles
    # Testing tiles with S2 and HLS (tiles are not clipped back into regions yet)
E:\S2_mosaics_runner_2026\regions_cutter2_10m
    # Testing clipping mask on regions geotiffs
E:\S2_mosaics_runner_2026\all_regions_tiled_10m
    # All regions tiled with new cookie cutter function for one month in august 2025
E:\S2_mosaics_runner_2026\csv\S2_regions_10m
    # First csv tests, as well inside file location the clipped pipelines and afforestation tests
E:\S2_mosaics_runner_2026\Coldwater_10m.zip\Coldwater_10m
    # Coldwater for LAI, fcover, mosaic
E:\S2_mosaics_runner_2026\old
    # Older tests with coldwater and tmx pipelines
E:\S2_mosaics_runner_2026\S2_regions_fcov_30m
    # Testing with hls, s2, lai, fcov, with different resolutions
Unknown:
E:\S2_mosaics_runner_2026\S2_regions_fcov_30m # Might be 10 regions with FCOVER




E:\User_Stories
    E:\User_Stories\100regions_june_july_may_10m
        # Around 27 of the regions were ran
    E:\User_Stories\all pipelines 2025_10m
        # Pipeline



E:\Testing1
    # Testing while HLS wasn't working with LEAFProduction
    # List dates for a region not working in tiles mode (fixed)
    # Fixing vegparams
    # Some of the user requirement tests



E:\HLS_mosaics_runner_2026\benchmark\tiles 
#- containing first HLS tests on LEAF Runner
    E:\HLS_mosaics_runner_2026\benchmark\tiles\S2_hls_1000_regions_30m 
    # For 1000 regions in regions mode
    E:\HLS_mosaics_runner_2026\benchmark\tiles\S2_hls_1000_tiles_30m
    # For 1000 regions in tiles mode



E:\TestingLEAF
# Files testing with multiple dates for tiles mode, until it worked: final result was dates from 2017, 2018, 2019 that worked with LAI and FCOVER
# Vegetation params exportation testing tool
    # E:\TestingLEAF\sakurajima_9y3_30m 
    # Contains the final result of 9 years in a row of one month of august (2017-2025)
    # E:\TestingLEAF\test_csv_30m 
    # Showing that csv with LEAFProduction works, but vegparams needs to be fixed



E:\TestingLEAFFinal
# The final testing with LEAFProduction with one csv file for all bands
    # E:\TestingLEAFFinal\outputs # Contains both the csv and geotiff outputs in one location - both ran succesfully
# Fcov is written onto the same csv file as the 12 bands