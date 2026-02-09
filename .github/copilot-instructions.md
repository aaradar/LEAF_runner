## Purpose
Help Codex/AI agents become productive quickly in this repository by summarizing the runtime entry points, data flows, file-level responsibilities, and common developer workflows.

## Big picture (what this project does)
- This repo produces mosaicked satellite imagery and derived vegetation biophysical maps (LAI, fAPAR, fCOVER, Albedo, etc.).
- Two main high-level flows: (1) Mosaic production (in `source/eoMosaic.py`) and (2) LEAF parameter estimation (in `source/LEAFProduction.py` + `sl2p_nets/` models).

## Entry points & how to run
- Primary runnable script: `Production.py` — it defines `main()` and uses `source` module helpers. Running `python Production.py` will execute the example parameter dictionaries defined in the file. This is the fastest way to exercise the pipeline locally.
- Modules to call directly for smaller pieces: `source/eoMosaic.py` (mosaic routines), `source/LEAFProduction.py` (vegetation product orchestration), `source/SL2P_NetsTools.py` (model application).
- Important runtime flags / env:
  - For HPC/distributed runs set `http_proxy` and `https_proxy` (README gives examples).
  - Earthdata (HLS) requires a `.netrc` with credentials or set NETRC env var.

## Key files and responsibilities (quick map)
- `Production.py` — top-level runner that builds ProdParams/CompParams and dispatches to mosaic or LEAF production.
- `source/eoParams.py` — parameter normalization and CLI vs dict handling (used by `Production.py`).
- `source/eoMosaic.py` — creates mosaics (uses STAC/odc/stackstac/odc_stac patterns).
- `source/LEAFProduction.py` — orchestrates SL2P workflow: builds mosaic, rescales bands, calls SL2P nets.
- `source/eoImage.py` — central constants and metadata (SSR_META_DICT), pixel QA/date names (`pix_QA`, `pix_date`), data-shape expectations.
- `source/SL2P_NetsTools.py` & `source/SL2P_V1.py` — load SL2P pickled models and helper options; `make_DS_options` expects a local `sl2p_nets/` folder with pickled FeatureCollections.
- `sl2p_nets/` — stores model pickles referenced by `SL2P_V1.get_SL2P_filenames()`.
- `requirements.txt` — canonical dependency list (odc_geo, odc_stac, stackstac, rioxarray, rasterio, xarray, dask, etc.).

## Data shapes & naming conventions agents must know
- Most functions accept and return xarray.Dataset or xarray.DataArray objects (not raw numpy arrays). Many functions expect band-ordered datasets, then use `.to_array(dim='band')`.
- Key band / variable names are centralized in `source/eoImage.py` (e.g., `LEAF_BANDS`, `pix_date`, `pix_QA`). Use these constants rather than hard-coding strings.
- SL2P expects input bands: `['cosVZA','cosSZA','cosRAA'] + SsrData['LEAF_BANDS']` (see `SL2P_V1.get_DS_bands`).

## Integration points & external dependencies
- STAC access: uses odc_stac / pystac_client / stackstac patterns (see `source/eoMosaic.py` and README examples). STAC item properties relied on: `view:sun_elevation`, `view:sun_azimuth`, or `sza`, `saa`, `vza`, `vaa` depending on source.
- Model files: `sl2p_nets/*.pkl` are loaded with `pickle` and expected to have GEE-like FeatureCollection structure. Ensure `sl2p_nets` exists and contains the files referenced by `SL2P_V1.get_SL2P_filenames()`.
- Runtime: the README demonstrates running both debug mode (local single-node) and distributed Dask mode. `requirements.txt` lists packages required to run the code locally.

## Project-specific conventions and pitfalls
- Many modules include built-in example parameter dicts near the top (e.g., `Production.py` defines `ProdParams` and `CompParams`). Agents editing run configurations should update these dicts or pass equivalent dicts to `main()`.
- Functions log using simple print markers like `'<function_name>'`. Follow that pattern for quick, searchable traces.
- Image date band is stored as day-of-year integer in `pix_date`; code frequently uses `where(date>0)` to mask invalid pixels.
- Masking conventions differ by sensor: `eoUtils.apply_default_mask` has sensor-specific logic (SCL vs Fmask). Be careful when changing mask logic — tests or visual checks are recommended.

## Small concrete examples for common edits
- To change the default run: edit `ProdParams` in `Production.py` and run `python Production.py`.
- To locate where mosaic vs leaf production is chosen: inspect `Production.py:main()` — it calls `eoPM.which_product()` then either `leaf.LEAF_production()` or `eoMz.MosaicProduction()`.
- To add a new vegetation product option, follow `SL2P_V1.make_VP_options()` pattern (scale factor, outmin/outmax, variable ID).

## Quick checklist for PRs that modify processing logic
- Confirm `sl2p_nets/` still contains the model pickles you need (or update `SL2P_V1.get_SL2P_filenames`).
- Run `python Production.py` locally (uses small built-in example) to smoke-test changes.
- If touching STAC/ODC code, ensure network/env vars (proxy or NETRC) are set for your environment.

## Where to ask for help / further context
- Start by inspecting `README.md` and `Production.py` for runnable examples.
- For model-related questions, inspect `sl2p_nets/` and `source/SL2P_V1.py` / `source/SL2P_NetsTools.py`.

---
If anything here is unclear or you want a different level of detail (example runs, CLI templates, or unit-test suggestions), tell me which area to expand and I will iterate.
