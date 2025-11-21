
'''
ATPRK downscale example (20m -> 10m) for Sentinel-2 using 4 x 10m bands as predictors.
'''

import numpy as np
import rasterio
import matplotlib.pyplot as plt

from rasterio.enums import Resampling
from pykrige.ok import OrdinaryKriging
from scipy.ndimage import uniform_filter
from scipy import stats
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed



###################################################################################################
# Purpose: Load a Sentinel-2 band from disk.
###################################################################################################
def load_band_rasterio(path):
    """Load a single-band raster with rasterio. Returns array (float) and meta.
       Args:
           path (string): file path to raster file"""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        meta = src.meta.copy()

    return arr, meta




###################################################################################################
# Description: Aggregate 10m array to 20m by averaging each 2x2 block. Assumes b10 shape is 
#              divisible by 2. Returns aggregated array of shape (h/2, w/2).
###################################################################################################
def aggregate_10_to_20_mean(b10):
    """
    Aggregate 10m array to 20m by averaging each 2x2 block. Assumes b10 shape is divisible by 2.
    Returns aggregated array of shape (h/2, w/2).
    """
    # if shapes are odd, trim last row/col
    h, w = b10.shape
    h2 = (h // 2) * 2
    w2 = (w // 2) * 2
    b10_trim = b10[:h2, :w2]

    # reshape trick to block-average 2x2
    b10_blocks = b10_trim.reshape(h2//2, 2, w2//2, 2)
    agg = b10_blocks.mean(axis=(1, 3))
    
    return agg



###################################################################################################
# Description: Fit multiple linear regression at coarse scale. This function returns coefficients
#              (including intercept) and predicted coarse.
###################################################################################################
def fit_regression_on_coarse_slow(CoarseImg, AgregatedImgs):
    """
      Args:
        CoarseImg: A band image (flattened array) in coarse resolution, e.g., 20-m SWIR1 or 2
        AgregatedImgs: A list of aggregated images (e.g., B2,B3,B4,B8 for S2) with the same resolution as CoarseImg.
    """
    n = CoarseImg.size

    X = np.column_stack([p.ravel() for p in AgregatedImgs] + [np.ones(n)])
    y = CoarseImg.ravel()

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = (X @ coef).reshape(CoarseImg.shape)

    return coef, y_hat




###################################################################################################
# Description: Fit multiple linear regression at coarse scale via normal equations (no full X). 
#              This function returns coefficients (including intercept) and predicted coarse.
###################################################################################################
def fit_regression_on_coarse_fast(CoarseImg, AgregatedImgs):
    """
      Args:
        CoarseImg: A band image (flattened array) in coarse resolution, e.g., 20-m SWIR1 or 2
        AgregatedImgs: A list of aggregated images (e.g., B2,B3,B4,B8 for S2) with the same resolution as CoarseImg.
    """
    y = CoarseImg.astype(np.float64).ravel()
    P = [p.astype(np.float64).ravel() for p in AgregatedImgs]
    P.append(np.ones_like(y))  # intercept
    k = len(P)

    XT_X = np.zeros((k, k), dtype=np.float64)
    XT_y = np.zeros((k,), dtype=np.float64)

    # compute upper triangle and mirror (dot products use BLAS)
    for i in range(k):
        Xi = P[i]
        XT_y[i] = Xi.dot(y)
        for j in range(i, k):
            XT_X[i, j] = Xi.dot(P[j])
            XT_X[j, i] = XT_X[i, j]

    coef = np.linalg.solve(XT_X, XT_y)
    # predicted
    y_hat = np.zeros_like(y, dtype=np.float64)
    for c, p in zip(coef, P):
        y_hat += c * p
    y_hat = y_hat.reshape(CoarseImg.shape)
    return coef, y_hat





###################################################################################################
# Description: Apply regression coefficients at fine scale 10-m resolution images [B2,B3,B4,B8].
#              This prediction is the first step in ATPRK, before kriging residuals are added.
#
###################################################################################################
def predict_fine_from_regression_slow(coef, predictors10):
    """
      Args:
        coef: regression coefficients (including intercept);
        predictors10: A list of 10-m resolution images/arrays [B2,B3,B4,B8].
    """
    n = predictors10[0].size   # number of pixels in predictors10
    Xf = np.column_stack([p.ravel() for p in predictors10] + [np.ones(n)])

    # Produce the 10 m regression prediction of the coarse band.
    y_fine = (Xf @ coef).reshape(predictors10[0].shape)

    return y_fine




###################################################################################################
# Description: Apply regression coefficients at fine scale 10-m resolution images [B2,B3,B4,B8].
#              This prediction is the first step in ATPRK, before kriging residuals are added.
#
###################################################################################################
def predict_fine_from_regression_fast(coef, predictors10):
    """
      Args:
        coef: regression coefficients (including intercept);
        predictors10: A list of 10-m resolution images/arrays [B2,B3,B4,B8].
    """
    # predictors10: list of 2D arrays (float)
    pred = np.zeros_like(predictors10[0], dtype=np.float64)
    for i in range(len(predictors10)):
        pred += coef[i] * predictors10[i].astype(np.float64)
    pred += coef[-1]
    return pred




###################################################################################################
# Compute centroid coordinates for raster (meters assumed)
# meta must include 'transform' (affine tuple (a,b,c,d,e,f)) and width/height
# transform[2] = top-left x, transform[5] = top-left y; transform[0] pixel width, transform[4] pixel height (often negative)
###################################################################################################
def compute_centroid_coords(meta):
    """
    Compute centroid coordinates (x,y) for each pixel in a raster given rasterio meta.
    Returns arrays of shape (h,w) for x and y.
    """
    transform = meta['transform']
    width  = meta['width']
    height = meta['height']
    cols   = np.arange(width)
    rows   = np.arange(height)
    
    xx = transform[2] + cols * transform[0] + transform[0] / 2.0
    yy = transform[5] + rows * transform[4] + transform[4] / 2.0

    xgrid, ygrid = np.meshgrid(xx, yy)
    return xgrid, ygrid




###################################################################################################
# Helper: get coarse points within a radius of a window bounding box
###################################################################################################
def _select_coarse_points_in_radius(x20v, y20v, resv, win_x_min, win_x_max, win_y_min, win_y_max, radius, max_points):
    """
    Select coarse points whose coordinates are within (window bbox expanded by radius).
    Returns arrays xk, yk, rk (residuals), optionally subsampled to max_points.
    """
    # Expand bbox by radius (meters)
    minx = win_x_min - radius
    maxx = win_x_max + radius
    miny = win_y_min - radius
    maxy = win_y_max + radius

    mask = (x20v >= minx) & (x20v <= maxx) & (y20v >= miny) & (y20v <= maxy)
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return np.array([]), np.array([]), np.array([])

    if idx.size > max_points:
        # uniform random sample to limit points
        sel = np.random.choice(idx, size=max_points, replace=False)
    else:
        sel = idx

    return x20v[sel], y20v[sel], resv[sel]



###################################################################################################
# Worker: krige a single window (fine window indices)
# - window is defined in fine-pixel coordinates (row_start,row_stop, col_start,col_stop)
# - predicts residuals for fine pixels in this window only
###################################################################################################
def _worker_krige_window(args):
    """
    args: tuple containing all necessary data (to keep pickling simple)
    returns: (row_start, row_stop, col_start, col_stop, z_window) where z_window matches window shape
    """
    (row_start, row_stop, col_start, col_stop,
     x20v, y20v, resv, x10, y10, fine_shape,
     radius, max_coarse_points, min_coarse_points,
     variogram_model) = args

    # bounding box of window in coordinates
    # compute coordinates for fine pixels in this window and their flat indices
    rows = np.arange(row_start, row_stop)
    cols = np.arange(col_start, col_stop)
    # x10 and y10 are full grids (2D). Extract subgrids
    x_window = x10[np.ix_(rows, cols)]
    y_window = y10[np.ix_(rows, cols)]
    flat_xw = x_window.ravel()
    flat_yw = y_window.ravel()

    win_x_min, win_x_max = flat_xw.min(), flat_xw.max()
    win_y_min, win_y_max = flat_yw.min(), flat_yw.max()

    # select coarse points within radius expanded bbox
    xk, yk, rk = _select_coarse_points_in_radius(x20v, y20v, resv,
                                                 win_x_min, win_x_max, win_y_min, win_y_max,
                                                 radius, max_coarse_points)

    # If not enough points, return zeros (no kriging correction)
    if xk.size < min_coarse_points:
        z_window = np.zeros_like(flat_xw, dtype=np.float64)
        return (row_start, row_stop, col_start, col_stop, z_window.reshape(x_window.shape))

    # Optionally, we could also further restrict xk,yk to nearest N by centroid distance
    # Build OrdinaryKriging and predict
    try:
        OK = OrdinaryKriging(xk, yk, rk, variogram_model=variogram_model, verbose=False, enable_plotting=False)
        z_pred, ss = OK.execute('points', flat_xw, flat_yw)
        z_window = z_pred.reshape(x_window.shape)
    except Exception as e:
        # If kriging fails for numerical reasons, return zeros
        print(f"Local kriging failed for window {(row_start, col_start)}: {e}")
        z_window = np.zeros_like(flat_xw, dtype=np.float64)
        z_window = z_window.reshape(x_window.shape)

    return (row_start, row_stop, col_start, col_stop, z_window)



###################################################################################################
# Main optimized ATPRK function (windowed kriging, parallel)
###################################################################################################
def atprk_downscale_optimized(coarse20, coarse_meta, predictors10, predictors10_meta,
                              window_size_fine=2048, overlap=128, radius_m=3000,
                              max_coarse_points=2000, min_coarse_points=30,
                              n_workers=4, variogram_model='spherical',
                              sample_for_kriging=False, sample_size=2000):
    """
    Optimized ATPRK pipeline for very large tiles.
    Returns: (atprk_10, reg_pred_fine, kriged_residuals)
    - coarse20: 2D array (20m)
    - coarse_meta: rasterio-style meta (transform, height, width)
    - predictors10: list of 2D arrays (10m) e.g., [B2,B3,B4,B8], same shape as predictors10_meta
    - predictors10_meta: rasterio-style meta for 10m
    - window_size_fine: window size in fine pixels (10m)
    - overlap: overlap in pixels between windows
    - radius_m: search radius (meters) to collect coarse points for local kriging
    """
    # 0) types and basic checks
    predictors10 = [p.astype(np.float32) for p in predictors10]
    coarse20     = coarse20.astype(np.float32)

    # 1) Aggregate predictors to 20m
    predictors20 = [aggregate_10_to_20_mean(p) for p in predictors10]

    # 2) Ensure shapes align (resample (not implemented here) if necessary)
    if predictors20[0].shape != coarse20.shape:
        raise ValueError("Aggregated predictors shape does not match coarse shape — resampling required.")

    # 3) Fit regression (fast)
    coef, reg_pred_coarse = fit_regression_on_coarse_fast(coarse20, predictors20)
    print("Regression coefficients:", coef)

    # 4) Predict at fine scale
    reg_pred_fine = predict_fine_from_regression_fast(coef, predictors10)

    # 5) residuals on coarse grid -> flatten coords for kriging sources
    residuals_coarse = coarse20.astype(np.float64) - reg_pred_coarse.astype(np.float64)
    x20, y20 = compute_centroid_coords(coarse_meta)
    x20v = x20.ravel()
    y20v = y20.ravel()
    resv = residuals_coarse.ravel()

    # optionally subsample coarse residuals for kriging (global sample)
    if sample_for_kriging:
        if resv.size > sample_size:
            idx = np.random.choice(resv.size, size=sample_size, replace=False)
            x20v = x20v[idx]; y20v = y20v[idx]; resv = resv[idx]

    # 6) Prepare fine coords grid
    x10, y10 = compute_centroid_coords(predictors10_meta)
    fine_h, fine_w = predictors10[0].shape

    # 7) Build window list (row_start, row_stop, col_start, col_stop)
    rows = list(range(0, fine_h, window_size_fine - overlap))
    cols = list(range(0, fine_w, window_size_fine - overlap))

    windows = []
    for r0 in rows:
        r1 = min(r0 + window_size_fine, fine_h)
        for c0 in cols:
            c1 = min(c0 + window_size_fine, fine_w)
            windows.append((r0, r1, c0, c1))

    print(f"Total windows: {len(windows)} (window_size={window_size_fine}, overlap={overlap})")

    # 8) Launch parallel kriging for each window
    tasks = []
    for (r0,r1,c0,c1) in windows:
        args = (r0, r1, c0, c1,
                x20v, y20v, resv,
                x10, y10, (fine_h, fine_w),
                radius_m, max_coarse_points, min_coarse_points, variogram_model)
        tasks.append(args)

    kriged_residuals = np.zeros((fine_h, fine_w), dtype=np.float64)

    # Use process pool
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_worker_krige_window, task): task for task in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Local kriging windows"):
            try:
                r0, r1, c0, c1, z_win = fut.result()
                kriged_residuals[r0:r1, c0:c1] += z_win  # windows overlap -> sum contributions
            except Exception as e:
                task = futures[fut]
                print(f"Window {task[:4]} failed: {e}")

    # Optional: average overlapping areas
    # Build overlap-count grid to divide summed overlaps
    count_grid = np.zeros_like(kriged_residuals)
    for (r0,r1,c0,c1) in windows:
        count_grid[r0:r1, c0:c1] += 1.0
    # Avoid division by zero
    mask = count_grid > 0
    kriged_residuals[mask] /= count_grid[mask]

    # 9) Final ATPRK result
    atprk_10 = reg_pred_fine + kriged_residuals

    return atprk_10.astype(np.float32), reg_pred_fine.astype(np.float32), kriged_residuals.astype(np.float32), coef



def wald_validation(original10, downscale_func, coarse_band20, coarse_meta, predictors10, predictors10_meta):
    """
    Wald-style validation: simulate by downsampling the 10m 'original10' to 20m,
    apply the downscaling pipeline to the simulated coarse, and compare result with original10.
    Here, original10 is a true 10m band (e.g., B04) used to test algorithm.)
    """
    # create simulated coarse by block-averaging original10
    sim_coarse = aggregate_10_to_20_mean(original10)
    # create meta for simulated coarse by copying coarse_meta? For simplicity we will re-use coarse_meta
    # call atprk on sim_coarse with provided predictors (use provided predictors10)
    atprk_sim, reg_sim, res_sim, coef = downscale_func(sim_coarse, coarse_meta, predictors10, predictors10_meta)
    # compare atprk_sim with original10 (trim to matched shapes)
    # NOTE: our aggregation trimmed odd dims; ensure shapes align
    h, w = atprk_sim.shape
    original10_crop = original10[:h, :w]
    # metrics
    rmse = np.sqrt(np.mean((original10_crop - atprk_sim)**2))
    r, _ = stats.pearsonr(original10_crop.ravel(), atprk_sim.ravel())
    print(f"Wald test RMSE: {rmse:.4f}, Pearson r: {r:.4f}")
    return rmse, r



###################################################################################################
# USER: set your file paths
###################################################################################################
B02_10m_fp = "B02_10m.tif"
B03_10m_fp = "B03_10m.tif"
B04_10m_fp = "B04_10m.tif"
B08_10m_fp = "B08_10m.tif"
COARSE_20m_fp = "B11_20m.tif"      # the band to downscale (20m)
OUT_10m_fp = "B11_downscaled_10m.tif"



# -------------------------------------------------------
# MAIN runnable block
# -------------------------------------------------------
if __name__ == "__main__":
    # 1. Load inputs
    b02_10, meta10   = load_band_rasterio(B02_10m_fp)
    b03_10, _        = load_band_rasterio(B03_10m_fp)
    b04_10, _        = load_band_rasterio(B04_10m_fp)
    b08_10, _        = load_band_rasterio(B08_10m_fp)
    coarse20, meta20 = load_band_rasterio(COARSE_20m_fp)

    predictors10 = [b02_10, b03_10, b04_10, b08_10]

    # 2. Run OPTIMIZED ATPRK
    atprk_10, reg10, resid10, coef = atprk_downscale_optimized(
        coarse20,                 # coarse_res_image
        meta20,                   # coarse_meta
        predictors10,             # list of predictors at 10m
        meta10,                   # predictors metadata
        window_size_fine=2048,
        overlap=128,
        radius_m=3000,
        max_coarse_points=2000,
        min_coarse_points=30,
        n_workers=4,
        variogram_model='spherical',
        sample_for_kriging=False,
        sample_size=2000
    )

    print("Optimized ATPRK done.")

    # 3. Save result as GeoTIFF
    out_meta = meta10.copy()
    out_meta.update({"dtype": "float32", "count": 1})

    with rasterio.open(OUT_10m_fp, "w", **out_meta) as dst:
        dst.write(atprk_10.astype(np.float32), 1)

    print("Saved downscaled band to:", OUT_10m_fp)

    # 4. Optional Wald validation
    try:
        rmse, r = wald_validation(b04_10, atprk_downscale_optimized,
                                  coarse20, meta20, predictors10, meta10)
        print("Wald validation RMSE:", rmse, "r:", r)
    except Exception as e:
        print("Wald validation failed:", e)

    # 5. Quick visualization
    plt.figure(figsize=(12, 6))
    vmin, vmax = np.percentile(atprk_10, (2, 98))

    plt.subplot(1, 3, 1)
    plt.title("Regression pred (10m)")
    plt.imshow(reg10, vmin=vmin, vmax=vmax)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Kriged residuals (10m)")
    plt.imshow(resid10,
               vmin=np.percentile(resid10, 2),
               vmax=np.percentile(resid10, 98))
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Final Optimized ATPRK 10m")
    plt.imshow(atprk_10, vmin=vmin, vmax=vmax)
    plt.axis("off")

    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    # 1. Load inputs
    b02_10, meta10   = load_band_rasterio(B02_10m_fp)
    b03_10, _        = load_band_rasterio(B03_10m_fp)
    b04_10, _        = load_band_rasterio(B04_10m_fp)
    b08_10, _        = load_band_rasterio(B08_10m_fp)
    coarse20, meta20 = load_band_rasterio(COARSE_20m_fp)

    predictors10 = [b02_10, b03_10, b04_10, b08_10]

    # 2. Run ATPRK (note: may take minutes depending on tile size)
    atprk_10, reg10, resid10, coef = atprk_downscale(coarse20, meta20, predictors10, meta10,
                                                    sample_for_kriging=True, max_krige_points=2500)
    print("ATPRK done.")

    # 3. Save result as GeoTIFF (10m grid metadata)
    out_meta = meta10.copy()
    out_meta.update({"dtype": "float32", "count": 1})
    with rasterio.open(OUT_10m_fp, "w", **out_meta) as dst:
        dst.write(atprk_10.astype(np.float32), 1)
    print("Saved downscaled band to:", OUT_10m_fp)

    # 4. (Optional) Wald validation if you have a reference 10m band to test
    # Example: use B04 as 'original' and try to downscale its 20m version to 10m
    # Here we use b04_10 as truth
    try:
        rmse, r = wald_validation(b04_10, atprk_downscale, coarse20, meta20, predictors10, meta10)
    except Exception as e:
        print("Wald validation failed (maybe shapes mismatch). Error:", e)

    # 5. Quick plots
    plt.figure(figsize=(12, 6))
    vmin, vmax = np.percentile(atprk_10, (2, 98))
    plt.subplot(1, 3, 1); plt.title("Regression pred (10m)"); plt.imshow(reg10, vmin=vmin, vmax=vmax); plt.axis('off')
    plt.subplot(1, 3, 2); plt.title("Kriged residuals (10m)"); plt.imshow(resid10, vmin=np.percentile(resid10, 2), vmax=np.percentile(resid10, 98)); plt.axis('off')
    plt.subplot(1, 3, 3); plt.title("Final ATPRK 10m"); plt.imshow(atprk_10, vmin=vmin, vmax=vmax); plt.axis('off')
    plt.tight_layout()
    plt.show()
