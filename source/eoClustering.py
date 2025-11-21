import xarray as xr
import numpy as np
import faiss
import time





###################################################################################################
# Description: This function conducts clustering for a given image cube (img_cube) in 
#              xarray.dataset format 
#
###################################################################################################
def faiss_cluster_xarray(img_cube: xr.DataArray, n_clusters=100, sample_fraction=0.1, seed=42):
    """
    Perform k-means clustering using FAISS-CPU on an image cube in xarray format.

    Parameters
    ----------
    img_cube : xr.DataArray
        Image cube with dimensions (byand, y, x) or (y, x, band).
    n_clusters : int
        Number of clusters to generate.
    sample_fraction : float
        Fraction of pixels used for training centroids (to speed up clustering).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    cluster_map : xr.DataArray
        Cluster label map with same spatial shape as the input image.
    centroids : np.ndarray
        Array of cluster centroids (n_clusters, n_bands).
    """

    start_time = time.time()
    
    #==============================================================================================
    # Step 1: Rearrange to (n_pixels, n_bands)
    #==============================================================================================
    if 'band' in img_cube.dims:
        img = img_cube.transpose('y', 'x', 'band').values
    else:
        raise ValueError("Input DataArray must have a 'band' dimension.")

    n_y, n_x, n_bands = img.shape
    img_2d = img.reshape(-1, n_bands).astype('float32')

    #==============================================================================================
    # Step 2: Remove invalid pixels (e.g., NaNs)
    #==============================================================================================
    valid_mask = np.all(np.isfinite(img_2d), axis=1)
    valid_data = img_2d[valid_mask]

    #==============================================================================================
    # Step 3: Subsample for training (optional but speeds things up)
    #==============================================================================================
    if sample_fraction < 1.0:
        np.random.seed(seed)
        n_sample = int(len(valid_data) * sample_fraction)
        sample_idx = np.random.choice(len(valid_data), n_sample, replace=False)
        train_data = valid_data[sample_idx]
    else:
        train_data = valid_data
    
    #==============================================================================================
    # Step 4: Train k-means using FAISS
    #==============================================================================================
    print(f"Training FAISS KMeans with {n_clusters} clusters on {train_data.shape[0]} samples...")
    kmeans = faiss.Kmeans(d=n_bands, k=n_clusters, niter=50, verbose=True, seed=seed)
    kmeans.train(train_data)

    centroids = kmeans.centroids
    
    #==============================================================================================
    # Step 5: Assign all valid pixels to nearest centroid
    #==============================================================================================
    index = faiss.IndexFlatL2(n_bands)
    index.add(centroids)
    _, labels = index.search(valid_data, 1)
    labels = labels.flatten()

    #==============================================================================================
    # Step 6: Rebuild full image with invalid pixels
    #==============================================================================================
    full_labels = np.full(img_2d.shape[0], fill_value=-1, dtype=np.int32)
    full_labels[valid_mask] = labels
    cluster_map = full_labels.reshape(n_y, n_x)

    #==============================================================================================
    # Step 7: Convert to xarray.DataArray
    #==============================================================================================
    cluster_da = xr.DataArray(
        cluster_map,
        dims=("y", "x"),
        coords={"y": img_cube.y, "x": img_cube.x},
        name="cluster_labels"
    )

    print(f"Clustering completed in {time.time() - start_time:.2f} seconds.")
    return cluster_da, centroids
