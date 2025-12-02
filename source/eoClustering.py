import xarray as xr
import numpy as np
import faiss
import time

import eoImage as Img



###################################################################################################
# Description: This function conducts k-mean clustering using FAISS on a given image cube 
#              (ImgCube) in either xarray or np.ndarray format.
#
# revision history:  2025-Nov-26  Lixin Sun  initial creation
###################################################################################################
def faiss_cluster(ImgCube, nClusters, nIters=50, MinSamples=1000, GPU=False):
  """
    Inputs
      ImgCube: A given image cube either in xArray or np.ndarray format.
      nClusters(int): Number of clusters to generate.
      nIters(int): Number of iterations for KMeans training.
      MinSamples(int): Minimum number of samples for each cluster.      
      GPU(Boolean): Whether to use GPU for FAISS (if available).

    Returns
      cluster_map (xr.DataArray): Cluster label map with same spatial shape as the input image.
      centroids(np.ndarray): Array of cluster centroids (n_clusters, n_bands).
    """

  start_time = time.time()
  
  #================================================================================================
  # Covert to numpy array format as necessary
  #================================================================================================
  img_type_str = Img.identify_image_type(ImgCube).lower()
  
  # Convert xarray DataSet to DataArray if necessary
  if 'dataset' in img_type_str:
    ImgCube = xr.concat([ImgCube[var] for var in ImgCube.data_vars], dim='band')
  
  # Handle np.ndarray and xarray.DataArray inputs
  if 'numpy' in img_type_str:
    if ImgCube.ndim != 3:
      raise ValueError("NumPy array must be (H, W, C).")
    
    npImg = ImgCube
    
  elif 'xarray' in img_type_str:  
    band_dim_candidates = ['band', 'bands', 'wavelength', 'spectral']
    band_dim = None
    for d in ImgCube.dims:
      if d.lower() in band_dim_candidates:
        band_dim = d
        break
      
      if band_dim is None:
        # If no band dimension, assume last dimension is band
        band_dim = ImgCube.dims[-1]

      other_dims = [d for d in ImgCube.dims if d != band_dim]
      npImg = ImgCube.transpose(*other_dims, band_dim).values
    
  else:
    raise ValueError(f"Unsupported input type: {img_type_str}.")
  
  #================================================================================================
  # Covert to 2D numpy array format
  #================================================================================================
  H, W, C = npImg.shape
  img_2d  = npImg.reshape(-1, C).astype('float32')
   
  #================================================================================================
  # Remove invalid pixels (e.g., NaNs or all bands are zero)
  #================================================================================================
  valid_mask = np.all(np.isfinite(img_2d), axis=1) & np.any(img_2d != 0, axis=1)
  valid_data = img_2d[valid_mask]

  #================================================================================================
  # Subsample for training (optional but speeds things up)
  #================================================================================================
  RANDOM_SEED = 42
  #nSamples    = int(len(valid_data) * sample_fraction)

#   if sample_fraction < 1.0:
#     np.random.seed(RANDOM_SEED)        # calculate the number of samples for training
#     sample_idx = np.random.choice(len(valid_data), nSamples, replace=False)
#     train_data = valid_data[sample_idx]

#   else:
#     train_data = valid_data

  train_data = valid_data 

  #==============================================================================================
  # Train k-means using FAISS
  #==============================================================================================
  print(f"Training FAISS KMeans with {nClusters} clusters on {train_data.shape[0]} samples...")
  kmeans = faiss.Kmeans(d=C, 
                        k=nClusters, 
                        niter=nIters, 
                        verbose=True, 
                        seed=RANDOM_SEED, 
                        gpu=GPU,
                        max_points_per_centroid = MinSamples)  

  kmeans.train(train_data)

  centroids = kmeans.centroids
    
  #==============================================================================================
  # Assign all valid pixels to nearest centroid
  #==============================================================================================
  index = faiss.IndexFlatL2(C)
  index.add(centroids)

  _, labels = index.search(valid_data, 1)
  labels = labels.flatten()

  #==============================================================================================
  # Rebuild full image with invalid pixels
  #==============================================================================================
  full_labels = np.full(img_2d.shape[0], fill_value=-1, dtype=np.int32)
  full_labels[valid_mask] = labels
  cluster_map = full_labels.reshape(H, W)

  print(f"Clustering completed in {time.time() - start_time:.2f} seconds.")

  return cluster_map, centroids



# KeyStrings = ['blue_', 'green_', 'red_', 'edge1_', 'edge2_', 'edge3_', 'nir08_', 'swir16_', 'swir22_']
# KeyStrings_10m = ['blue_', 'green_', 'red_', 'nir08_']

# DataDir = 'C:\\Work_Data\\S2_mosaic_vancouver2020_20m_for_testing_gap_filling'
# taget_img, taget_mask, taget_profile = Img.load_TIF_files_to_npa(DataDir, KeyStrings, 'Aug')

# cluster_map, centroids = faiss_cluster(taget_img, 20, 400, 100000)
# outFile = DataDir + '\\' + 'cluster_map_20m.tif'

# Img.save_npa_as_geotiff(outFile, cluster_map, taget_profile)

# np.set_printoptions(precision=4, suppress=True)
# print(centroids)

# water_clusters = []
# N, C = centroids.shape
# for i in range(N):
#   grn = centroids[i, 1]
#   nir = centroids[i, 3]
#   ndwi = (nir - grn) / (nir + grn + 1e-6)
#   if ndwi > 0.1:
#     water_clusters.append(i)

