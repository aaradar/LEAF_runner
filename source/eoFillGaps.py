# import numpy as np
# import glob
# from sklearn.cluster import KMeans
# from osgeo import gdal, gdal_array

# # Enable GDAL exceptions
# gdal.UseExceptions()

# # Step 1: Get list of all TIFF files (Adjust path as needed)
# tif_files = sorted(glob.glob("C:/Work_Data/test_clustering/*.tif"))  # Sort to maintain order

# if not tif_files:
#     raise FileNotFoundError("No TIFF files found in the specified directory!")

# # Step 2: Read all TIFF files and stack as a multi-band NumPy array
# bands = []
# for file in tif_files:
#     ds = gdal.Open(file, gdal.GA_ReadOnly)
#     if ds is None:
#         raise RuntimeError(f"Error opening {file}")
    
#     band = ds.GetRasterBand(1).ReadAsArray()  # Read single-band raster
#     bands.append(band)

# # Convert list of 2D arrays into a 3D NumPy array (height, width, bands)
# img_stack = np.dstack(bands)
# height, width, num_bands = img_stack.shape
# print(f"Image dimensions: {height}x{width} with {num_bands} bands")

# # Step 3: Reshape data for clustering (each pixel is a feature vector)
# X = img_stack.reshape((-1, num_bands))  # Shape: (num_pixels, num_bands)

# # Step 4: Perform K-Means Clustering
# n_clusters = 8  # Adjust as needed
# k_means = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=42)
# k_means.fit(X)

# # Step 5: Reshape cluster labels back to image dimensions
# X_cluster = k_means.labels_.reshape(height, width)

# # Step 6: Save the clustered image as a new raster
# output_file = "C:/Work_Data/test_clustering/clustered_image.tif"
# driver = gdal.GetDriverByName("GTiff")
# out_ds = driver.Create(output_file, width, height, 1, gdal.GDT_Byte)

# out_ds.GetRasterBand(1).WriteArray(X_cluster)
# out_ds.FlushCache()
# out_ds = None  # Close file

# print(f"Clustering completed! Saved result to {output_file}")


import numpy as np
import rasterio
import faiss

import eoImage as eoImg


###################################################################################################
# Description: This function loads multiple single-band TIFF images and stacks them into a 
#              multi-band image.
###################################################################################################
def load_band_images(ImagePaths, NoVal = 0):
  """
    Inputs: 
      ImagePaths: list of strings defining the full paths to all band images;
      NoVal: value representing no-data in the images (e.g., 0, -9999, etc.)

    Returns:
      image (H, W, C)
      mask  (H, W) boolean mask of valid pixels
      profile: rasterio profile of the first band (for saving)
  """
  #================================================================================================
  # Load all band images
  #================================================================================================
  bands = []
  profile = None
  for fp in ImagePaths:
    with rasterio.open(fp) as src:
      bands.append(src.read(1).astype(np.float32))
      if profile is None:
        profile = src.profile  # keep metadata if needed later
  
  #================================================================================================
  # Stack band images into a multi-band image
  #================================================================================================
  image = np.stack(bands, axis=-1)  # (H, W, C)
  
  mask  = np.all(image != NoVal, axis=-1)

  return image, mask, profile




###################################################################################################
# Description: This function separate valid pixels in "img" into two groups: reference pixels and
#              gap pixels
#             
###################################################################################################
def separate_valid_pixels(img, valid_mask, gap_mask):
  """
    Inputs:
      img: (H, W, C) multispectral image
      valid_mask: (H, W) boolean mask (True = valid pixel)
      gap_mask: (H, W) boolean mask (True = gap pixel)

    Returns:
      ref_pixels: (N_ref, C) array of reference pixel spectra
      gap_pixels: (N_gap, C) array of gap pixel spectra
      ref_coords: (N_ref, 2) array of (row, col) coordinates of reference pixels
      gap_coords: (N_gap, 2) array of (row, col) coordinates of gap pixels
  """

  # Only consider valid pixels
  combined_mask = valid_mask.copy()

  # Reference pixels: valid & NOT gap
  ref_pixels_mask = combined_mask & (~gap_mask)

  # Gap pixels: valid & gap
  gap_pixels_mask = combined_mask & gap_mask

  # Extract pixel spectra into arrays
  ref_pixels = img[ref_pixels_mask]  # shape (N_ref, C)
  gap_pixels = img[gap_pixels_mask]  # shape (N_gap, C)

  # Optional: coordinates of each pixel in the original image
  ref_coords = np.column_stack(np.where(ref_pixels_mask))  # (row, col)
  gap_coords = np.column_stack(np.where(gap_pixels_mask))

  return ref_pixels, gap_pixels, ref_coords, gap_coords




###################################################################################################
# Description: This function builds FAISS index and coordinate lookup table from "img" and "mask".
#
# Revision history:  2025-Nov-20  Lixin Sun
###################################################################################################
def build_faiss_index(img, mask, metric="cosine", use_ivf=False, nlist=1024):
  """
    Inputs:
      img: H x W x C multispectral image
      mask: H x W boolean (True = valid pixel)
      metric: "cosine", "euclidean"
      use_ivf: bool, whether to use IVF index
      nlist: number of IVF lists (initially created clusters if use_ivf=True)

    Returns:
      index: faiss index (searchable)
      coords: (N_valid, 2) array mapping index -> (row, col)
      valid_pixels: (N_valid, C) array used to build index (float32)
  """

  H, W, C = img.shape
  #================================================================================================
  # Extract and flaten all valid pixels into (N, C), where N and C are number of valid pixels and 
  # number of bands, respectively.
  #================================================================================================
  valid_pixels = img[mask].reshape(-1, C).astype("float32")

  # Store original pixel coordinates
  coords = np.column_stack(np.where(mask))  # shape (N, 2)

  #================================================================================================
  # Build index container depending on "metric" and "use_ivf"
  #================================================================================================
  if metric == "cosine":
    # Normalize vectors -> inner product == cosine similarity
    faiss.normalize_L2(valid_pixels)

    if use_ivf:
      quantizer = faiss.IndexFlatIP(C)
      index     = faiss.IndexIVFFlat(quantizer, C, nlist, faiss.METRIC_INNER_PRODUCT)
      index.train(valid_pixels)
    else:
      index = faiss.IndexFlatIP(C)

  elif metric == "euclidean":
    if use_ivf:
      quantizer = faiss.IndexFlatL2(C)
      index     = faiss.IndexIVFFlat(quantizer, C, nlist, faiss.METRIC_L2)
      index.train(valid_pixels)
    else:
      index = faiss.IndexFlatL2(C)

  else:
    raise ValueError("Unsupported metric: choose 'cosine' or 'euclidean'")
  
  #================================================================================================
  # Add valid pixels, which must be a 2-D NumPy array of float32, to the index container
  #================================================================================================
  index.add(valid_pixels)

  return index, coords








###################################################################################################
# Description: This function search querys  index for efficiently searching valid pixels in the given image.
#
# Revision history:  2025-Nov-20  Lixin Sun
###################################################################################################
def query_targets(target_pixels, index, coords, k=5, metric="cosine"):
  """
  target_pixels: array (M, C)
  returns:
      nn_coords: (M, k, 2)
      scores: similarity or distance values
  """

  Xq = target_pixels.astype("float32")

  # Normalize if cosine similarity
  if metric == "cosine":
    faiss.normalize_L2(Xq)

  # FAISS batch search
  scores, I = index.search(Xq, k)

  # Convert FAISS indices → image coordinates
  nn_coords = coords[I]

  return nn_coords, scores








###################################################################################################
# Description: This function creates an index map for a given image cube using two or three bands. 
#
# Revision history:  2025-Oct-29  Lixin Sun
###################################################################################################
def create_index_map(ImgCube, BandNames, BufferWidth):
  '''
    Args:
      ImgCube(xarray.DataSet): An xarray DataSet containing band images;
      BandNames(List): A list of two or three band names (strings) to be used for index creation;
      BufferWidth(float): A float specifying the spectral buffer width (in surface reflectance);
      
    Returns:
      index_map(xarray.DataSet): An xarray DataSet containing created index map.
  '''
  
  #================================================================================================
  # Make sure the specified index bands are included in the input image cube
  #================================================================================================
  for band_name in BandNames:
    if band_name not in ImgCube.band.values:
      print(f"<create_index_map> The specified band '{band_name}' is not in the input image cube!")
      return None 
    
  #================================================================================================
  # Create an empty index map that has identical spatial dimensions as the input image cube
  #================================================================================================
  index_map = xr.zeros_like(ImgCube.isel(band=0))  # Initialize index map with zeros

  #================================================================================================     
  # Create index map using two or three bands
  #================================================================================================
  if len(BandNames) == 2: 




#############################################################################################################
# Description: This function returns a dictionary that contains statistics (mean, STD and number of pixels)
#              on each cluster.
#
# Revision history:  2025-Feb-26  Lixin Sun  
#############################################################################################################
def cluster_stats(k_means, Data, n_clusters):
  cluster_stats = {}

  for cluster_id in range(n_clusters):
    cluster_pixels = Data[k_means.labels_ == cluster_id]  # Extract pixels in this cluster
    
    if cluster_pixels.size > 0:  # Ensure there are pixels in the cluster
      mean_values = np.mean(cluster_pixels, axis=0)  # Mean per band
      std_values  = np.std(cluster_pixels, axis=0)  # Std deviation per band
    else:
      mean_values = std_values = np.zeros(num_bands)  # Placeholder if empty

    cluster_stats[cluster_id] = {
      "mean": mean_values,
      "std": std_values,
      "num_pixels": cluster_pixels.shape[0]
    }

  # Print results
  for cluster_id, stats in cluster_stats.items():
    print(f"Cluster {cluster_id}:")
    print(f"  Mean per band: {stats['mean']}")
    print(f"  Std deviation per band: {stats['std']}")
    print(f"  Number of pixels: {stats['num_pixels']}\n")



import numpy as np
import glob
from sklearn import cluster
from osgeo import gdal, osr

# Enable GDAL exceptions
gdal.UseExceptions()

# Step 1: Get list of all TIFF files (Adjust path as needed)
tif_files  = sorted(glob.glob("C:/Work_Data/S2_tile55_923_mosaic/*.tif"))  # Ensure proper ordering
spec_files = [f for f in tif_files if 'tile' in f]

if not spec_files:
  raise FileNotFoundError("No TIFF files found in the specified directory!")
else:
  print('\n All spectral file names:\n')
  print(spec_files)

# Step 2: Read all TIFF files and stack them into a multi-band NumPy array
bands = []
ds_ref = gdal.Open(spec_files[0], gdal.GA_ReadOnly)  # Use first file as reference

if ds_ref is None:
  raise RuntimeError(f"Error opening {spec_files[0]}")

geotransform = ds_ref.GetGeoTransform()  # Spatial reference
projection   = ds_ref.GetProjection()  # Projection info
width        = ds_ref.RasterXSize
height       = ds_ref.RasterYSize

for file in spec_files:
  ds = gdal.Open(file, gdal.GA_ReadOnly)
  if ds is None:
    raise RuntimeError(f"Error opening {file}")
    
  band = ds.GetRasterBand(1).ReadAsArray()  # Read single-band raster
  bands.append(band)

# Convert list of 2D arrays into a 3D NumPy array (height, width, bands)
img_stack = np.dstack(bands).astype(np.float32) / 100.0
_, _, num_bands = img_stack.shape
print(f"Image dimensions: {height}x{width} with {num_bands} bands")

# Step 3: Reshape data for clustering (each pixel is a feature vector)
X = img_stack.reshape((-1, num_bands))   # Shape: (num_pixels, num_bands)
sample_size = min(1000000, X.shape[0])  # Limit to 100k samples
X_sample = X[np.random.choice(X.shape[0], sample_size, replace=False)]

# Step 4: Perform K-Means Clustering
n_clusters = 500  # Adjust as needed
k_means = cluster.MiniBatchKMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=42)
k_means.fit(X_sample)

cluster_stats(k_means, X_sample, n_clusters)

# Step 5: Reshape cluster labels back to image dimensions
#X_cluster = k_means.labels_.reshape(height, width)
X_cluster = k_means.predict(X).reshape(height, width)

# Step 6: Save the clustered image as a new raster with original georeferencing
output_file = "C:/Work_Data/test_clustering/clustered_image_MBKM_500.tif"
driver = gdal.GetDriverByName("GTiff")
out_ds = driver.Create(output_file, width, height, 1, gdal.GDT_Int16)

out_ds.SetGeoTransform(geotransform)  # Preserve spatial reference
out_ds.SetProjection(projection)  # Preserve projection info
out_ds.GetRasterBand(1).WriteArray(X_cluster)
out_ds.FlushCache()
out_ds = None  # Close file

print(f"Clustering completed! Saved result to {output_file} with georeferencing.")
