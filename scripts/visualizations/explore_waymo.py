# NOTE: if authentication fails run the following in your terminal: gcloud auth application-default login

import glob
from typing import Optional
import warnings
# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action='ignore', category=FutureWarning)

import dask.dataframe as dd
#from waymo_open_dataset import v2

# good cloud storage api
import gcsfs

# google cloud storage uri bucket path
dataset_dir = "gs://waymo_open_dataset_v_2_0_1"

# name of the driving segment parquet file
context_name = '10017090168044687777_6380_000_6400_000'

def read(tag: str) -> dd.DataFrame:
  """Creates a Dask DataFrame for the component specified by its tag."""
  #paths = tf.io.gfile.glob(f'{dataset_dir}/{tag}/{context_name}.parquet')
  gcs_path = f"{dataset_dir}/{tag}/{context_name}.parquet"
  print(f"Querying {gcs_path}")
  fs = gcsfs.GCSFileSystem(token="google_default")
  paths = fs.glob(gcs_path)
  print(f"Found the following parquet files:\n{paths}")
  return dd.read_parquet(paths)


read("training/lidar")

