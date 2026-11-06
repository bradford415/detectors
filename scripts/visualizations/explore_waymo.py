# NOTE: if authentication fails run the following in your terminal: gcloud auth application-default login

import glob
from typing import Optional
import warnings
# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action='ignore', category=FutureWarning)

import dask.dataframe as dd
from waymo_open_dataset import v2

# good cloud storage api
import gcsfs

# google cloud storage uri bucket path
dataset_dir = "gs://waymo_open_dataset_v_2_0_1"

def read(tag: str, context: str) -> dd.DataFrame:
  """Creates a Dask DataFrame for the component specified by its tag."""
  #paths = tf.io.gfile.glob(f'{dataset_dir}/{tag}/{context_name}.parquet')
  gcs_path = f"{dataset_dir}/{tag}/{context_name}.parquet"
  print(f"Querying {gcs_path}")
  fs = gcsfs.GCSFileSystem(token="google_default")
  paths = fs.glob(gcs_path)
  print(f"Found the following parquet files:\n{paths}")
  return dd.read_parquet(paths, filesystem=fs)

# name of the driving segment parquet file
context_name = '10017090168044687777_6380_000_6400_000'
cam_lidar_df = read("training/lidar", context=context_name)


# Lazily read camera images and boxes 
context_name = '10023947602400723454_1120_000_1140_000'
cam_image_df = read('training/camera_image', context=context_name)
cam_box_df = read('training/camera_box', context=context_name)

# Combine DataFrame for individual components into a single DataFrame.

# Camera cam_box_df will be grouped, so each row will have a camera image
# and all associated boxes.
image_w_box_df = v2.merge(cam_image_df, cam_box_df, right_group=True)

# Show raw data
image_w_box_df.head()

# Example how to access data fields via v2 object-oriented API
print(f'Available {image_w_box_df.shape[0].compute()} rows:')
for i, (_, r) in enumerate(image_w_box_df.iterrows()):
  # Create component dataclasses for the raw data
  cam_image = v2.CameraImageComponent.from_dict(r)
  cam_box = v2.CameraBoxComponent.from_dict(r)
  print(
      f'context_name: {cam_image.key.segment_context_name}'
      f' ts: {cam_image.key.frame_timestamp_micros}'
      f' camera_name: {cam_image.key.camera_name}'
      f' image size: {len(cam_image.image)} bytes.'
      f' Has {len(cam_box.key.camera_object_id)} camera labels:'
  )

  for j, (object_id, x, y) in enumerate(zip(
      cam_box.key.camera_object_id, cam_box.box.center.x, cam_box.box.center.y
  )):
    print(f'\tid: {object_id},  center: ({x:.1f}, {y:.1f}) px')
    if j > 2:
      print('\t...')
      break
  if i > 2:
    print('...')
    break



