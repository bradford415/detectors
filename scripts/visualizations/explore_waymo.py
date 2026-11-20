# NOTE: if authentication fails run the following in your terminal: gcloud auth application-default login

import glob
from typing import Optional
import warnings

# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action="ignore", category=FutureWarning)

import dask.dataframe as dd
from waymo_open_dataset import v2

# good cloud storage api
import gcsfs

import matplotlib.pyplot as plt
import numpy as np

from waymo_open_dataset.v2.perception.utils import lidar_utils
from waymo_open_dataset.v2.perception import lidar as lidar_module

#mode = "overview"
mode = "viz-lidar"

# google cloud storage uri bucket path
dataset_dir = "gs://waymo_open_dataset_v_2_0_1"


def read(tag: str, context: str) -> dd.DataFrame:
    """Creates a Dask DataFrame for the component specified by its tag."""
    # paths = tf.io.gfile.glob(f'{dataset_dir}/{tag}/{context_name}.parquet')
    gcs_path = f"{dataset_dir}/{tag}/{context}.parquet"
    print(f"Querying {gcs_path}")
    fs = gcsfs.GCSFileSystem(token="google_default")
    paths = fs.glob(gcs_path)
    print(f"Found the following parquet files:\n{paths}")
    return dd.read_parquet(paths, filesystem=fs)



if mode == "overview":
    # name of the driving segment parquet file
    lidar_context_name = "10017090168044687777_6380_000_6400_000"
    cam_lidar_df = read("training/lidar", context=lidar_context_name)


    # Lazily read camera images and boxes
    cam_context_name = "10023947602400723454_1120_000_1140_000"
    cam_image_df = read("training/camera_image", context=cam_context_name)
    cam_box_df = read("training/camera_box", context=cam_context_name)

    # Combine DataFrame for individual components into a single DataFrame.

    # Camera cam_box_df will be grouped, so each row will have a camera image
    # and all associated boxes.
    image_w_box_df = v2.merge(cam_image_df, cam_box_df, right_group=True)

    # Show raw data
    image_w_box_df.head()

    # Example how to access data fields via v2 object-oriented API
    print(f"Available {image_w_box_df.shape[0].compute()} rows:")
    for i, (_, r) in enumerate(image_w_box_df.iterrows()):
        # Create component dataclasses for the raw data
        cam_image = v2.CameraImageComponent.from_dict(r)
        cam_box = v2.CameraBoxComponent.from_dict(r)
        print(
            f"context_name: {cam_image.key.segment_context_name}"
            f" ts: {cam_image.key.frame_timestamp_micros}"
            f" camera_name: {cam_image.key.camera_name}"
            f" image size: {len(cam_image.image)} bytes."
            f" Has {len(cam_box.key.camera_object_id)} camera labels:"
        )

        for j, (object_id, x, y) in enumerate(
            zip(cam_box.key.camera_object_id, cam_box.box.center.x, cam_box.box.center.y)
        ):
            print(f"\tid: {object_id},  center: ({x:.1f}, {y:.1f}) px")
            if j > 2:
                print("\t...")
                break
        if i > 2:
            print("...")
            break

    # @title Per-object trajectories

    # Lazily read lidar boxes
    lidar_box_df = read("training/lidar_box", context=lidar_context_name)
    # Group all available boxes per object
    lidar_box_df = (
        lidar_box_df.groupby(["key.segment_context_name", "key.laser_object_id"])
        .agg(list)
        .reset_index()
    )


    # Read a single row, which contains data for all available frames.
    _, row = next(iter(lidar_box_df.iterrows()))

    # Create component object
    lidar_box = v2.LiDARBoxComponent.from_dict(row)
    print(
        f"Object {lidar_box.key.laser_object_id=} seen on"
        f" {len(lidar_box.key.frame_timestamp_micros)} frames"
    )
    print("Its trajectory across frames:")
    print(f"\t{lidar_box.box.center.x=}")
    print(f"\t{lidar_box.box.center.y=}")
    print(f"\t{lidar_box.box.center.z=}")


    ##### object level
    ##### at this level each row of a table represents a single object at a single frame.
    # @title Camera image, lidar, boxes with keypoints

    # Lazily read DataFrames for all components.
    association_df = read(
        "training/camera_to_lidar_box_association", context=cam_context_name
    )
    cam_hkp_df = read("training/camera_hkp", context=cam_context_name)
    cam_box_df = read("training/camera_box", context=cam_context_name)
    cam_img_df = read("training/camera_image", context=cam_context_name)
    lidar_box_df = read("training/lidar_box", context=cam_context_name)
    lidar_df = read("training/lidar", context=cam_context_name)

    # Join all DataFrame objects for all components together.
    cam_image_w_box_df = v2.merge(cam_box_df, cam_img_df)
    cam_image_w_box_w_hkp_df = v2.merge(cam_image_w_box_df, cam_hkp_df)
    cam_obj_df = v2.merge(association_df, cam_image_w_box_w_hkp_df)
    obj_df = v2.merge(cam_obj_df, lidar_box_df)
    t = v2.merge(lidar_df, obj_df, left_group=True, right_group=True)

    # Create a row iterator (continue to the next cell)
    it = iter(t.iterrows())

    # Actually read the data.
    _, row = next(it)

    # Create all component objects
    camera_hkp = v2.CameraHumanKeypointsComponent.from_dict(row)
    camera_box = v2.CameraBoxComponent.from_dict(row)
    camera_image = v2.CameraImageComponent.from_dict(row)
    lidar_box = v2.LiDARBoxComponent.from_dict(row)
    lidar = v2.LiDARComponent.from_dict(row)

    print(
        f"Found {len(lidar_box.key.laser_object_id)} objects on"
        f" {lidar.key.segment_context_name=} {lidar.key.frame_timestamp_micros=}"
    )
    for laser_object_id, camera_object_id, camera_name, cam_kp_x in zip(
        lidar_box.key.laser_object_id,
        camera_box.key.camera_object_id,
        camera_image.key.camera_name,
        camera_hkp.camera_keypoints.keypoint_2d.location_px.x,
    ):
        print(
            f"\t{laser_object_id=} {camera_object_id=} {camera_name=} with"
            f" {len(cam_kp_x)} camera keypoints"
        )
elif mode == "viz-lidar":
    context_name = "10017090168044687777_6380_000_6400_000"

    lidar_df = read("training/lidar", context=context_name)
    calibration_df = read("training/lidar_calibration", context=context_name)
    lidar_pose_df = read("training/lidar_pose", context=context_name)
    vehicle_pose_df = read("training/vehicle_pose", context=context_name)
    
    # lidar_df.to_csv("output/waymo-dataset/lidar_df.csv", single_file=True)
    # calibration_df.to_csv("output/waymo-dataset/lidar_calibration_df.csv", single_file=True)
    # lidar_pose_df.to_csv("output/waymo-dataset/lidar_pose_df.csv", single_file=True)
    # vehicle_pose_df.to_csv("output/waymo-dataset/vehicle_pose_df.csv", single_file=True)
    
    
    # Per-pixel LiDAR pose is provided for the TOP LiDAR.
    top_lidar_df = lidar_df[lidar_df["key.laser_name"] == 1]

    merged_df = v2.merge(top_lidar_df, calibration_df)
    merged_df = v2.merge(merged_df, lidar_pose_df)
    merged_df = v2.merge(merged_df, vehicle_pose_df)
    
    merged_df.to_csv("output/waymo-dataset/merged_df.csv", single_file=True)

    # grab the first row of the df
    result = next(iter(merged_df.iterrows()), None)
    if result is None:
        raise RuntimeError("No merged TOP LiDAR data found")

    _, row = result
    
    # Convert the merged row into Waymo component objects.
    lidar = v2.LiDARComponent.from_dict(row)
    calibration = v2.LiDARCalibrationComponent.from_dict(row)
    lidar_pose = v2.LiDARPoseComponent.from_dict(row)
    vehicle_pose = v2.VehiclePoseComponent.from_dict(row)

    print("Frame:", lidar.key.frame_timestamp_micros)
    print("Sensor:", lidar_module.LaserName(lidar.key.laser_name).name)
    print("Range-image shape:", lidar.range_image_return1.shape)

    # Convert the first-return range image into motion-corrected XYZ points.
    xyz = lidar_utils.convert_range_image_to_point_cloud(
        range_image=lidar.range_image_return1,
        calibration=calibration,
        pixel_pose=lidar_pose.range_image_return1,
        frame_pose=vehicle_pose,
        keep_polar_features=False,
    ).numpy()
    
    print("Point-cloud shape:", xyz.shape)
    print("First point:", xyz[0]) 
    
    
    ############ plot a top down view #################
    # Keep nearby points to make the scene easier to see.
    nearby = (
        (np.abs(xyz[:, 0]) < 75)
        & (np.abs(xyz[:, 1]) < 75)
    )

    points = xyz[nearby]

    plt.figure(figsize=(12, 10))

    scatter = plt.scatter(
        points[:, 0],
        points[:, 1],
        c=points[:, 2],  # color by height
        s=0.3,
        cmap="viridis",
    )

    plt.scatter(
        0,
        0,
        c="red",
        marker="x",
        s=100,
        label="Vehicle origin",
    )

    plt.xlabel("X — forward (meters)")
    plt.ylabel("Y — left (meters)")
    plt.title("Waymo TOP LiDAR — first return")
    plt.axis("equal")
    plt.grid(alpha=0.2)
    plt.colorbar(scatter, label="Z — height (meters)")
    plt.legend()
    plt.show()
    plt.savefig("output/waymo-dataset/top_lidar_scan_top_down.png", dpi=300, bbox_inches="tight")



    ############## plot a 3d view ###################
    # Subsample because rendering every point in 3D is slow.
    sample = points[::10]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(projection="3d")

    plot = ax.scatter(
        sample[:, 0],
        sample[:, 1],
        sample[:, 2],
        c=sample[:, 2],
        s=0.5,
        cmap="viridis",
    )

    ax.set_xlabel("X — forward")
    ax.set_ylabel("Y — left")
    ax.set_zlabel("Z — up")
    ax.set_title("Waymo TOP LiDAR")
    fig.colorbar(plot, ax=ax, label="Height")
    plt.show()
    plt.savefig("output/waymo-dataset/top_lidar_scan_3d.png", dpi=300, bbox_inches="tight")
    