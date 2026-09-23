import argparse
import json
import os
import sys

import cv2
import numpy as np


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def validate_manifest(dataset_dir, manifest_name, expected_width, expected_height):
    manifest_path = os.path.join(dataset_dir, manifest_name)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(manifest_path)

    manifest = load_json(manifest_path)
    frames = manifest.get("frames", [])
    if not frames:
        raise ValueError(f"{manifest_name} contains no frames")

    timestamps = []
    positions = []

    if int(manifest["w"]) != expected_width or int(manifest["h"]) != expected_height:
        raise ValueError(f"{manifest_name} resolution does not match metadata.json")

    for frame_index, frame in enumerate(frames):
        for field in ("file_path", "lidar_file_path", "timestamp", "transform_matrix"):
            if field not in frame:
                raise ValueError(f"Frame {frame_index} is missing '{field}'")

        image_path = os.path.join(dataset_dir, frame["file_path"])
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Frame {frame_index} image cannot be read: {image_path}")
        if image.shape[1] != expected_width or image.shape[0] != expected_height:
            raise ValueError(f"Frame {frame_index} image has unexpected dimensions")

        lidar_path = os.path.join(dataset_dir, frame["lidar_file_path"])
        points = np.load(lidar_path)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Frame {frame_index} LiDAR scan must have shape (N, 3)")
        if not np.isfinite(points).all():
            raise ValueError(f"Frame {frame_index} LiDAR scan contains non-finite values")

        transform = np.asarray(frame["transform_matrix"], dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f"Frame {frame_index} has an invalid camera transform")

        if not np.isfinite(float(frame["timestamp"])):
            raise ValueError(f"Frame {frame_index} has an invalid timestamp")

        timestamps.append(float(frame["timestamp"]))
        positions.append(transform[:3, 3])

    if any(later <= earlier for earlier, later in zip(timestamps, timestamps[1:])):
        raise ValueError(f"{manifest_name} timestamps are not strictly increasing")

    return len(frames), np.asarray(positions)


def main():
    parser = argparse.ArgumentParser(description="Validate a captured Gaussian Splatting dataset.")
    parser.add_argument("dataset_dir", help="Path to a dataset containing transforms.json")
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    metadata = load_json(metadata_path)
    expected_width, expected_height = metadata["image_resolution"]

    manifest_counts = {}
    timestamps = []
    positions = []
    for split_name in ("train", "val", "test"):
        count, split_positions = validate_manifest(
            dataset_dir,
            f"transforms_{split_name}.json",
            int(expected_width),
            int(expected_height),
        )
        manifest_counts[split_name] = count
        positions.append(split_positions)

    total_count, all_positions = validate_manifest(
        dataset_dir,
        "transforms.json",
        int(expected_width),
        int(expected_height),
    )
    if sum(manifest_counts.values()) != total_count:
        raise ValueError("Train/validation/test frame counts do not match transforms.json")

    if not np.allclose(np.concatenate(positions), all_positions):
        raise ValueError("Split manifests do not preserve transforms.json order")

    if len(all_positions) > 1 and np.allclose(all_positions, all_positions[0]):
        raise ValueError("Camera trajectory contains no movement")

    ply_path = os.path.join(dataset_dir, "lidar_initialization.ply")
    if not os.path.isfile(ply_path):
        raise FileNotFoundError(ply_path)

    print(f"Dataset valid: {total_count} frames at {expected_width}x{expected_height}")
    print(f"Splits: {manifest_counts}")


if __name__ == "__main__":
    main()
