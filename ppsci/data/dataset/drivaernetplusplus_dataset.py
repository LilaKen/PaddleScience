# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu mohamed.elrefaie@tum.de

This module is part of the research presented in the paper:
"DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks".

The module defines two Paddle Datasets for loading and transforming 3D car models from the DrivAerNet++ dataset:
1. DrivAerNetPlusPlusDataset: Handles point cloud data, allowing loading, transforming, and augmenting 3D car models.
"""


from __future__ import annotations

import logging
import os
from typing import Callable
from typing import Optional
from typing import Tuple

import numpy as np
import paddle
import pandas as pd


class DrivAerNetPlusPlusDataset(paddle.io.Dataset):
    """
    Paddle Dataset class for the DrivAerNet dataset, handling loading, transforming, and augmenting 3D car models.

    This dataset is designed for tasks involving aerodynamic simulations and deep learning models,
    specifically for predicting aerodynamic coefficients (e.g., Cd values) from 3D car models.

    Args:
        input_keys (Tuple[str, ...]): Tuple of strings specifying the input keys.
            These keys correspond to the features extracted from the dataset,
            typically the 3D vertices of car models.
            Example: ("vertices",)
        label_keys (Tuple[str, ...]): Tuple of strings specifying the label keys.
            These keys correspond to the ground-truth labels, such as aerodynamic
            coefficients (e.g., Cd values).
            Example: ("cd_value",)
        weight_keys (Tuple[str, ...]): Tuple of strings specifying the weight keys.
            These keys represent optional weighting factors used during model training
            to handle class imbalance or sample importance.
            Example: ("weight_keys",)
        subset_dir (str): Path to the directory containing subsets of the dataset.
            This directory is used to divide the dataset into different subsets
            (e.g., train, validation, test) based on provided IDs.
        ids_file (str): Path to the file containing the list of IDs for the subset.
            The file specifies which models belong to the current subset (e.g., training IDs).
        root_dir (str): Root directory containing the 3D STL files of car models.
            Each 3D model is expected to be stored in a file named according to its ID.
        csv_file (str): Path to the CSV file containing metadata for the car models.
            The CSV file includes information such as aerodynamic coefficients,
            and may also map model IDs to specific attributes.
        num_points (int): Number of points to sample or pad each 3D point cloud to.
            If the model has more points than `num_points`, it will be subsampled.
            If it has fewer points, zero-padding will be applied.
        transform (Optional[Callable]): Optional transformation function applied to each sample.
            This can include augmentations like scaling, rotation, or jittering.
        pointcloud_exist (bool): Whether the point clouds are pre-processed and saved as `.pt` files.
            If `True`, the dataset will directly load the pre-saved point clouds
            instead of generating them from STL files.

    Examples:
    >>> import ppsci
    >>> dataset = ppsci.data.dataset.DrivAerNetPlusPlusDataset(
    ...     input_keys=("vertices",),
    ...     label_keys=("cd_value",),
    ...     weight_keys=("weight_keys",),
    ...     subset_dir="/path/to/subset_dir",
    ...     ids_file="train_ids.txt",
    ...     root_dir="/path/to/DrivAerNetPlusPlusDataset",
    ...     csv_file="/path/to/aero_metadata.csv",
    ...     num_points=1024,
    ...     transform=None,
    ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        weight_keys: Tuple[str, ...],
        subset_dir: str,
        ids_file: str,
        root_dir: str,
        csv_file: str,
        num_points: int,
        transform: Optional[Callable] = None,
        pointcloud_exist: bool = True,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.input_keys = input_keys
        self.label_keys = label_keys
        self.weight_keys = weight_keys
        self.subset_dir = subset_dir
        self.ids_file = ids_file
        self.cache = {}

        try:
            self.data_frame = pd.read_csv(csv_file)
        except Exception as e:
            logging.error(f"Failed to load CSV file: {csv_file}. Error: {e}")
            raise
        self.transform = transform
        self.num_points = num_points
        self.pointcloud_exist = pointcloud_exist

        try:
            with open(os.path.join(self.subset_dir, self.ids_file), "r") as file:
                subset_ids = file.read().split()
            self.subset_indices = self.data_frame[
                self.data_frame["Design"].isin(subset_ids)
            ].index.tolist()
            self.data_frame = self.data_frame.loc[self.subset_indices].reset_index(
                drop=True
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error loading subset file {self.ids_file}: {e}")

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data_frame)

    def _sample_or_pad_vertices(
        self, vertices: paddle.Tensor, num_points: int
    ) -> paddle.Tensor:
        """
        Subsamples or pads the vertices of the model to a fixed number of points.

        Args:
            vertices: The vertices of the 3D model as a paddle.Tensor.
            num_points: The desired number of points for the model.

        Returns:
            The vertices standardized to the specified number of points.
        """
        num_vertices = vertices.shape[0]
        if num_vertices > num_points:
            indices = np.random.choice(num_vertices, num_points, replace=False)
            vertices = vertices[indices]
        elif num_vertices < num_points:
            padding = paddle.zeros(
                shape=(num_points - num_vertices, 3), dtype="float32"
            )
            vertices = paddle.concat(x=(vertices, padding), axis=0)
        return vertices

    def _load_point_cloud(self, design_id: str) -> Optional[paddle.Tensor]:
        load_path = os.path.join(self.root_dir, f"{design_id}.paddle_tensor")
        if os.path.exists(load_path) and os.path.getsize(load_path) > 0:
            try:
                vertices = paddle.load(path=str(load_path))

                num_vertices = vertices.shape[0]

                if num_vertices > self.num_points:
                    indices = np.random.choice(
                        num_vertices, self.num_points, replace=False
                    )
                    vertices = vertices.numpy()[indices]
                    vertices = paddle.to_tensor(vertices)

                return vertices
            except (EOFError, RuntimeError, ValueError) as e:
                raise Exception(
                    f"Error loading point cloud from {load_path}: {e}"
                ) from e

    def __getitem__(self, idx: int, apply_augmentations: bool = True):
        """
        Retrieves a sample and its corresponding label from the dataset, with an option to apply augmentations.

        Args:
            idx (int): Index of the sample to retrieve.
            apply_augmentations (bool, optional): Whether to apply data augmentations. Defaults to True.

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]: The sample (point cloud) and its label (Cd value).
        """
        if paddle.is_tensor(idx):
            idx = idx.tolist()

        if idx in self.cache:
            return self.cache[idx]

        row = self.data_frame.iloc[idx]
        design_id = row["Design"]
        cd_value = row["Average Cd"]
        if self.pointcloud_exist:
            try:
                vertices = self._load_point_cloud(design_id)
                if vertices is None:
                    raise ValueError(
                        f"Point cloud for design {design_id} is not found or corrupted."
                    )
            except Exception as e:
                raise ValueError(
                    f"Failed to load point cloud for design {design_id}: {e}"
                )

        cd_value = np.array(float(cd_value), dtype=np.float32).reshape([-1])
        self.cache[idx] = (
            {self.input_keys[0]: vertices},
            {self.label_keys[0]: cd_value},
            {self.weight_keys[0]: np.array(1, dtype=np.float32)},
        )

        return (
            {self.input_keys[0]: vertices},
            {self.label_keys[0]: cd_value},
            {self.weight_keys[0]: np.array(1, dtype=np.float32)},
        )