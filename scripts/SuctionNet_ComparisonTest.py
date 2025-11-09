# This script was produced from the notebook available at ../notebooks/9-SuctionNet_ComparisonTest.ipynb
# For further details on this code, check the aforementioned notebook


# 1 Setup

import MinkowskiEngine as ME
print(f'MinkowskiEngine version: {ME.__version__}')
import torch
print(f'PyTorch version: {torch.__version__}')

import matplotlib
matplotlib.use('Agg')  # ensures no GUI backend is used

from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import open3d as o3d
import pandas as pd
import seaborn as sns
import numpy as np
import copy
import sys
import os
import io
from urllib.request import urlretrieve
from datetime import datetime
from zoneinfo import ZoneInfo
from contextlib import redirect_stdout
from typing import Literal

sys.path.append(os.path.abspath("../source/FCGF"))
from model import load_model
from util.visualization import get_colored_point_cloud_feature
from util.misc import extract_features
from scripts.benchmark_util import run_ransac
from util.pointcloud import make_open3d_point_cloud, make_open3d_feature_from_numpy


# 1.1 Auxiliary Functions

# 1.1.1 Visualization (Not Applicable for the script case)

# 1.1.2 Iteration Counter

def get_iterations(captured_output):
    """
    Parses the captured debug output to extract the iteration counts for each alignment.
    Returns a DataFrame where each row corresponds to one alignment, with columns for the scene ID,
    target ID, source ID, RANSAC iterations, and ICP iterations.
    """
    
    # List to store iteration data for each alignment.
    iterations = []

    # Clean the captured output: remove tabs, extra spaces, and split it into individual lines.
    captured_output = captured_output.replace('[Open3D DEBUG]', '').replace('\t', '').strip().split('\n')

    # Loops through the output lines
    for line in captured_output:
        
        # Split the line into words.
        line = line.split()
        if not line:  # Skip empty lines.
            continue

        # Get current scene
        if line[0] == 'Scene:':             # checks if it starts with 'Scene:'
            # Extracts IDs from the line
            scene_id = int(line[1])
            image_id = int(line[4])
            object_id = int(line[7])
            # Resets the ICP iterations counter for the next pair
            ICP_iterations = 0          

        # Get ransac iterations
        elif line[0] == 'RANSAC':
            RANSAC_iterations = int(line[3])
          
        # If the line indicates an ICP iteration, increment the ICP iterations counter
        elif line[0] == 'ICP':
            ICP_iterations += 1

        # When the line starts with 'Done', it signals the end of the current alignment
        elif line[0] == 'Done':
            iterations.append({
                'Scene': scene_id,
                'Image': image_id,
                'Object': object_id,
                'RANSAC Iterations': RANSAC_iterations,
                'ICP Iterations': ICP_iterations
            })

    iterations_df = pd.DataFrame(iterations)
    return iterations_df


def augment_results(results, captured_output):
    """
    Augments the main results DataFrame with iteration counts by merging it with the iterations DataFrame.
    The merge is performed as a left join on the 'Scene', 'Image', and 'Object' columns.
    """
    iterations_df = get_iterations(captured_output)
    augmented_results = results.merge(iterations_df, 'left', on=['Scene', 'Image', 'Object'])

    # Reoder the columns
    desired_order = ['Setting', 'Scene', 'Image', 'Object', 'RANSAC Iterations', 'ICP Iterations',
                     'RANSAC: Fitness', 'ICP: Fitness', 'RANSAC: Inlier RMSE', 'ICP: Inlier RMSE',
                     'RANSAC: Transformation', 'ICP: Transformation']
    augmented_results = augmented_results[desired_order]

    return augmented_results


class Tee(io.StringIO):
    """
    A 'tee' that writes output to multiple streams:
    - the console (sys.__stdout__)
    - an internal buffer (which you can retrieve later)
    """
    def write(self, text):
        # write to console
        sys.__stdout__.write(text)
        # Also write to this StringIO's buffer (use super() since it is the parent class of this one)
        super().write(text)

    def flush(self):
        # Ensure both this buffer and console are flushed
        sys.__stdout__.flush()
        super().flush()


# 2 Data Preparation (Not Applicable for the script case)
# --> in the notebook, we already verified that in this case it is not necessary to treat data


# 3 Initial Translation

def initial_translation(pcd):
    """
    Computes the initial translation matrices to center the given point cloud at the origin.

    Args:
        pcd (open3d.geometry.PointCloud): Point cloud to center

    Returns:
        numpy.ndarray: Translation matrix to center the point cloud
    """

    T = np.eye(4)
    T[:3, 3] = [0, 0, 0] - pcd.get_center()
    
    pcd_moved = copy.deepcopy(pcd)
    pcd_moved.transform(T)
    
    return T, pcd_moved


# 4 Pipeline Implementation

# 4.1 Geometric-Only Pipeline

def preprocess_cloud(pcd, voxel_size):
    """
    Performs the preprocessing of a given cloud. Thus, this function:
    downsamples, estimate the normals and extract features (through
    FPFH algorithm) of the input cloud.

    Args:
        pcd (open3d.geometry.PointCloud): Input point cloud
        voxel_size (float): Resulting size of voxels after downsampling

    Returns:
        open3d.geometry.PointCloud: Downsampled cloud
        open3d.registration.Feature: Features for registration
    """

    #downsampling
    pcd_down = pcd.voxel_down_sample(voxel_size)

    #normals estimation
    radius_normal = 2*voxel_size
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    #FPFH
    radius_feature = 5*voxel_size
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_down, pcd_fpfh


def execute_GlobalRegistration(source_down, target_down, source_fpfh, target_fpfh, inlier_th):
    """
    Executes the Global Registration (through RANSAC algorithm) of
    the input source cloud (after being downsampled), given its
    FPFH features, in order to align it to the target cloud.

    Args:
        source_down (open3d.geometry.PointCloud): Downsampled source cloud
        target_down (open3d.geometry.PointCloud): Downsampled target cloud
        source_fpfh (open3d.registration.Feature): FPFH features of source cloud
        target_fpfh (open3d.registration.Feature): FPFH features of target cloud
        inlier_th (float): Maximum threshold distance for a pair alignment to be valid

    Returns:
        open3d.pipelines.registration.RegistrationResult: Class that contains the registration results
    """

    global_registration = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                          source_down, target_down, source_fpfh, target_fpfh,
                          True,                                                                             #mutual filter activated
                          inlier_th,                                                                        #max_correspondence_distance
                          o3d.pipelines.registration.TransformationEstimationPointToPoint(False),           #point-to-point estimation without scaling
                          3,                                                                                #ransac_n
                          [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),          #checkers to be used
                           o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(inlier_th)],
                          o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))              #convergence criteria

    return global_registration


def execute_ICPrefinement(source, target, inlier_th, trans_init, voxel_size):
    """
    Executes the local ICP refinement of a initial transformation.

    Args:
        source (open3d.geometry.PointCloud): Source cloud
        target (open3d.geometry.PointCloud): Target cloud
        dist_threhsold (float): Threshold distance for a correspondence pair to be considered valid
        trans_init (numpy.ndarray): Initial transformation matrix
        voxel_size (float): Resulting size of voxels after downsampling

    Returns:
        open3d.pipelines.registration.RegistrationResult: Class that contains the registration results
    """

    #target normals estimation
    radius_normal = 2*voxel_size
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    #performs the point-to-plane ICP
    ICP_registration = o3d.pipelines.registration.registration_icp(source, target, inlier_th, trans_init,
                                                                   o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return ICP_registration


def GlobalRegistration_withICP(source, target, voxel_size, inlier_th):
    """
    Executes the complete pipeline of Global Registration (FPFH + RANSAC) with ICP refinement.

    Args:
        source (open3d.geometry.PointCloud): Source cloud
        target (open3d.geometry.PointCloud): Target cloud
        voxel_size (float): Resulting size of voxels after downsampling
        dist_threhsold_ICP (float): Threshold distance for a correspondence pair to be considered valid during ICP

    Returns:
        open3d.pipelines.registration.RegistrationResult: Class that contains the registration results
    """

    #Preprocessing (downsampling + FPFH feature extraction)
    source_down, source_fpfh = preprocess_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_cloud(target, voxel_size)

    #Global Alignment (RANSAC)
    result_ransac = execute_GlobalRegistration(source_down, target_down, source_fpfh, target_fpfh, inlier_th)

    #Local Refinement (ICP)
    trans_init = result_ransac.transformation
    result_icp = execute_ICPrefinement(source, target, inlier_th, trans_init, voxel_size)

    return result_ransac, result_icp


def run_GO_single_pair(setting: str, dataset_dir: str, voxel_size: float, inlier_th: float, results: list, scene_id: int, image_id: int, obj_id: int):
    """Auxiliary function that runs the registration methodoly (initial translation, RANSAC and ICP) to a given scene, image and object."""

    print(f'Scene: {scene_id:04d} | Image: {image_id:04d} | Object: {obj_id:03d}')

    # Degine source path (CAD's dense point cloud)    
    model_dir = f'{dataset_dir}/models/ply'
    source_path = f'{model_dir}/{obj_id-1:03d}.ply'

    # Define target path depending on the scenario we are testing
    if setting == 'full_scene':
        target_path = f'{dataset_dir}/acquired/full_scene/scene_{scene_id:04d}/scene_{scene_id:04d}_image_{image_id:04d}.ply'
    elif setting == 'single_object':
        target_path = f'{dataset_dir}/acquired/single_object/scene_{scene_id:04d}/image_{image_id:04d}/scene_{scene_id:04d}_image_{image_id:04d}_object_{obj_id:02d}.ply'

    # Read clouds
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)

    # Apply the initial translation to centralize both clouds around the origin
    T_source_0, source_moved = initial_translation(source)
    T_target_0, target_moved = initial_translation(target)

    # Execute RANSAC + ICP alignment
    result_ransac, result_icp = GlobalRegistration_withICP(source_moved, target_moved, voxel_size, inlier_th)

    # Update the obtained transformation to include also the initial translation
    T_ransac = np.linalg.inv(T_target_0) @ result_ransac.transformation @ T_source_0
    T_icp   = np.linalg.inv(T_target_0) @ result_icp.transformation   @ T_source_0

    # Append the new entry in the results table and returns
    results.append({
        "Setting": setting,
        "Scene": scene_id,
        "Image": image_id,
        "Object": obj_id,
        "RANSAC: Fitness": result_ransac.fitness,
        "ICP: Fitness": result_icp.fitness,
        "RANSAC: Inlier RMSE": result_ransac.inlier_rmse,
        "ICP: Inlier RMSE": result_icp.inlier_rmse,
        "RANSAC: Transformation": T_ransac,
        "ICP: Transformation": T_icp
    })
    print('\tDone')
    return results


def GO_pipeline(dataset_dir: str, setting: Literal['full_scene', 'single_object'], voxel_size: float, inlier_th: float):

    # Ensure setting is valid
    if setting not in ('full_scene', 'single_object'):
        raise ValueError(f"Invalid setting '{setting}'. Must be 'full_scene' or 'single_object'.")
    print(f"Using setting: {setting}\n-------------------------")

    # Initialize the list which will store all results
    results = []

    # When considering targets as the full clouds with all objects
    if setting == 'full_scene':
        full_scene_dir = f'{dataset_dir}/acquired/full_scene'                   # set path to folder with full scene clouds
        for scene in sorted(os.listdir(full_scene_dir)):                        # for each scene (sorted numerically)
            scene_id = int(scene.split('_')[-1])                                # get scene id
            # if scene_id > 91: continue                                          # optional limit. Useful when debugging
            scene_dir = f'{full_scene_dir}/{scene}'                             # get path of a specific scene's folder
            label_dir = f'{dataset_dir}/acquired/raw/{scene}/realsense/label'   # get path of labels for that scene
            for image in sorted(os.listdir(scene_dir)):                         # for each image (sorted numerically)
                image_id = int(image.split('_')[-1].split('.')[0])              # get its id
                # if image_id > 3: continue                                       # optional limit. Useful when debugging
                label = o3d.io.read_image(f'{label_dir}/{image_id:04d}.png')    # get the label image of the current image
                label = np.array(label)                                         # convert it to a numpy array 
                obj_list = np.unique(label)                                     # get list of all objects included in the image
                obj_list = obj_list[obj_list != 0]                              # filter out the background (id 0)
                for obj_id in obj_list:                                         # for each object, run the alignment
                    results = run_GO_single_pair(setting, dataset_dir, voxel_size, inlier_th, results, scene_id, image_id, obj_id)

    # When considering targets as individual clouds of each object
    elif setting == 'single_object':
        single_object_dir = f'{dataset_dir}/acquired/single_object'
        for scene in sorted(os.listdir(single_object_dir)):
            scene_id = int(scene.split('_')[-1])
            # if scene_id > 91: continue
            scene_dir = f'{single_object_dir}/{scene}'
            for image in sorted(os.listdir(scene_dir)):
                image_id = int(image.split('_')[-1])
                # if image_id > 3: continue
                image_dir = f'{scene_dir}/{image}'
                for obj_file in os.listdir(image_dir):                          # we get object id directly from the file name (no need for label)
                    obj_id = int(obj_file.split('_')[-1].split('.')[0])
                    results = run_GO_single_pair(setting, dataset_dir, voxel_size, inlier_th, results, scene_id, image_id, obj_id)
    
    return results


def run_GO_pipeline(dataset_dir: str, setting: Literal['full_scene', 'single_object'], voxel_size: float, inlier_th: float, output_dir: str):

    # Create output folder to store our results
    GO_out_dir = f"{output_dir}/geometric_only-{setting}"
    os.makedirs(GO_out_dir, exist_ok=True)

    # Create a Tee object to capture the output while printing to the console
    tee_buffer = Tee()

    # Redirect stdout to the Tee object during the pipeline execution
    with redirect_stdout(tee_buffer):

        # Set Open3D's verbosity level to Debug to capture detailed iteration information
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

        results = GO_pipeline(dataset_dir, setting, voxel_size, inlier_th)

        # Returns Open3D's verbosity level to default mode
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # Convert the results to a pandas table
    results = pd.DataFrame(results)

    # Retrieve the captured output as a string
    captured_output = tee_buffer.getvalue()

    # Merge the iteration information into the main results DataFrame
    results = augment_results(results, captured_output)

    # Save results as .csv file
    results.to_csv(f'{GO_out_dir}/registration_table_full.csv', index=False)

    # Filter results for good alignments (i.e. fitness > 30%) and save the filtered version too
    filt_results = results[results['ICP: Fitness'] > 0.3].copy()
    filt_results.to_csv(f'{GO_out_dir}/registration_table_filtered.csv', index=False)

    return results, filt_results


# 4.2 Deep-Learning Pipeline

# 4.2.1 Model Initialization

def download_weights(weights_url, weights_filename):
    # Check if the weight folder has already been created, otherwise creates it
    fcgf_weights_folder = '../weights/FCGF'
    if not os.path.exists(fcgf_weights_folder):
        os.makedirs(fcgf_weights_folder)

    # Check if the selected pre-trained weights were already downloaded, otherwise download them
    fcgf_weight_path = os.path.join(fcgf_weights_folder, weights_filename)
    if not os.path.isfile(fcgf_weight_path):
        print(f'Downloading weight at {fcgf_weight_path}...')
        urlretrieve(weights_url, fcgf_weight_path)
    else:
        print(f'Selected weights already available at {fcgf_weight_path}')
    return fcgf_weight_path


def initialize_model(model_weights):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # the model is define by the weights in fcgf_weight_path
    checkpoint = torch.load(model_weights, map_location=device, weights_only=False)
    config = checkpoint['config']

    num_feats = 1
    Model = load_model(config.model)
    model = Model(
        num_feats,
        config.model_n_out,
        bn_momentum=0.05,
        normalize_feature=config.normalize_feature,
        conv1_kernel_size=config.conv1_kernel_size,
        D=3)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    model = model.to(device)
    return model, device


def setup_FCGF(weights_url, weights_filename):

    fcgf_weight_path = download_weights(weights_url, weights_filename)
    model, device = initialize_model(model_weights=fcgf_weight_path)

    return model, device


# 4.2.2 Feature Extraction (and Visualization -- not Applicable for the script case)

def extract_FCGF(pcd, voxel_size, model, device):
    with torch.no_grad():
        # Extract FCGF through a function given by the author at ../source/FCGF/util/misc.py
        xyz_down, features = extract_features(
            model, xyz=np.array(pcd.points), rgb=None, normal=None,
            voxel_size=voxel_size, device=device, skip_check=True
        )
    return xyz_down, features


# 4.2.3 RANSAC Registration

def do_single_pair_matching(xyz_down_src, features_src, xyz_down_tgt, features_tgt, voxel_size, inlier_th):
  
    #Downsampled clouds
    xyz_src = make_open3d_point_cloud(xyz_down_src)
    xyz_tgt = make_open3d_point_cloud(xyz_down_tgt)

    # Features
    feat_src = make_open3d_feature_from_numpy(features_src.detach().cpu().numpy())
    feat_tgt = make_open3d_feature_from_numpy(features_tgt.detach().cpu().numpy())

    # Transformation
    if len(xyz_src.points) < len(xyz_tgt.points):
        trans = run_ransac(xyz_src, xyz_tgt, feat_src, feat_tgt, voxel_size, inlier_th)
    else:
        trans = run_ransac(xyz_tgt, xyz_src, feat_tgt, feat_src, voxel_size, inlier_th)
        trans = np.linalg.inv(trans)
    
    return trans


# 4.2.4 ICP Registration (and Full Pipeline + Iteration Counter)

def run_FCGF(pcd_src, pcd_tgt, voxel_size, inlier_th, model, device):
    
    # Feature Extraction
    xyz_down_src, features_src = extract_FCGF(pcd_src, voxel_size, model, device)
    xyz_down_tgt, features_tgt = extract_FCGF(pcd_tgt, voxel_size, model, device)

    # RANSAC
    # This returns just the transformation
    ransac_trans = do_single_pair_matching(xyz_down_src, features_src,
                                     xyz_down_tgt, features_tgt,
                                     voxel_size, inlier_th)
    # So we use the evaluate function to return the full registration results
    # In this way, we can access fitness, inlier_tmse and transformation
    result_ransac = o3d.pipelines.registration.evaluate_registration(pcd_src, pcd_tgt, inlier_th, ransac_trans)

    # ICP
    result_icp = execute_ICPrefinement(pcd_src, pcd_tgt, inlier_th, ransac_trans, voxel_size)

    return result_ransac, result_icp, xyz_down_src, features_src, xyz_down_tgt, features_tgt


def save_xyz_and_features(DL_out_dir: str, scene_id: int, image_id: int, obj_id: int, xyz_down, features, model: bool = False):
    """Auxiliary function to store the downsampled clouds and the obtained features. It allow us to visualize the features"""

    # Convert to numpy
    if isinstance(xyz_down, torch.Tensor):
        xyz_np = xyz_down.detach().cpu().numpy()
    else:
        xyz_np = np.asarray(xyz_down)

    if isinstance(features, torch.Tensor):
        feat_np = features.detach().cpu().numpy()
    else:
        feat_np = np.asarray(features)

    # Create output folders
    save_dir = f"{DL_out_dir}/xyz_and_features/{'models' if model else 'acquired'}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Define filename as save it
    filename = f'{save_dir}/scene_{scene_id:04d}_image_{image_id:04d}_object_{obj_id:03d}.npz'
    np.savez_compressed(filename, xyz=xyz_np.astype(np.float32), feat=feat_np.astype(np.float32))
    return


def run_DL_single_pair(setting: str, dataset_dir: str, DL_out_dir: str, voxel_size: float, inlier_th: float,
                       results: list, scene_id: int, image_id: int, obj_id: int, model, device):
    """Auxiliary function that runs the registration methodoly (initial translation, RANSAC and ICP) to a given scene, image and object."""

    print(f'Scene: {scene_id:04d} | Image: {image_id:04d} | Object: {obj_id:03d}')

    # Degine source path (CAD's dense point cloud)    
    model_dir = f'{dataset_dir}/models/ply'
    source_path = f'{model_dir}/{obj_id-1:03d}.ply'

    # Define target path depending on the scenario we are testing
    if setting == 'full_scene':
        target_path = f'{dataset_dir}/acquired/full_scene/scene_{scene_id:04d}/scene_{scene_id:04d}_image_{image_id:04d}.ply'
    elif setting == 'single_object':
        target_path = f'{dataset_dir}/acquired/single_object/scene_{scene_id:04d}/image_{image_id:04d}/scene_{scene_id:04d}_image_{image_id:04d}_object_{obj_id:02d}.ply'

    # Read clouds
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)

    # Apply the initial translation to centralize both clouds around the origin
    T_source_0, source_moved = initial_translation(source)
    T_target_0, target_moved = initial_translation(target)

    # Execute RANSAC + ICP alignment
    result_ransac, result_icp, xyz_down_src, features_src, xyz_down_tgt, features_tgt  = run_FCGF(source_moved, target_moved, voxel_size, inlier_th, model, device)

    # Save features and downsampled clouds
    save_xyz_and_features(DL_out_dir, scene_id, image_id, obj_id, xyz_down_src, features_src, model=True)
    save_xyz_and_features(DL_out_dir, scene_id, image_id, obj_id, xyz_down_tgt, features_tgt, model=False)

    # Update the obtained transformations to include also the initial translation
    T_ransac = np.linalg.inv(T_target_0) @ result_ransac.transformation @ T_source_0
    T_icp   = np.linalg.inv(T_target_0) @ result_icp.transformation   @ T_source_0

    # Append the new entry in the results table and returns
    results.append({
        "Setting": setting,
        "Scene": scene_id,
        "Image": image_id,
        "Object": obj_id,
        "RANSAC: Fitness": result_ransac.fitness,
        "ICP: Fitness": result_icp.fitness,
        "RANSAC: Inlier RMSE": result_ransac.inlier_rmse,
        "ICP: Inlier RMSE": result_icp.inlier_rmse,
        "RANSAC: Transformation": T_ransac,
        "ICP: Transformation": T_icp
    })
    print('\tDone')
    return results


def DL_pipeline(dataset_dir: str, DL_out_dir: str, setting: Literal['full_scene', 'single_object'], voxel_size: float, inlier_th: float, model, device):

    # Ensure setting is valid
    if setting not in ('full_scene', 'single_object'):
        raise ValueError(f"Invalid setting '{setting}'. Must be 'full_scene' or 'single_object'.")
    print(f"Using setting: {setting}\n-------------------------")

    # Initialize the list which will store all results
    results = []

    # When considering targets as the full clouds with all objects
    if setting == 'full_scene':
        full_scene_dir = f'{dataset_dir}/acquired/full_scene'                   # set path to folder with full scene clouds
        for scene in sorted(os.listdir(full_scene_dir)):                        # for each scene (sorted numerically)
            scene_id = int(scene.split('_')[-1])                                # get scene id
            # if scene_id > 91: continue                                          # optional limit. Useful when debugging
            scene_dir = f'{full_scene_dir}/{scene}'                             # get path of a specific scene's folder
            label_dir = f'{dataset_dir}/acquired/raw/{scene}/realsense/label'   # get path of labels for that scene
            for image in sorted(os.listdir(scene_dir)):                         # for each image (sorted numerically)
                image_id = int(image.split('_')[-1].split('.')[0])              # get its id
                # if image_id > 3: continue                                       # optional limit. Useful when debugging
                label = o3d.io.read_image(f'{label_dir}/{image_id:04d}.png')    # get the label image of the current image
                label = np.array(label)                                         # convert it to a numpy array 
                obj_list = np.unique(label)                                     # get list of all objects included in the image
                obj_list = obj_list[obj_list != 0]                              # filter out the background (id 0)
                for obj_id in obj_list:                                         # for each object, run the alignment
                    results = run_DL_single_pair(setting, dataset_dir, DL_out_dir, voxel_size, inlier_th, results, scene_id, image_id, obj_id, model, device)

    # When considering targets as individual clouds of each object
    elif setting == 'single_object':
        single_object_dir = f'{dataset_dir}/acquired/single_object'
        for scene in sorted(os.listdir(single_object_dir)):
            scene_id = int(scene.split('_')[-1])
            # if scene_id > 91: continue
            scene_dir = f'{single_object_dir}/{scene}'
            for image in sorted(os.listdir(scene_dir)):
                image_id = int(image.split('_')[-1])
                # if image_id > 3: continue
                image_dir = f'{scene_dir}/{image}'
                for obj_file in os.listdir(image_dir):                          # we get object id directly from the file name (no need for label)
                    obj_id = int(obj_file.split('_')[-1].split('.')[0])
                    results = run_DL_single_pair(setting, dataset_dir, DL_out_dir, voxel_size, inlier_th, results, scene_id, image_id, obj_id, model, device)
    
    return results


def run_DL_pipeline(dataset_dir: str, setting: Literal['full_scene', 'single_object'], voxel_size: float, inlier_th: float, output_dir: str,
                    weights_url='https://node1.chrischoy.org/data/publications/fcgf/2019-09-18_14-15-59.pth', weights_filename='ResUNetBN2C-16feat-3conv.pth'):

    # Setup the deep-learning model with the desired weights
    model, device = setup_FCGF(weights_url, weights_filename)

    # Create output folder to store our results
    DL_out_dir = f"{output_dir}/deep_learning-{setting}"
    os.makedirs(DL_out_dir, exist_ok=True)

    # Create a Tee object to capture the output while printing to the console
    tee_buffer = Tee()

    # Redirect stdout to the Tee object during the pipeline execution
    with redirect_stdout(tee_buffer):

        # Set Open3D's verbosity level to Debug to capture detailed iteration information
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        
        # Execute the pipeline
        results = DL_pipeline(dataset_dir, DL_out_dir, setting, voxel_size, inlier_th, model, device)

        # Returns Open3D's verbosity level to default mode
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # Convert the results to a pandas table
    results = pd.DataFrame(results)

    # Retrieve the captured output as a string
    captured_output = tee_buffer.getvalue()

    # Merge the iteration information into the main results DataFrame
    results = augment_results(results, captured_output)

    # Saves the results as .csv file
    results.to_csv(f'{DL_out_dir}/registration_table_full.csv', index=False)

    # Filter results for good alignments (i.e. fitness > 30%) and save the filteres version too
    filt_results = results[results['ICP: Fitness'] > 0.3].copy()
    filt_results.to_csv(f'{DL_out_dir}/registration_table_filtered.csv', index=False)

    return results, filt_results


# 5 Testing

timestamp = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime('%Y-%m-%d_%H-%M-%S')
run_name = 'complete_test'
output_dir = f"../output/SuctionNet/{run_name}-{timestamp}"
dataset_dir = '../data/SuctionNet'
inlier_th = 0.005


# 5.1 Running the Pipelines

go_voxel_size = 0.005
GO_full, filt_GO_full = run_GO_pipeline(dataset_dir, 'full_scene', go_voxel_size, inlier_th, output_dir)
print("============================================== Geometric-Only Full-Scene Results ==============================================")
print(GO_full.to_string(index=False))
print("===============================================================================================================================")

GO_object, filt_GO_object = run_GO_pipeline(dataset_dir, 'single_object', go_voxel_size, inlier_th, output_dir)
print("============================================== Geometric-Only Single-Object Results ==============================================")
print(GO_object.to_string(index=False))
print("==================================================================================================================================")

dl_voxel_size = 0.0025
DL_full, filt_DL_full = run_DL_pipeline(dataset_dir, 'full_scene', dl_voxel_size, inlier_th, output_dir)
print("============================================== Deep-Learning Full-Scene Results ==============================================")
print(DL_full.to_string(index=False))
print("==================================================================================================================================")

DL_object, filt_DL_object = run_DL_pipeline(dataset_dir, 'single_object', dl_voxel_size, inlier_th, output_dir)
print("============================================== Deep-Learning Single-Object Results ==============================================")
print(DL_object.to_string(index=False))
print("==================================================================================================================================")


# 5.2 Assessing Recall

if len({len(GO_full), len(GO_object), len(DL_full), len(DL_object)}) != 1:
    print("The lengths don't match. Verify the obtained tables")

pct_go_object = 100 * len(filt_GO_object) / len(GO_object)
pct_go_full = 100 * len(filt_GO_full) / len(GO_full)
pct_dl_object = 100 * len(filt_DL_object) / len(DL_object)
pct_dl_full = 100 * len(filt_DL_full) / len(DL_full)

pct_df = pd.DataFrame(
    {'Setting': ['Single Objects', 'Full Scenes'],
     'Geometric-Only': [pct_go_object, pct_go_full],
     'Deep-Learning': [pct_dl_object, pct_dl_full]}
)

filename = f"{output_dir}/recall_estimated_totals.csv"
pct_df.to_csv(filename, index=False)
print(f'Estimated Recall totals saved at: {filename}')

print("============================================== Estimated Recall ==============================================")
print(pct_df.to_string(index=False))
print("==============================================================================================================")


# 5.3 Assessing Alignment Quality

def assess_results(results, pipeline: Literal['geometric_only', 'deep_learning'], setting: Literal['full_scene', 'single_object'], output_dir, groupby_col="Scene"):
    """
    Given a registration result table, computes the mean performance (and standard deviation) of the
    alignment for all clouds of a specific scene and for that whole split we selected from a dataset.

    Args:
        results (pd.DataFrame): Table containing the registration results of the split

    Returns:
        pd.DataFrame: Table with the overall (mean and standard) performance results
    """

    # Ensure parameters are valid
    if setting not in ('full_scene', 'single_object'):
        raise ValueError(f"Invalid setting '{setting}'. Must be 'full_scene' or 'single_object'.")
    if pipeline not in ('geometric_only', 'deep_learning'):
        raise ValueError(f"Invalid pipeline '{pipeline}'. Must be 'geometric_only' or 'deep_learning'.")

    # Get the non requested grouping column
    other_id = "Object" if groupby_col == "Scene" else "Scene"

    # 1) Compute per-group mean
    #    Dropping 'Source','Target','Transformation' from the grouping
    #    And, for each "col+group" filter out the 0 entries
    analysis_mean = results.copy()                                                                                                  # create a copy of results
    analysis_mean = analysis_mean.drop([other_id, "Setting", "Image", "RANSAC: Transformation", "ICP: Transformation"], axis="columns")        # remove unnecessary columns
    analysis_mean = analysis_mean.groupby(groupby_col).agg(lambda x: x[x != 0].mean()).reset_index()                                    #  compute the averages of each scene
    analysis_mean = analysis_mean.rename(columns={"RANSAC: Fitness": "RANSAC: Mean Fitness",
                                                  "ICP: Fitness": "ICP: Mean Fitness",
                                                  "RANSAC: Inlier RMSE": "RANSAC: Mean Inlier RMSE",
                                                  "ICP: Inlier RMSE": "ICP: Mean Inlier RMSE",
                                                  "RANSAC Iterations": "Mean RANSAC Iterations",
                                                  "ICP Iterations": "Mean ICP Iterations"})

    # 2) Compute per-group standard deviation
    analysis_std = results.copy()
    analysis_std = analysis_std.drop([other_id, "Setting", "Image", "RANSAC: Transformation", "ICP: Transformation"], axis="columns")
    analysis_std = analysis_std.groupby(groupby_col).std().reset_index()
    analysis_std = analysis_std.rename(columns={"RANSAC: Fitness": "RANSAC: STD Fitness",
                                                "ICP: Fitness": "ICP: STD Fitness",
                                                "RANSAC: Inlier RMSE": "RANSAC: STD Inlier RMSE",
                                                "ICP: Inlier RMSE": "ICP: STD Inlier RMSE",
                                                "RANSAC Iterations": "STD RANSAC Iterations",
                                                "ICP Iterations": "STD ICP Iterations"})

    # 3) Merge the means and std columns side by side
    #    (We use 'Scene' as the key to match rows)
    analysis = pd.merge(analysis_mean, analysis_std, on=groupby_col, how="left")

    # 4) Compute overall means (using per-scene means) and overall std (using all samples) 
    total = pd.DataFrame({groupby_col: "TOTAL",
                          "Mean RANSAC Iterations": results["RANSAC Iterations"][results["RANSAC Iterations"] != 0].mean(),
                          "STD RANSAC Iterations": results["RANSAC Iterations"].std(),
                          "Mean ICP Iterations": results["ICP Iterations"].mean(),
                          "STD ICP Iterations": results["ICP Iterations"].std(),
                          "RANSAC: Mean Fitness": results["RANSAC: Fitness"].mean(),
                          "RANSAC: STD Fitness": results["RANSAC: Fitness"].std(),
                          "ICP: Mean Fitness": results["ICP: Fitness"].mean(),
                          "ICP: STD Fitness": results["ICP: Fitness"].std(),
                          "RANSAC: Mean Inlier RMSE": results["RANSAC: Inlier RMSE"].mean(),
                          "RANSAC: STD Inlier RMSE": results["RANSAC: Inlier RMSE"].std(),
                          "ICP: Mean Inlier RMSE": results["ICP: Inlier RMSE"].mean(),
                          "ICP: STD Inlier RMSE": results["ICP: Inlier RMSE"].std()}, index=[0])

    # 5) Compute the inter-group std deviation using the per-group means
    inter_groups_std = pd.DataFrame({groupby_col: f"Inter-{groupby_col} STD",
                                     "RANSAC: STD Fitness": analysis["RANSAC: Mean Fitness"].std(),
                                     "ICP: STD Fitness": analysis["ICP: Mean Fitness"].std(),
                                     "RANSAC: STD Inlier RMSE": analysis["RANSAC: Mean Inlier RMSE"].std(),
                                     "ICP: STD Inlier RMSE": analysis["ICP: Mean Inlier RMSE"].std(),
                                     "STD RANSAC Iterations": analysis["Mean RANSAC Iterations"].std(),
                                     "STD ICP Iterations": analysis["Mean ICP Iterations"].std(),
                                     # Leave the "mean" columns of the inter-group std row blank
                                     "RANSAC: Mean Fitness": "",
                                     "ICP: Mean Fitness": "",
                                     "RANSAC: Mean Inlier RMSE": "",
                                     "ICP: Mean Inlier RMSE": "",
                                     "Mean RANSAC Iterations": "",
                                     "Mean ICP Iterations": ""}, index=[0])

    # Concatenate everything
    analysis = pd.concat([analysis, total, inter_groups_std], ignore_index=True)

    # Recreate identification columns
    analysis['Setting'] = 'Full Scene' if setting == 'full_scene' else 'Single Object'
    analysis['Pipeline'] = 'Geometric-Only' if pipeline == 'geometric_only' else 'Deep-Learning'

    # Reorder columns
    desired_order = ["Setting", "Pipeline", groupby_col, "Mean RANSAC Iterations", "STD RANSAC Iterations", "Mean ICP Iterations", "STD ICP Iterations",
                     "RANSAC: Mean Fitness", "RANSAC: STD Fitness", "ICP: Mean Fitness", "ICP: STD Fitness",
                     "RANSAC: Mean Inlier RMSE", "RANSAC: STD Inlier RMSE", "ICP: Mean Inlier RMSE", "ICP: STD Inlier RMSE"]
    analysis = analysis[desired_order]

    # Save the obtained registration results summary table as a .csv in the output folder
    output_folder = f'{output_dir}/{pipeline}-{setting}'
    filename = f"{output_folder}/local_evaluation_per_{'object' if groupby_col == 'Object' else 'scene'}.csv"
    analysis.to_csv(filename, index=False)
    print(f'Local evaluation table saved at: {filename}')

    return analysis


# 5.3.1 Results per Scene

GO_full_assess = assess_results(filt_GO_full, 'geometric_only', 'full_scene', output_dir)
GO_object_assess = assess_results(filt_GO_object, 'geometric_only', 'single_object', output_dir)
DL_full_assess = assess_results(filt_DL_full, 'deep_learning', 'full_scene', output_dir)
DL_object_assess = assess_results(filt_DL_object, 'deep_learning', 'single_object', output_dir)

total_GO_full = GO_full_assess[GO_full_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()
total_GO_object = GO_object_assess[GO_object_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()
total_DL_full = DL_full_assess[DL_full_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()
total_DL_object = DL_object_assess[DL_object_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()

total_comparison = pd.concat([total_GO_full, total_DL_full, total_GO_object, total_DL_object])

# Save the obtained registration results summary table as a .csv in the output folder
filename = f"{output_dir}/overall_evaluation_per_scene_totals.csv"
total_comparison.to_csv(filename, index=False)
print(f'Overall comparison table (totals per pipeline/setting combination) saved at: {filename}')

print("============================================== Comparison Table ==============================================")
print(total_comparison.to_string(index=False))
print("==============================================================================================================")


# 5.3.2 Results per Object

GO_full_assess_obj = assess_results(filt_GO_full, 'geometric_only', 'full_scene', output_dir, groupby_col="Object")
GO_object_assess_obj = assess_results(filt_GO_object, 'geometric_only', 'single_object', output_dir, groupby_col="Object")
DL_full_assess_obj = assess_results(filt_DL_full, 'deep_learning', 'full_scene', output_dir, groupby_col="Object")
DL_object_assess_obj = assess_results(filt_DL_object, 'deep_learning', 'single_object', output_dir, groupby_col="Object")


dfs = [GO_full, DL_full, GO_object, DL_object]
filt_dfs = [filt_GO_full, filt_DL_full, filt_GO_object, filt_DL_object]
labels = ['GO Full', 'DL Full', 'GO Single', 'DL Single']

obj_recall = pd.DataFrame({'Object': dfs[0]['Object'].unique()})

for i, (df, filt_df) in enumerate(zip(dfs, filt_dfs)):

    total = df[['Object', 'Image']].groupby('Object').agg(total=('Image', 'count')).reset_index()
    aligned = filt_df[['Object', 'Image']].groupby('Object').agg(aligned=('Image', 'count')).reset_index()

    merged = pd.merge(total, aligned, on='Object', how='left')
    merged[f'pct {labels[i]}'] = 100 * merged['aligned'] / merged['total']

    obj_recall = pd.merge(obj_recall, merged[['Object', f'pct {labels[i]}']], on='Object', how='left')

obj_recall[[f'pct {label}' for label in labels]] = obj_recall[[f'pct {label}' for label in labels]].fillna(0)
obj_recall['avg'] = obj_recall[[f'pct {label}' for label in labels]].mean(axis=1)
obj_recall = obj_recall.sort_values(by='avg', ascending=False).reset_index(drop=True).drop('avg', axis=1)

filename = f"{output_dir}/objects_recall_detailed.csv"
obj_recall.to_csv(filename, index=False)
print(f'Per-object recall results on different pipelines/settings table saved at: {filename}')

print("============================================== Per-Object Recall Results ==============================================")
print(obj_recall.to_string(index=False))
print("=======================================================================================================================")


# Build the combined per-object performance table of all settings/pipelines
dfs_obj_temp = [GO_full_assess_obj, GO_object_assess_obj, DL_full_assess_obj, DL_object_assess_obj]
dfs_obj_list = []
for df in dfs_obj_temp:
    d = df.copy()                                                                           # avoid changing originals
    d = d[(d['Object'] != 'TOTAL') & (~d['Object'].astype(str).str.contains('Inter-'))]     # remove total and inter-{} STD rows
    dfs_obj_list.append(d)
df_obj = pd.concat(dfs_obj_list, ignore_index=True)                                         # concatenate all rows

# Build Top/Bottom summary from the recall table
summary_rows = []

for pipeline in obj_recall.columns[1:]:

    # Sort descending by recall for this pipeline
    sorted_df = obj_recall.sort_values(by=pipeline, ascending=False).reset_index(drop=True)
    
    # Map pipeline column name to labels used in df_obj
    setting_label = 'Single Object' if 'Single' in pipeline else 'Full Scene'
    pipeline_label = 'Deep Learning' if 'DL' in pipeline else 'Geometric-Only'

    # Top 3
    for i in range(3):
        summary_rows.append({
            'Setting': setting_label,
            'Pipeline': pipeline_label,
            'Rank': f'Top {i+1}',
            'Object': sorted_df.loc[i, 'Object'],
            'Recall (%)': sorted_df.loc[i, pipeline]
        })
    
    # Bottom 3
    n = len(sorted_df)
    for i in range(3):
        summary_rows.append({
            'Setting': setting_label,
            'Pipeline': pipeline_label,
            'Rank': f'Bottom {i+1}',     # labeled 1..3; we’ll order them later
            'Object': sorted_df.loc[n-1-i, 'Object'],
            'Recall (%)': sorted_df.loc[n-1-i, pipeline]
        })

# Build DataFrame
summary_df = pd.DataFrame(summary_rows)

# Enforce the desired rank (row) order
rank_order = ['Top 1', 'Top 2', 'Top 3', 'Bottom 3', 'Bottom 2', 'Bottom 1']
summary_df['Rank'] = pd.Categorical(summary_df['Rank'], categories=rank_order, ordered=True)

# Final column order
summary_df = summary_df[['Setting', 'Pipeline', 'Rank', 'Object', 'Recall (%)']]

# Sort so each Setting/Pipeline block appears in the desired rank order
summary_df = summary_df.sort_values(['Setting', 'Pipeline', 'Rank']).reset_index(drop=True)

# Merge with all extra performance columns
perf_cols = [c for c in df_obj.columns if c not in ['Setting', 'Pipeline', 'Object']]
summary_df = summary_df.merge(
    df_obj[['Setting', 'Pipeline', 'Object'] + perf_cols],
    on=['Setting', 'Pipeline', 'Object'],
    how='left'
)

filename = f"{output_dir}/objects_recall_top_bottom.csv"
summary_df.to_csv(filename, index=False)
print(f'Top and bottom objects\' recall results summary table saved at: {filename}')

print("============================================== Top and Bottom Objects\' Recall Summary ==============================================")
print(summary_df.to_string(index=False))
print("=====================================================================================================================================")


y = max(len(obj_recall['Object'].unique())*0.2, 5)
x = max(y/3, 7)
plt.figure(figsize=(x, y))

sns.heatmap(obj_recall.set_index('Object'), cbar_kws={'label': 'Recall (%)', 'aspect': 100}, cmap='viridis_r', linewidths=0.5)

plt.title('Recall per Object per Pipeline-Setting')
plt.xticks(
    np.arange(len(labels))+0.5,
    ['Geometric-Only\nFull Scene', 'Deep Learning\nFull Scene',
    'Geometric-Only\nSingle Object', 'Deep Learning\nSingle Object'],
    rotation=0
)
plt.yticks(np.arange(len(obj_recall)) + 0.5, obj_recall['Object'], rotation=0)
plt.ylabel('Object ID')

plt.tight_layout()
plt.savefig(f'{output_dir}/recall_heatmap.png', dpi=300, bbox_inches='tight')
print(f'Recall heatmap saved at: {output_dir}/recall_heatmap.png')
# plt.show()


# 5.4 Visualizing Results (Not Applicable for the script case)
