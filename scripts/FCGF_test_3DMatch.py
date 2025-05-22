# 1 Setting Up the Environment

import MinkowskiEngine as ME
print(f'MinkowskiEngine version: {ME.__version__}')
import torch
print(f'PyTorch version: {torch.__version__}')

# After that, we just need to import everything we are going to use.

import os
import io
import sys
import copy
import math
import shutil
import subprocess
import numpy as np
import pandas as pd
import open3d as o3d
from urllib.request import urlretrieve
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Literal
from contextlib import redirect_stdout

# Get the absolute path of the source directory
sys.path.append(os.path.abspath("../source/FCGF"))


# 2 Input Data

test_path = '../data/FCGF/threedmatch_test'
if not os.path.exists(test_path):
  print(f'Downloading data at {test_path}\n\n=================================================================\n')
  subprocess.run(["bash", "../source/FCGF/scripts/download_3dmatch_test.sh", test_path], check=True)
else:
    print(f'The data is already available at {test_path}')


# train_path = '../data/FCGF/threedmatch_train/'
# if not os.path.exists(train_path):
#   print(f'Downloading data at {train_path}\n=================================================================')
#   !bash ../source/FCGF/scripts/download_datasets.sh {train_path}
# else:
#     print(f'The data is already available at {train_path}')

# Check if the weight folder has already been created, otherwise creates it
fcgf_weights_folder = '../weigths/FCGF'
if not os.path.exists(fcgf_weights_folder):
    os.makedirs(fcgf_weights_folder)

# Check if the selected pre-trained weights were already downloaded, otherwise download them
fcgf_weight = 'ResUNetBN2C-16feat-3conv.pth'
fcgf_weight_path = os.path.join(fcgf_weights_folder, fcgf_weight)
if not os.path.isfile(fcgf_weight_path):
  print(f'Downloading weight at {fcgf_weight_path}...')
  urlretrieve("https://node1.chrischoy.org/data/publications/fcgf/2019-09-18_14-15-59.pth",
              fcgf_weight_path)
else:
    print(f'Selected weights already available at {fcgf_weight_path}')


# 3 Local Refinement
## 3.1 Retrieving Transformations from Logs

def get_transformation_from_content(content: list[str], line_idx: int) -> np.ndarray:
    """Extract the 4×4 transformation matrix from pre-loaded log lines.

    The log is organized in 5-line blocks per pair:
      1) Header: "<tgt_ID> <src_ID> <nFrags>"
      2–5) Four rows of the 4×4 matrix.

    Args:
        content (list[str]): All lines of the log file, as returned by `f.readlines()`.
        line_idx (int): Index of the header line in `content`.

    Returns:
        np.ndarray: A (4,4) float array containing the transformation matrix.
    """

    # List to accumulate the next four rows of the matrix as lists of floats
    transformation = []

    # Iterate 4 times (since the matrix is 4x4 -> 4 rows
    for i in  range(4):

        # Update the current row of the matrix
        line_idx += 1
        
        # Extract, cleans newline/tab characters and split the line into tokens
        line = content[line_idx].strip().split()
        
        # Convert each element to float and append as one row
        transformation.append([float(value) for value in line])

    return np.array(transformation)


## 3.2 Adding an ICP Stage

def execute_ICPrefinement(source, target, inlier_th, trans_init, voxel_size):
    """Executes the local ICP refinement for an initial transformation.

    Args:
        source (open3d.geometry.PointCloud): Source cloud
        target (open3d.geometry.PointCloud): Target cloud
        inlier_th (float): Threshold distance for a correspondence pair to be considered valid
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


def write_refined_log(icp_log_path, tgt_ID, src_ID, nFrags, transformation):
    """Append a refined registration result (ICP output) to the scene’s log file.

    Parameters
    ----------
    icp_log_path : str
        Path for the refined log file that will store the icp results.
    tgt_ID : int
        Target fragment ID.
    src_ID : int
        Source fragment ID.
    nFrags : int
        Number of fragments used in the initial guess.
    transformation : np.ndarray, shape (4,4)
        4×4 homogeneous transformation matrix from ICP refinement.
    """
        
    # Use mode 'a' to append new entries rather than overwrite existing ones
    with open(icp_log_path, 'a') as f:

        # First line: IDs and fragment count
        f.write(f'{tgt_ID} {src_ID} {nFrags}\n')

        # Next 4 lines: the rows of the 4×4 transformation matrix
        for row in transformation:
            # Format each value to 12 decimal places with a space between them
            line = " ".join(f"{val:.12f}" for val in row)
            f.write(f'{line}\n')


def ICP_stage(output_folder, test_path, inlier_th, voxel_size):
    """
    Run a full ICP pipeline over all initial guesses logs and collect results.

    For each scene, reads the initial-guess logs, performs:
      1. RANSAC evaluation on full-resolution clouds.
      2. ICP refinement starting from the RANSAC transformation.
      3. Logs the refined transformation.
      4. Appends metrics to a list of results which is returned as a DataFrame.

    Parameters
    ----------
    output_folder : str
        Base output folder containing 'registration/initial_guesses_logs'.
    test_path : str
        Root folder where point-cloud scene subdirectories reside.
    inlier_th : float
        Distance threshold for inlier determination in evaluation.
    voxel_size : float
        Downsampling voxel size used in ICP refinement.

    Returns
    -------
    pd.DataFrame
        Table of metrics and transformations for each source–target pair.
    """    
    
    results = []

    # Directory containing one log per scene of initial FCGF guesses (RANSAC)
    ransac_logs_folder = f'{output_folder}/registration/initial_guesses_logs'

    # Define the refined logs directory
    icp_logs_folder = f"{output_folder}/registration/logs"

    # If the folder already exists, deletes it to remove the old entries
    # otherwise it would append the entries of the new run on top of the old ones
    if os.path.exists(icp_logs_folder):    
        shutil.rmtree(icp_logs_folder)     
    
    # Creates a new folder
    os.makedirs(icp_logs_folder)

    # Iterate over each log file in the folder
    for log in os.listdir(ransac_logs_folder):
        
        # Extract scene name (before the '_FCGF' suffix in filename)
        scene = log.split('_FCGF')[0]            
        print(f'Set: {scene}')

        # Define the ransac log path of a given scene
        ransac_log_path = f'{ransac_logs_folder}/{log}'

        # Define the current scene's refined log path
        icp_log_path = f'{icp_logs_folder}/{scene}_FCGF.log'

        with open(ransac_log_path, 'r') as f:

            # Read all lines once
            content = f.readlines()
            n = len(content)

            # Each entry consists of 5 lines: one header + 4 rows of matrix
            # so the step is set to 5 to read only the headers
            for i in range(0, n, 5):

                # Access the line, clean and split it
                line = content[i]
                line = line.strip().split()

                # Retrieve the pair information
                tgt_ID = int(line[0])
                src_ID = int(line[1])
                nFrags = int(line[2])

                # Load the related point clouds
                source = o3d.io.read_point_cloud(os.path.join(test_path, scene, 'cloud_bin_%s.ply' %src_ID))
                target = o3d.io.read_point_cloud(os.path.join(test_path, scene, 'cloud_bin_%s.ply' %tgt_ID))
                
                # Retrieves the RANSAC transformation of the pair
                ransac_tran = get_transformation_from_content(content, i)
                
                # Evaluate RANSAC on the full clouds
                # (if we use ransac_reg.fitness or ransac_reg.inlier_rmse, these
                #  results would be computed on the downsampled clouds used by RANSAC)
                ransac_eval = o3d.pipelines.registration.evaluate_registration(source, target, inlier_th, ransac_tran)

                print(f'\tMatching {tgt_ID:03d} {src_ID:03d}')

                # Set Open3D's verbosity level to Debug to capture detailed iteration information
                o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

                # Refine the RANSAC guess with ICP
                icp_reg = execute_ICPrefinement(source, target, inlier_th, ransac_tran, voxel_size)

                # Returns Open3D's verbosity level to default mode
                o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

                # Append the refined transformation to the log
                write_refined_log(icp_log_path, tgt_ID, src_ID, nFrags, icp_reg.transformation)

                # Collect all information for the results table
                new_result = {'Scene': scene,
                              'Target': tgt_ID,
                              'Source': src_ID,
                              'RANSAC: Fitness': ransac_eval.fitness,
                              'ICP: Fitness': icp_reg.fitness,
                              'RANSAC: Inlier RMSE': ransac_eval.inlier_rmse,
                              'ICP: Inlier RMSE': icp_reg.inlier_rmse,
                              'Initial Guess': ransac_tran, #wrapper on matrix -> check notebok 3-ICP_Pipeline_Datasets for details
                              'Transformation': icp_reg.transformation
                }
                results.append(new_result)

                print('\tDone')

        # Print the refined log path for that scene
        print(f'\tLogging at:" {icp_log_path}')

    # Convert the collected results to a pandas DataFrame
    results_table = pd.DataFrame(results)
    return results_table



# 4 Iteration Counter

def get_iterations(captured_output, stage: Literal['RANSAC', 'ICP']):
    """
    Parses the captured output obtained in bedug mode to extract
    the iteration counts for each alignment.

    Parameters
    ----------
    captured_output : str
        Vriable storing the captured output
    stage : str
        Specifies from which stage the iterations should be extracted. Must be either 'RANSAC' or 'ICP'.

    Returns
    -------
    pd.DataFrame
        Table with the iteration count for the given stage.
    """   

    if stage not in ('RANSAC', 'ICP'):
        raise ValueError(f"Invalid stage '{stage}'. Must be 'RANSAC' or 'ICP'.")
        print(f"Using stage: {stage}")

    # List to store iteration data for each alignment
    iterations = []

    # Clean the captured output: remove tabs, extra spaces, and split it into individual lines.
    captured_output = captured_output.replace('[Open3D DEBUG]', '').replace('\t', '').strip().split('\n')

    # Loops through the output lines
    for line in captured_output:

        # Split the line into words
        line = line.split()
        if not line:  # Skip empty lines
            continue

        # Check the case (and length to avoid index errors)
        if stage == 'RANSAC' and len(line) >= 3:

            # Get current scene
            if line[2] == 'Set:':           # checks if it starts with 'Set:'
                cur_scene = line[3]   

            # Get currrent matching pair
            elif line[2] == 'Matching':     # Checks if it starts with Matching
                target_ID = int(line[3])    # Obtain the target ID from the second field
                source_ID = int(line[4])    # Obtain the target ID from the third field
                ICP_iterations = 0   
                
            # Get ransac iterations
            elif line[0] == 'RANSAC':
                RANSAC_iterations = int(line[3])
            
            # When the line starts with 'Overlap', it signals the end of the current alignment
            # If the alignment was achieved (30% overlap), store the results
            elif line[2] == 'Overlap' and float(line[4]) > 0.3: 
                
                # Stores the iterations of the current pair
                # Appending dictionaries is to a list is a straightforward way of build DataFrames later
                # Each dictionary in the list is a row of the table and the columns are the dictionaries keys
                iterations.append({
                    'Scene': cur_scene,
                    'Target': target_ID,
                    'Source': source_ID,
                    'RANSAC Iterations': RANSAC_iterations
                })

        # When dealing with the refinement stage
        elif stage == 'ICP':
                
            # Get current scene
            if line[0] == 'Set:':           # checks if it starts with 'Set:'
                cur_scene = line[1]         # extracts the scene name from the following text in the line

            # Get currrent matching pair
            elif line[0] == 'Matching':     # Checks if it starts with Matching
                target_ID = int(line[1])    # Obtain the target ID from the second field
                source_ID = int(line[2])    # Obtain the target ID from the third field
                ICP_iterations = 0          # Resets the ICP iterations counter for the next pair

            # If the line indicates an ICP iteration, increment the ICP iterations counter
            elif line[0] == 'ICP':
                ICP_iterations += 1

            # When the alignment is completed, store it in the list
            elif line[0] == 'Done':
                iterations.append({
                        'Scene': cur_scene,
                        'Target': target_ID,
                        'Source': source_ID,
                        'ICP Iterations': ICP_iterations
                    })
                # print(f'{cur_scene} {target_ID} {source_ID}')
                # print('--------------------------------------')

    iterations_df = pd.DataFrame(iterations)
    return iterations_df


def augment_results(results, ransac_captured_output, icp_captured_output):
    """Augments the main results DataFrame with RANSAC and ICP iteration counts.

    This function merges the provided `results` DataFrame with iteration count DataFrames
    extracted from captured console outputs of the RANSAC and ICP stages. The merging is 
    done as a left join on the 'Scene', 'Target', and 'Source' columns. The resulting 
    DataFrame is then reordered to place the iteration columns in more intuitive positions.

    Args:
        results (pd.DataFrame): The main DataFrame containing alignment results.
        ransac_captured_output (str): Captured console output from the RANSAC stage.
        icp_captured_output (str): Captured console output from the ICP stage.

    Returns:
        pd.DataFrame: The augmented DataFrame containing iteration counts for RANSAC and ICP.
    """
    
    # Extract iteration counts from the captured outputs
    ransac_iter_df = get_iterations(ransac_captured_output, stage='RANSAC')
    icp_iter_df = get_iterations(icp_captured_output, stage='ICP')
        
    # Merge the iteration counts into the results DataFrame
    augmented_results = results.merge(ransac_iter_df, 'left', on=['Scene', 'Target', 'Source'])
    augmented_results = augmented_results.merge(icp_iter_df, 'left', on=['Scene', 'Target', 'Source'])

    # Reoders the columns
    col_names = list(augmented_results.columns.values)  # get labels
    col_names.insert(3, col_names[-2])                  # insert 2nd to last item (RANSAC Iterations) at index 3
    col_names.pop(-2)                                   # removes 2nd to last item
    col_names.insert(4, col_names[-1])                  # insert last item (ICP Iterations) at index 4
    col_names.pop()                                     # removes last item
    augmented_results = augmented_results[col_names]    # reorder df with the same order of the col_names list
    
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

# 5 Defining the Complete Pipeline

def check_feasible_subset(subset: int) -> None:
    """
     Checks if `subset` is a valid number of pairs to be considered.
     Raises ValueError if it is not. Otherwise returns None and the script continues.
     """
   
    # Check for negative values
    if subset < 0:
        raise ValueError("subset must be non-negative")

    # Compute delta
    delta = 1 + 8*subset

    # Perform the checks described previously
    root = math.isqrt(delta)                        # .isqrt() returns the nearest smaller integer of the sqrt 
    if root*root == delta and (1+root) % 2 == 0:
        return
    
    # If subset is not valid, raise error and display nearest feasible value
    else:
        # Compute the exact float N obtained from the quadratic formula
        N = (1+root)/2
        
        # By forcing it to int, we are truncating the decimals and rounding it down
        N_down = int(N)

        #Therefore, the upper bound is simply the lower plus 1
        N_up = N_down + 1  

        # Compute the bounds of feasible subset values (use // to once again avoid FP errors)
        P_lower = N_down*(N_down-1)//2
        P_upper= N_up*(N_up-1)//2

        # raise error 
        raise ValueError(
            f"subset={subset} is not a valid triangular number.\n"
            f"--> Nearest smaller feasible subset: {P_lower}\n"
            f"--> Nearest larger  feasible subset: {P_upper}"
        )

def run_benchmark(test_path, feature_path, voxel_size, inlier_th, subset, model, stdout=None):
    """
    Invoke the 3DMatch benchmark script via subprocess, streaming its stdout
    to both the real console and an optional buffer.
    """
    # Build the command-line invocation
    args = [
        sys.executable,                                # use same Python interpreter
        "../source/FCGF/scripts/benchmark_3dmatch.py", # path to the benchmark driver
        "--source", test_path,                         # input scenes directory
        "--target", feature_path,                      # features output directory
        "--voxel_size", str(voxel_size),               # downsampling parameter
        "--inlier_th", str(inlier_th),                 # inlier threshold
        "--model", model,                              # FCGF model weigths
        "--extract_features",                          # first stage: extract features
        "--evaluate_feature_match_recall",             # compute match recall
        "--evaluate_registration",                     # run geometric registration
    ]
    
    # If running a subset of pairs, add the flag
    if subset:
        args += ["--subset", str(subset)]
    
    # Launch the child process with its stdout piped back to us
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,  # create a real OS pipe for stdout
        stderr=subprocess.DEVNULL,  # ignore everything on stderr
        text=True,               # decode output as text (not bytes)
        bufsize=1                # line-buffered mode for timely output
    )


    # Read each line from the child’s stdout as it arrives
    for line in proc.stdout:
        # Echo to the real console immediately
        sys.__stdout__.write(line)
        # Also store into our provided buffer, if any
        if stdout is not None:
            stdout.write(line)

    # Close our reading end, wait for the process to exit
    proc.stdout.close()
    ret = proc.wait()

    # If the child exited with a non-zero code, raise an exception
    if ret != 0:
        raise subprocess.CalledProcessError(ret, args)

def execute_FCGF_Pipeline(voxel_size, inlier_th, subset, model, test_path, run_name):
    """Run the full FCGF pipeline: feature extraction, RANSAC, and ICP.

    Args:
        voxel_size (float): Downsampling size for point clouds.
        inlier_th (float): Inlier distance threshold.
        subset (int or None): Number of pairs to process; False for all.
        model (str): FCGF model path or identifier.
        test_path (str): Directory of test scenes.
        run_name (str): Base name for output folder.

    Returns:
        (output_folder: str, results: pd.DataFrame)
    """

    if subset:
       check_feasible_subset(subset)

    time = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime('%Y-%m-%d_%H-%M-%S')
    output_folder = f"../output/FCGF/{run_name}-{time}"
    feature_path = f"{output_folder}/features"

    print("Applying FCGF to TEST split using:\n--> voxel_size=%f\n--> distance threshold=%f" %(voxel_size, inlier_th))
    print('======================================================')

    # For RANSAC, since it is called from a script, we don't need to use a Tee object just a simple StringIO
    ransac_tee_buffer = io.StringIO()

    # No redirect_stdout needed: run_benchmark streams directly to sys.__stdout__ and into the Tee buffer
    # Set Open3D's verbosity level to Debug to capture detailed iteration information
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    # Execute FCGF with only the RANSAC stage
    run_benchmark(test_path, feature_path, voxel_size, inlier_th, subset, model, ransac_tee_buffer)
    # Returns Open3D's verbosity level to default mode
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    # Retrieve the captured output as a string
    ransac_captured_output = ransac_tee_buffer.getvalue()

    # Create thee Tee object to capture outputs from ICP stage while printing to the console
    icp_tee_buffer = Tee()

    # Redirect stdout to the Tee object during ICP execution
    with redirect_stdout(icp_tee_buffer):
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        results = ICP_stage(output_folder, test_path, inlier_th, voxel_size)
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    icp_captured_output = icp_tee_buffer.getvalue()

    # Merge the iteration information into the main results DataFrame
    results = augment_results(results, ransac_captured_output, icp_captured_output)

    # save the obtained table as a .csv in the output folder
    filename = f"{output_folder}/registration/registration_table_FCGF.csv"
    results.to_csv(filename, index=False)
    print('-------------------------------------------------------------------------------------------------------')
    print(f'Registration results table saved at: {filename}')

    return output_folder, results

# 6 Testing


subset = False
voxel_size = 0.025
inlier_th = 0.05            # 5 cm --> this must be the same for ICP
model = fcgf_weight_path

run_name = "FCGF_test_complete_wIterations"

output_folder, results = execute_FCGF_Pipeline(voxel_size, inlier_th, subset, model, test_path, run_name)
print("============================================== Registration Table ==============================================")
print(results.to_string(index=False))
print("================================================================================================================")



## 6.1 Quantitative Analysis

def get_matching_pairs(file):
    """
    Parse a matching-pairs text file and return a mapping of scene names to lists of matching ID pairs.

    Args:
        file (str): Path to the matching_pairs.txt file. Each block starts with a line `Set: <scene_name>`
                    followed by lines of `<tgt_ID> <src_ID>` pairs.

    Returns:
        dict[str, list[list[int]]]: A dictionary where keys are scene names and values are lists of
                                    [target_ID, source_ID] pairs.
    """

    # Initialize scene variable and dictionary that will store the collected info
    scene = None
    matching_pairs = defaultdict(list)
    
    # Read the matching_pairs.txt file
    with open(file, 'r') as f:
        # Iterate through each line
        for line in f:
            
            # Clean and split the line
            line = line.strip().split()

            # If the first word is 'Set:', it indicates the start of a block for a different scene
            if line[0] == 'Set:':
                scene = line[1]     # Update the current scene
                continue

            # If it is not a new scene, we are still in the same block
            elif scene:             # Check whether we truly are in a block
                # Collect the pairs for that scene
                matching_pairs[scene].append([int(line[0]), int(line[1])])
 
    return matching_pairs


def get_scene_reducedGT(log_path, info_path, scene_out_gt_path, matching_pairs, scene):
    """
    Filter and write reduced ground-truth files (gt.log and gt.info) for a single scene,
    including only specified matching pairs.

    Args:
        log_path (str): Path to the original gt.log file for this scene.
        info_path (str): Path to the original gt.info file for this scene.
        scene_out_gt_path (str): Directory where reduced files will be written.
        matching_pairs (dict[str, list[list[int]]]): Mapping of scene names to matching pairs.
        scene (str): Name of the scene to process (must be a key in matching_pairs).

    Returns:
        None
    """

    # Ensure output directory exists
    os.makedirs(scene_out_gt_path, exist_ok=True)

    # Output file paths
    out_log_path = os.path.join(scene_out_gt_path, 'gt.log')
    out_info_path = os.path.join(scene_out_gt_path, 'gt.info')

    # Compute number of fragments from pair count: N = (1 + sqrt(1+8*P)) / 2
    n_pairs = len(matching_pairs[scene])
    num_frag = int((1 + np.sqrt(1 + 8*n_pairs))/2)

    # Auxiliary flag to determine whether to copy the info of a pair
    copy = False

    # Open the complete gt log file in reading mode  
    with open(log_path, 'r') as i:
        # And the reduced gt log file in writing mode
        with open(out_log_path, 'w') as o:
            # Iterate through the lines of the complete file
            for idx, line in enumerate(i):

                # Clean and split the line
                line = line.strip().split()

                # Every 5 lines, we have a different pair info
                if idx%5 == 0:
                    # If the pair is included in the subset of matching pairs to be considered
                    if [int(line[0]), int(line[1])] in matching_pairs[scene]:
                        # Write <target_ID> <source_ID> <num_frag> and set flag to copy its block
                        o.write(f"{line[0]}\t {line[1]}\t {num_frag}\n")
                        copy = True
                    # Otherwise, reset copy flag
                    else:
                        copy = False
                # If it's not a header line and the flag is set, copy the info (it's a row of the matrix)
                elif copy:
                    o.write(f"{line[0]}\t {line[1]}\t {line[2]}\t {line[3]}\n")

    # Do the same procedure for the gt info file
    with open(info_path, 'r') as i:
        with open(out_info_path, 'w') as o:
            for idx, line in enumerate(i):
                line = line.strip().split()
                if idx%7 == 0:
                    if [int(line[0]), int(line[1])] in matching_pairs[scene]:
                        o.write(f"{line[0]}\t {line[1]}\t {num_frag}\n")
                        copy = True
                    else:
                        copy = False
                elif copy:
                    o.write(f"{line[0]}\t {line[1]}\t {line[2]}\t {line[3]}\t {line[4]}\t {line[5]}\n")


def get_reducedGT(output_folder, test_path):
    """
    Generate reduced ground-truth files for all scenes in test_path, based on matching pairs.

    This function reads the matching_pairs.txt in the registration folder of output_folder,
    then for each scene directory ending with '-evaluation' under test_path, it filters the
    gt.log and gt.info files to only include the pairs listed and writes them under
    output_folder/groundtruth/<scene>.

    Args:
        output_folder (str): Base output directory containing 'registration/matching_pairs.txt'.
        test_path (str): Directory containing scene subdirectories with '-evaluation' suffix.

    Returns:
        None
    """

    # Create groundtruth output directory
    out_gt_path = f"{output_folder}/groundtruth"
    os.makedirs(out_gt_path, exist_ok=True)

    # Obtain matching pairs to be considered
    matching_pairs = get_matching_pairs(f"{output_folder}/registration/matching_pairs.txt")

    # Iterate over scene directories
    for filename in os.listdir(test_path):
        
        # Check if the file name ends with 'evaluation'
        aux = filename.split('-')
        if aux[-1] == 'evaluation':

            # Store the path to the complete files in the evaluation folder
            log_path = os.path.join(test_path, filename, 'gt.log')
            info_path = os.path.join(test_path, filename, 'gt.info')

            # Define the path for the reduced groundtruth files
            scene_out_gt_path = os.path.join(out_gt_path, filename)
            
            # Obtain the scene name
            scene = '-'.join(aux[:-1])

            # Generate the reduuced files
            get_scene_reducedGT(log_path, info_path, scene_out_gt_path, matching_pairs, scene)
            
    print(f'Reduced ground truth files saved at: {out_gt_path}')

get_reducedGT(output_folder, test_path)


eval_csv_path = f"{output_folder}/evaluation/registration_evaluation_FCGF.csv"
if os.path.isfile(eval_csv_path):
    reg_eval_FCGF = pd.read_csv(eval_csv_path)
    print("============================================== Evaluation Table ==============================================")
    print(reg_eval_FCGF.to_string(index=False))
    print("============================================================================================================")
else:
    print(f'No file found at: {eval_csv_path}\nPlease insert the .csv file with the obtained evaluation results in the expected path with the expected file name')


def assess_results(results):
    """
    Given a registration result table, computes the mean performance (and standard deviation) of the
    alignment for all clouds of a specific scene and for that whole split we selected from a dataset.

    Args:
        results (pd.DataFrame): Table containing the registration results of the split

    Returns:
        pd.DataFrame: Table with the overall (mean and standard) performance results
    """

    # 1) Compute per-scene mean
    #    (Dropping 'Source','Target','Transformation' from the grouping)
    analysis_mean = results.copy()                                                                              # create a copy of results
    analysis_mean = analysis_mean.drop(["Source", "Target", "Initial Guess", "Transformation"], axis="columns")  # remove unnecessary columns
    analysis_mean = analysis_mean.groupby("Scene").mean().reset_index()                                         #  compute the averages of each scene
    analysis_mean = analysis_mean.rename(columns={"RANSAC: Fitness": "RANSAC: Mean Fitness",
                                                  "ICP: Fitness": "ICP: Mean Fitness",
                                                  "RANSAC: Inlier RMSE": "RANSAC: Mean Inlier RMSE",
                                                  "ICP: Inlier RMSE": "ICP: Mean Inlier RMSE",
                                                  "RANSAC Iterations": "Mean RANSAC Iterations",
                                                  "ICP Iterations": "Mean ICP Iterations"})

    # 2) Compute per-scene standard deviation
    analysis_std = results.copy()
    analysis_std = analysis_std.drop(["Source", "Target", "Initial Guess", "Transformation"], axis="columns")
    analysis_std = analysis_std.groupby("Scene").std().reset_index()
    analysis_std = analysis_std.rename(columns={"RANSAC: Fitness": "RANSAC: STD Fitness",
                                                "ICP: Fitness": "ICP: STD Fitness",
                                                "RANSAC: Inlier RMSE": "RANSAC: STD Inlier RMSE",
                                                "ICP: Inlier RMSE": "ICP: STD Inlier RMSE",
                                                "RANSAC Iterations": "STD RANSAC Iterations",
                                                "ICP Iterations": "STD ICP Iterations"})

    # 3) Merge the means and std columns side by side
    #    (We use 'Scene' as the key to match rows)
    analysis = pd.merge(analysis_mean, analysis_std, on="Scene", how="left")

    # 4) Compute overall means (using per-scene means) and overall std (using all samples) 
    total = pd.DataFrame({"Scene": "TOTAL",
                          "Mean RANSAC Iterations": analysis_mean["Mean RANSAC Iterations"].mean(),
                          "STD RANSAC Iterations": [results["RANSAC Iterations"].std()],
                          "Mean ICP Iterations": analysis_mean["Mean ICP Iterations"].mean(),
                          "STD ICP Iterations": [results["ICP Iterations"].std()],
                          "RANSAC: Mean Fitness": analysis_mean["RANSAC: Mean Fitness"].mean(),
                          "RANSAC: STD Fitness": [results["RANSAC: Fitness"].std()],
                          "ICP: Mean Fitness": analysis_mean["ICP: Mean Fitness"].mean(),
                          "ICP: STD Fitness": [results["ICP: Fitness"].std()],
                          "RANSAC: Mean Inlier RMSE": analysis_mean["RANSAC: Mean Inlier RMSE"].mean(),
                          "RANSAC: STD Inlier RMSE": [results["RANSAC: Inlier RMSE"].std()],
                          "ICP: Mean Inlier RMSE": analysis_mean["ICP: Mean Inlier RMSE"].mean(),
                          "ICP: STD Inlier RMSE": [results["ICP: Inlier RMSE"].std()]}, index=[0])

    # 5) Compute the inter_scene std deviation using the per-scene means
    inter_scenes_std = pd.DataFrame({"Scene": "Inter-Scene STD",
                                     "RANSAC: STD Fitness": [analysis["RANSAC: Mean Fitness"].std()],
                                     "ICP: STD Fitness": [analysis["ICP: Mean Fitness"].std()],
                                     "RANSAC: STD Inlier RMSE": [analysis["RANSAC: Mean Inlier RMSE"].std()],
                                     "ICP: STD Inlier RMSE": [analysis["ICP: Mean Inlier RMSE"].std()],
                                     "STD RANSAC Iterations": [analysis["Mean RANSAC Iterations"].std()],
                                     "STD ICP Iterations": [analysis["Mean ICP Iterations"].std()],
                                     # Leave the "mean" columns of the inter-scene std row blank
                                     "RANSAC: Mean Fitness": "",
                                     "ICP: Mean Fitness": "",
                                     "RANSAC: Mean Inlier RMSE": "",
                                     "ICP: Mean Inlier RMSE": "",
                                     "Mean RANSAC Iterations": "",
                                     "Mean ICP Iterations": ""}, index=[0])

    # Concatenate everything
    analysis = pd.concat([analysis, total, inter_scenes_std], ignore_index=True)

    # Reorder columns
    desired_order = ["Scene", "Mean RANSAC Iterations", "STD RANSAC Iterations", "Mean ICP Iterations", "STD ICP Iterations",
                     "RANSAC: Mean Fitness", "RANSAC: STD Fitness", "ICP: Mean Fitness", "ICP: STD Fitness",
                     "RANSAC: Mean Inlier RMSE", "RANSAC: STD Inlier RMSE", "ICP: Mean Inlier RMSE", "ICP: STD Inlier RMSE"]
    analysis = analysis[desired_order]

    # Ensures the evaluation folder is present
    os.makedirs(f"{output_folder}/evaluation", exist_ok=True)
    
    # Save the obtained registration results summary table as a .csv in the output folder
    filename = f"{output_folder}/evaluation/mean_fitness_and_RMSE_table_FCGF.csv"
    analysis.to_csv(filename, index=False)
    print(f'Registration results summary table saved at: {filename}')

    return analysis


analysis = assess_results(results)
print("============================================== Analysis Table ==============================================")
print(analysis.to_string(index=False))
print("============================================================================================================")
