# 1 Setting Up the Environment

from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import open3d as o3d
import numpy as np
import pandas as pd
import urllib.request
import zipfile
import copy
import os
import io
import sys
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo
from contextlib import redirect_stdout



# 2 Input Data

# Define folder and file path
download_folder = "../data"
file_name = "3dmatch.zip"
file_url = "https://share.phys.ethz.ch/~gsg/pairwise_reg/3dmatch.zip"
file_path = os.path.join(download_folder, file_name)

# Create folder if it doesn't exist
os.makedirs(download_folder, exist_ok=True)

# Download the file if it doesn't exist
if not os.path.exists(file_path):
    print(f"Downloading {file_name}...")
    urllib.request.urlretrieve(file_url, file_path)
    print("Download completed.")
else:
    print(f"{file_name} already exists, skipping download.")


# Unzip the file if not already extracted
temp_extract_path = os.path.join(download_folder, os.path.splitext(file_name)[0])  # "3dmatch"
final_extract_path = os.path.join(download_folder, "3DMatch")  # Renamed folder

# Extract only if "3DMatch" does not exist
if not os.path.exists(final_extract_path):
    print(f"Extracting files to {final_extract_path}...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(download_folder)             # Extract
    os.rename(temp_extract_path, final_extract_path)    # Rename
    print("Extraction complete.")

    # Create validation folder
    validation_path = os.path.join(final_extract_path, "validation")
    train_path = os.path.join(final_extract_path, "train")
    os.makedirs(validation_path, exist_ok=True)

    # Move folders from "train" to "validation"
    txt_path = "../configs/3dmatch_val.txt"
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            folders_to_move = [line.strip() for line in f.readlines()]   # .strip() removes space and \n

        for folder in folders_to_move:
            src_folder = os.path.join(train_path, folder)
            dest_folder = os.path.join(validation_path, folder)
            if os.path.exists(src_folder):
                shutil.move(src_folder, dest_folder)
                print(f"Moved {folder} to validation folder.")
            else:
                print(f"Warning: {folder} not found in train folder.")
    else:
        print(f"Warning: {txt_path} not found. No folders were move")
              
else:
    print(f"Folder '{final_extract_path}' already exists, skipping extraction.")

print("Dataset is ready!")



data = {}                                               #HashMap of scenes for each dataset
data_root = "../data" 
datasets = ['3DMatch']      #os.listdir(data_root)                        #list of all datasets

for dataset in datasets:                                #for each dataset
    dataset_dir = os.path.join(data_root, dataset)      #get its directory
    scenes = {}                                         #initialize a hash map for its scenes

    for split in os.listdir(dataset_dir):               #for each split (train, test, val)
        split_dir = os.path.join(dataset_dir, split)    #get directory
        scenes[split] = os.listdir(split_dir)           #saves all of its scenes

    data[dataset] = scenes                              #save all splits for the dataset

print(data)



def get_split(data):
    """
    Prints the data split, i.e. the amount of scenes contained in
    each split (train, validation and test) of a given dataset.

    Args:
        data (dictionary): hash map of the data structured in the following way:
        {dataset: {train: [scene_list], test: [scene_list], validation: [scene_list]},
         dataset: {...},
         ...}

    Returns:
        None
    """

    for dataset in data.keys():
        print("========= %s =========" %dataset)
        print("Amount of scenes for each split:")
        print("-> Train: %d" %len(data[dataset]['train']))
        print("-> Test: %d" %len(data[dataset]['test']))
        print("-> Validation: %d" %len(data[dataset]['validation']))

get_split(data)



# 3 ICP Pipeline Implementation (Global Registration — FPFH + RANSAC — and Local Refinement — ICP)

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



def execute_GlobalRegistration(source_down, target_down, source_fpfh, target_fpfh, inlier_th, voxel_size):
    """
    Executes the Global Registration (through RANSAC algorithm) of
    the input source cloud (after being downsampled), given its
    FPFH features, in order to align it to the target cloud.

    Args:
        source_down (open3d.geometry.PointCloud): Downsampled source cloud
        target_down (open3d.geometry.PointCloud): Downsampled target cloud
        source_fpfh (open3d.registration.Feature): FPFH features of source cloud
        target_fpfh (open3d.registration.Feature): FPFH features of target cloud
        voxel_size (float): Resulting size of voxels after downsampling

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
    result_ransac = execute_GlobalRegistration(source_down, target_down, source_fpfh, target_fpfh, inlier_th, voxel_size)

    #Local Refinement (ICP)
    trans_init = result_ransac.transformation
    result_icp = execute_ICPrefinement(source, target, inlier_th, trans_init, voxel_size)

    return result_ransac, result_icp


def get_matching_indices(source, target, trans, search_voxel_size, K=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    source_copy.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)

    match_inds = []
    for i, point in enumerate(source_copy.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds


def compute_overlap_ratio(pcd0, pcd1, trans, voxel_size):
    pcd0_down = pcd0.voxel_down_sample(voxel_size)
    pcd1_down = pcd1.voxel_down_sample(voxel_size)
    matching01 = get_matching_indices(pcd0_down, pcd1_down, trans, voxel_size, 1)
    matching10 = get_matching_indices(pcd1_down, pcd0_down, np.linalg.inv(trans),
                                    voxel_size, 1)
    overlap0 = len(matching01) / len(pcd0_down.points)
    overlap1 = len(matching10) / len(pcd1_down.points)
    return max(overlap0, overlap1)



def ICP_pipeline_scene(frag_folder, voxel_size, inlier_th, matching_pairs):
    """
    Executes the complete ICP pipeline for a whole scene.

    Args:
        scene_dir (string): Directory of the scene
        voxel_size (float): Resulting size of voxels after downsampling
        dist_threhsold_ICP (float): Threshold distance for a correspondence
                                     pair to be considered valid during ICP

    Returns:
        pd.DataFrame: Pandas DataFrame with the registration results of the scene
        list: Array where each element is a point cloud of the scene
    """

    #table to save registrations results of each scene
    scene_results = pd.DataFrame({
    "Scene": pd.Series(dtype='str'),
    "Target": pd.Series(dtype='int'),
    "Source": pd.Series(dtype='int'),
    "RANSAC: Fitness": pd.Series(dtype='float'),
    "ICP: Fitness": pd.Series(dtype='float'),
    "RANSAC: Inlier RMSE": pd.Series(dtype='float'),
    "ICP: Inlier RMSE": pd.Series(dtype='float'),
    "Initial Guess": pd.Series(dtype='object'),
    "Transformation": pd.Series(dtype='object')
    })

    scene = frag_folder.split('/')[-2]

    for pair in matching_pairs[scene]:

        tgt_ID, src_ID = pair[0], pair[1]
        print('\tMatching %03d %03d' %(tgt_ID,src_ID))

        src_path = os.path.join(frag_folder, 'cloud_bin_%d.ply' %src_ID)
        tgt_path = os.path.join(frag_folder, 'cloud_bin_%d.ply' %tgt_ID)

        source = o3d.io.read_point_cloud(src_path)
        target = o3d.io.read_point_cloud(tgt_path)

        #Execute the complete pipeline
        ransac_reg, icp_reg = GlobalRegistration_withICP(source, target, voxel_size, inlier_th)

        ratio = compute_overlap_ratio(source, target, icp_reg.transformation, voxel_size)
        print('\t\tOverlap Ratio: %.4f' %ratio)

        # An alignmnet is only considered achieved with overlap ratio over 30%
        if ratio > 0.3:
            # Evaluate RANSAC on the full clouds
            # (if we use ransac_reg.fitness or ransac_reg.inlier_rmse, these
            #  results would be computed on the downsampled clouds used by RANSAC)
            ransac_eval = o3d.pipelines.registration.evaluate_registration(source, target, inlier_th, ransac_reg.transformation)
            #Add new result in the table
            new_result = pd.DataFrame({"Scene": scene,
                                       "Target": tgt_ID,
                                       "Source": src_ID,
                                       "RANSAC: Fitness": ransac_eval.fitness,
                                       "ICP: Fitness": icp_reg.fitness,
                                       "RANSAC: Inlier RMSE": ransac_eval.inlier_rmse,
                                       "ICP: Inlier RMSE": icp_reg.inlier_rmse,
                                       "Initial Guess": [ransac_reg.transformation],
                                       "Transformation": [icp_reg.transformation]})           #wrapper on matrix -> check observation below
            scene_results = pd.concat([scene_results, new_result], ignore_index=True)

    return scene_results




def get_matching_pairs(file):
    scene = None
    matching_pairs = defaultdict(list)
    with open(file, 'r') as f:
        for line in f:
            line = line.replace('\n','').split()

            if line[0] == 'Set:':
                scene = line[1]
                continue
            elif scene:
                matching_pairs[scene].append([int(line[0]), int(line[1])])
    return matching_pairs


def ICP_pipeline(dataset, split, voxel_size, inlier_th, subset=False):
    """
    Executes the complete ICP pipeline for a whole split
    (train, test or validation) of a given dataset.

    Args:
        dataset (string): Name of dataset
        split (string): Which split to be used ('train', 'validation' or 'test')
        voxel_size (float): Resulting size of voxels after downsampling
        dist_threhsold_ICP (float): Threshold distance for a correspondence
                                     pair to be considered valid during ICP

    Returns:
        pd.DataFrame: Pandas DataFrame with the registration results of the split
        collections.defaultdict: Hash map where keys are the scene's name and values are the list of all its clouds
    """

    #table to save registrations results of each scene
    dataset_results = pd.DataFrame({
    "Scene": pd.Series(dtype='str'),
    "Target": pd.Series(dtype='int'),
    "Source": pd.Series(dtype='int'),
    "RANSAC: Fitness": pd.Series(dtype='float'),
    "ICP: Fitness": pd.Series(dtype='float'),
    "RANSAC: Inlier RMSE": pd.Series(dtype='float'),
    "ICP: Inlier RMSE": pd.Series(dtype='float'),
    "Initial Guess": pd.Series(dtype='object'),
    "Transformation": pd.Series(dtype='object')
    })

    if subset:
        if not os.path.isfile('../configs/matching_pairs.txt'):
            print("The file with the subset of cloud pairs to be considered for the test was not found.\nPlease upload it at: ../configs/matching_pairs.txt'")
        else:
            matching_pairs = get_matching_pairs('../configs/matching_pairs.txt')

    for scene in data[dataset][split]:                                                              #for each scene
        print(f'Set: {scene}')

        frag_folder = f"{data_root}/{dataset}/{split}/{scene}/fragments"

        if not subset:
            num_frags = len(os.listdir(frag_folder))                           #compute number of fragments
            matching_pairs = {scene: []}
            for i in range(num_frags):
                for j in range(i + 2, num_frags):
                    matching_pairs[scene].append([i, j])

        scene_results = ICP_pipeline_scene(frag_folder, voxel_size, inlier_th, matching_pairs)     #apply pipeline
        
        dataset_results = pd.concat([dataset_results, scene_results], ignore_index=True)                    #append its results

    return dataset_results


## 3.1 Iterations Counter

def get_iterations(captured_output):
    """
    Parses the captured debug output from the ICP pipeline to extract the iteration counts for each alignment.
    Returns a DataFrame where each row corresponds to one alignment, with columns for the scene name,
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
        if line[0] == 'Set:':           # checks if it starts with 'Set:'
            cur_scene = line[1]         # extracts the scene name from the following text in the line

        # Get currrent matching pair
        elif line[0] == 'Matching':     # Checks if it starts with Matching
            target_ID = int(line[1])    # Obtain the target ID from the second field
            source_ID = int(line[2])    # Obtain the target ID from the third field
            ICP_iterations = 0          # Resets the ICP iterations counter for the next pair

        # Get ransac iterations
        elif line[0] == 'RANSAC':
            RANSAC_iterations = int(line[3])
          
        # If the line indicates an ICP iteration, increment the ICP iterations counter
        elif line[0] == 'ICP':
            ICP_iterations += 1

        # When the line starts with 'Overlap', it signals the end of the current alignment
        elif line[0] == 'Overlap': 

            # Stores the iterations of the current pair
            # Appending dictionaries is to a list is a straightforward way of build DataFrames later
            # Each dictionary in the list is a row of the table and the columns are the dictionaries keys
            iterations.append({
                'Scene': cur_scene,
                'Target': target_ID,
                'Source': source_ID,
                'RANSAC Iterations': RANSAC_iterations,
                'ICP Iterations': ICP_iterations
            })

    iterations_df = pd.DataFrame(iterations)
    return iterations_df


def augment_results(results, captured_output):
    """
    Augments the main results DataFrame with iteration counts by merging it with the iterations DataFrame.
    The merge is performed as a left join on the 'Scene', 'Target', and 'Source' columns.
    """
    iterations_df = get_iterations(captured_output)
    augmented_results = results.merge(iterations_df, 'left', on=['Scene', 'Target', 'Source'])

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



def execute_ICP_Pipeline(voxel_size, inlier_th, dataset, split, subset, run_name):
    """
    Executes the ICP pipeline and augments the results with RANSAC and ICP iteration counts.
    
    Parameters:
      - voxel_size: The voxel size for downsampling.
      - inlier_th: The inlier distance threshold.
      - dataset, split, subset: Parameters for specifying the dataset and experiment.
      - run_name: A name for the current run.
    
    Returns:
      - output_folder: The path to the output folder.
      - results: The augmented results DataFrame including iteration counts.
    """

    time = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime('%Y-%m-%d_%H-%M-%S')
    output_folder = f"../output/ICP_Pipeline/{run_name}-{time}"

    print("Applying ICP to TEST split using:\n--> voxel_size=%f\n--> distance threshold=%f" %(voxel_size, inlier_th))
    print('======================================================')

    # Create a Tee object to capture the output while printing to the console
    tee_buffer = Tee()

    # Redirect stdout to the Tee object during the ICP pipeline execution
    with redirect_stdout(tee_buffer):
        # Set Open3D's verbosity level to Debug to capture detailed iteration information
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        # Execute the ICP pipeline
        results = ICP_pipeline(dataset, split, voxel_size, inlier_th, subset)
        # Returns Open3D's verbosity level to default mode
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    # Retrieve the captured output as a string
    captured_output = tee_buffer.getvalue()

    # Merge the iteration information into the main results DataFrame
    results = augment_results(results, captured_output)

    return output_folder, results


# 5 Algorithm Testing and Results

voxel_size = 0.05            # best_voxel_size obtained at last validation
inlier_th = 0.05             # 5 cm --> this must be the same for ICP
dataset = "3DMatch"                        
split = 'test'
subset = False                # set to False to run all samples

run_name = "ICP_test_complete_wIterations"

output_folder, results = execute_ICP_Pipeline(voxel_size, inlier_th, dataset, split, subset, run_name)

print("============================================== Registration Table ==============================================")
print(results.to_string(index=False))
print("================================================================================================================")


def save_registration_table(df, output_folder):

    registration_folder = f"{output_folder}/registration"
    if not os.path.isdir(registration_folder):
        os.makedirs(registration_folder)

    reg_table_path =  f"{registration_folder}/registration_table_ICP.csv"
    df.to_csv(reg_table_path, index=False)
    print(f'Registration results table saved at: {reg_table_path}')

    return reg_table_path


reg_table_path = save_registration_table(results, output_folder)


## 5.1 Generating .log files

def string_to_nparray(matrix_str):
    rows_list = []
    for row in matrix_str.split('\n'):
        row = row.replace('[', '').replace(']','').split()
        row_array = np.array(row, dtype=float)
        rows_list.append(row_array)
    matrix = np.array(rows_list)
    return matrix


# In case you don't want to run all the testing again
# You can input the .csv file you previously obtained
# And you thiis cell to retrive the DF  table
def get_df_from_reg_table(reg_table_path):
    
    if not os.path.isfile(reg_table_path):
        print(f'No file found at: {reg_table_path}\n')
        return None
    else: 
        df = pd.read_csv(reg_table_path) 
        df['Source'] = df['Source'].astype(int)
        df['Target'] = df['Target'].astype(int)
        df['RANSAC Iterations'] = df['RANSAC Iterations'].astype(int)
        df['ICP Iterations'] = df['ICP Iterations'].astype(int)
        df['RANSAC: Fitness'] = df['RANSAC: Fitness'].astype(float)
        df['ICP: Fitness'] = df['ICP: Fitness'].astype(float)
        df['RANSAC: Inlier RMSE'] = df['RANSAC: Inlier RMSE'].astype(float)
        df['ICP: Inlier RMSE'] = df['ICP: Inlier RMSE'].astype(float)
        df['Initial Guess'] = df['Initial Guess'].apply(string_to_nparray)
        df['Transformation'] = df['Transformation'].apply(string_to_nparray)
        return df


results = get_df_from_reg_table(reg_table_path)   # change the path to the csv you desire


def get_nFrags(subset, data_root, data_dict, dataset, split):
    """
    Returns a dict mapping each scene name with the number of fragments to be written in the log header.
    If subset=True, reads matching_pairs.txt and computes frag_count.
    Otherwise counts .ply files in each scene's fragments folder.
    """

    nFrags_dict = defaultdict(int)

    if subset:
        if not os.path.isfile('../configs/matching_pairs.txt'):
            print("The file with the subset of cloud pairs to be considered for the test was not found.\nPlease upload it at: ../configs/matching_pairs.txt'")
            return None
        else:
            matching_pairs = get_matching_pairs('../configs/matching_pairs.txt')
            for scene in matching_pairs.keys():
                nPairs = len(matching_pairs[scene])
                nFrags = int((1 + np.sqrt(1 + 8*nPairs))/2)
                nFrags_dict[scene] = nFrags

    else:
        for scene in data_dict[dataset][split]:                                      #for each scene
                frag_folder = f"{data_root}/{dataset}/{split}/{scene}/fragments"
                nFrags_dict[scene] = len(os.listdir(frag_folder))               #compute number of fragments
    
    return nFrags_dict


def write_log(df, log_path, nFrags, final_transformation=True):
    """
    Writes a .log file in the 3dmatch format:
      target_id source_id nFrags
      (4×4 matrix)
    """

    with open(log_path, 'w') as f:
        for _, df_row in df.iterrows():
            source_id = df_row['Source']
            target_id = df_row['Target']
            if final_transformation:
                transformation = df_row['Transformation']
            else:
                transformation = df_row['Initial Guess']
            f.write(f"{target_id} {source_id} {nFrags}\n")
            for i in range(4):
                f.write(" ".join(map('{0:.12f}'.format, transformation[i])) + "\n")


def save_registration_logs(df, subset, data_root, data_dict, dataset, split, output_folder):
    """
    Splits the results DF by scene, computes the correct nFrags for each,
    and writes both final and initial‐guess logs.
    """

    logs_folder = f"{output_folder}/registration/logs"
    guesses_logs_folder = f'{output_folder}/registration/initial_guesses_logs'
    os.makedirs(logs_folder, exist_ok=True)
    os.makedirs(guesses_logs_folder, exist_ok=True)

    nFrags_dict = get_nFrags(subset, data_root, data_dict, dataset, split)

    scene_list = df['Scene'].unique()                       # list of all unique scene
    for scene in scene_list:
        df_scene = df[df['Scene'] == scene]                 # filters DF for a specific scene
        log_path = f"{logs_folder}/{scene}_ICP.log"
        guess_log_path = f'{guesses_logs_folder}/{scene}_ICP.log'
        nFrags = nFrags_dict[scene]
        print("writing:", log_path)
        print("Writing:", guess_log_path)
        write_log(df_scene, log_path, nFrags)
        write_log(df_scene, guess_log_path, nFrags, final_transformation=False)


save_registration_logs(results, subset, data_root, data, dataset, split, output_folder)



eval_csv_path = f"{output_folder}/evaluation/registration_evaluation_ICP.csv"
if os.path.isfile(eval_csv_path):
    reg_eval_ICP = pd.read_csv(eval_csv_path)
    print("============================================== Evaluation Table ==============================================")
    print(reg_eval_ICP.to_string(index=False))
    print("============================================================================================================")
else:
    print(f'No file found at: {eval_csv_path}\nPlease insert the .csv file with the obtained evaluation results in the expected path with the expected file name')


## 5.3 Results Analysis

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
    filename = f"{output_folder}/evaluation/mean_fitness_and_RMSE_table_ICP.csv"
    analysis.to_csv(filename, index=False)
    print(f'Registration results summary table saved at: {filename}')

    return analysis


analysis = assess_results(results)
print("============================================== Analysis Table ==============================================")
print(analysis.to_string(index=False))
print("============================================================================================================")
