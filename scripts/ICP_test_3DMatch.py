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
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo



# 2 Input Data
# Now, as mentioned in the beginning, we are going to use the 3DMatch dataset.
# More specifically, we will adopt the preprocessed by OverlapPedrato an avilable
# at https://share.phys.ethz.ch/~gsg/pairwise_reg/3dmatch.zip
# In order to read it, first we create a folder dedicated to the data used in our
# tests. Then, we download the `.zip` file containing the data and unzip it into the data folder.
# Besides that, for this case, we must correct the data split by creating a
# `validation` split as explained in the notes of this project's meetings.
# The scenes that should be used for validation are defined in `configs/3dmatch_val.txt`.

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


# Then we can create a HashMap (dictionary) where each dataset is a key and its
# values are another hash map which contains the scenes of each split (train,
# test and validation). By defining this, later we can read any pair of clouds
# using this HashMap as a directory reference.
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


# We can also define a function to visualize the given data split data (pre processed
# by OverlapPredator and also used for GeoTransformer).

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
# Now, we define the same functions as before to implement the complete pipeline.
# OBS: the only diference here is that for the ICP refinement, since a plane-to-plane
# ICP algorithm was used, we had to modify its function to estimate the normals of
# the target cloud. This was needed because in the preprocessing function, only the
# normals of the downsampled clouds are estimated, but the ICP uses the clouds at
# their original resolution.

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



def execute_GlobalRegistration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
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

    dist_threshold = 1.5*voxel_size

    global_registration = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                          source_down, target_down, source_fpfh, target_fpfh,
                          True,                                                                             #mutual filter activated
                          dist_threshold,                                                                   #max_correspondence_distance
                          o3d.pipelines.registration.TransformationEstimationPointToPoint(False),           #point-to-point estimation without scaling
                          3,                                                                                #ransac_n
                          [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),          #checkers to be used
                           o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_threshold)],
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

# Now we can define an overall registration function that combines all the previous ones.

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
    result_ransac = execute_GlobalRegistration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

    #Local Refinement (ICP)
    trans_init = result_ransac.transformation
    result_icp = execute_ICPrefinement(source, target, inlier_th, trans_init, voxel_size)

    return result_icp

# Hence, we can apply it to all our data. As required by the [3DMatch](https://3dmatch.cs.princeton.edu/#geometric-registration-benchmark)
# dataset, in order to properly evaluate it, **for each scene, we have to extensively try
# to register each non-consecutive fragment pair**: # $ (P_i, P_j)_{i+1<j} $

# First we define a function that performs the registration for all non-consecutive
# pairs of clouds from a specific scene.

# Notice that here you can use a `matching_pairs` list that defines a subset of pairs
# to be considered for testing. This is done to allow to run faster and partial tests.
# Besides that, only pairs with an overlap of at least `30%` are saved, following the
# author's evaluation convention (i.e., an alignment is considered achieved only if
# it produced a registration with over 30% overlap)

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
    "Fitness": pd.Series(dtype='float'),
    "Inlier RMSE": pd.Series(dtype='float'),
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
        final_reg = GlobalRegistration_withICP(source, target, voxel_size, inlier_th)

        ratio = compute_overlap_ratio(source, target, final_reg.transformation, voxel_size)
        print('\t\tOverlap Ratio: %.4f' %ratio)

        if ratio > 0.3:
            #Add new result in the table
            new_result = pd.DataFrame({"Scene": scene,
                                       "Target": tgt_ID,
                                       "Source": src_ID,
                                       "Fitness": final_reg.fitness,
                                       "Inlier RMSE": final_reg.inlier_rmse,
                                       "Transformation": [final_reg.transformation]})           #wrapper on matrix -> check observation below
            scene_results = pd.concat([scene_results, new_result], ignore_index=True)

    return scene_results


# OBS: To save the transformation in the table, we need to use a wrapper around the
# numpy array, so we pass it as a list using [ ]. This is needed because Pandas expects
# each column/row to be a sequence of values. So, without the brackets, Pandas would
# try to iterate over *final_reg.transformation* and add each row of the transformation
# matrix as separate rows of the dataframe, which is not desired. By enclosing it in
# brackets, we ensure that the entire transformation matrix is inserted as a single
# entry in the DataFrame.

# Then we can also define a function to perform the pipeline on all the scenes of
# a given split of a dataset.

# OBS: here we use the `defaultdict(list)` just to avoid having to initialize each
# key. In this way, any time we add something to a key, in case that key does not
# exist yet, it is automatically initialized with its default value as a empty list.

# As mentioned before, here you can use a `configs/matching_pairs.txt` file that defines
# which subset of pairs should be consider for testing. This is done to allow to run
# fast and partial tests. This file is generated by the FCGF algorithm, in such a
# way that first we perform the test on FCGF, which randomly samples non-conversutive
#  for alignment. Then, it saves the pairs it considered in a `.txt` file, which we
# use here to guarantee that we valuate ICP with the same samples (pairs).

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
    "Fitness": pd.Series(dtype='float'),
    "Inlier RMSE": pd.Series(dtype='float'),
    "Transformation": pd.Series(dtype='object')
    })

    if subset:
        if not os.path.isfile('../configs/matching_pairs.txt'):
            print("The file with the subset of cloud pairs to be considered for the test was not found.\nPlease upload it at: ../configs/matching_pairs.txt'")
        else:
            matching_pairs = get_matching_pairs('../configs/matching_pairs.txt')

    for scene in data[dataset][split]:                                                              #for each scene
        print("Set: %s," %scene)

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



# 5 Algorithm Testing and Results
# Now we can use this best combination on the test
# split to assess how thisalgorithm is performing.

voxel_size = 0.05            # best_voxel_size obtained at last validation
inlier_th = 0.05             # 5 cm --> this must be the same for ICP
dataset = "3DMatch"                        
split = 'test'
subset = True

run_name = "script_test_subset3"

time = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime('%Y-%m-%d_%H-%M-%S')
output_folder = f"../output/ICP_Pipeline/{run_name}-{time}"

print("Applying ICP to TEST split using:\n--> voxel_size=%f\n--> distance threshold=%f" %(voxel_size, inlier_th))
print('======================================================')
results = ICP_pipeline(dataset, split, voxel_size, inlier_th, subset)              #apply pipeline

print("============================================== Registration Table ==============================================")
print(results.to_string(index=False))
print("================================================================================================================")


# Export results table so we can use it later without needing to run the testing again.

def save_registration_table(df, output_folder):

    registration_folder = f"{output_folder}/registration"
    if not os.path.isdir(registration_folder):
        os.makedirs(registration_folder)

    reg_table_path =  f"{registration_folder}/registration_table_ICP.csv"
    df.to_csv(reg_table_path, index=False)
    print(f'Registration results table saved at: {reg_table_path}')

    return reg_table_path

reg_table_path = save_registration_table(results, output_folder)



# 5.1 Generating .log files
# From the result table, we can obtain the .log file to evaluate the registration
# with the MATLAB script provided by the 3DMatch's author.

# First, we define an auxiliary function to retrieve the registration table 
# results, which was previously saved as a `.csv` file, back to a Pandas 
# DataFrame. This was done to prevent the need of running the tests again 
# in case the notebook was reset.

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
        df['Fitness'] = df['Fitness'].astype(float)
        df['Inlier RMSE'] = df['Inlier RMSE'].astype(float)
        df['Transformation'] = df['Transformation'].apply(string_to_nparray)
        return df

results = get_df_from_reg_table(reg_table_path)   # change the path to the csv you desire

# Now with the retrieved DataFrame we can define
# the function to generate the `.log` files to be stored.

def get_nFrags(subset, dataset, split):

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
        for scene in data[dataset][split]:                                      #for each scene
                frag_folder = f"{data_root}/{dataset}/{split}/{scene}/fragments"
                nFrags_dict[scene] = len(os.listdir(frag_folder))               #compute number of fragments
    
    return nFrags_dict


def write_log(df, log_path, nFrags):

    with open(log_path, 'w') as f:
        for _, df_row in df.iterrows():
            source_id = df_row['Source']
            target_id = df_row['Target']
            transformation = df_row['Transformation']
            f.write(f"{target_id} {source_id} {nFrags}\n")
            for i in range(4):
                f.write(" ".join(map('{0:.12f}'.format, transformation[i])) + "\n")


def save_registration_logs(df, subset, dataset, split, output_folder):
    
    logs_folder = f"{output_folder}/registration/logs"
    if not os.path.isdir(logs_folder):
        os.makedirs(logs_folder)

    nFrags_dict = get_nFrags(subset, dataset, split)

    scene_list = results['Scene'].unique()                  # list of all unique scene
    for scene in scene_list:
        df_scene = df[df['Scene'] == scene]                 # filters DF for a specific scene
        log_path = f"{logs_folder}/{scene}_ICP.log"
        nFrags = nFrags_dict[scene]
        print("writing:", log_path)
        write_log(df_scene, log_path, nFrags)

save_registration_logs(results, subset, dataset, split, output_folder)


# We can then use these `.log` files in the MATLAB script
# to evaluate the results in terms of precision and recall.

# This will produce a `.csv` file. You can then import and visualize it here.

eval_csv_path = f"{output_folder}/evaluation/registration_evaluation_ICP.csv"
if os.path.isfile(eval_csv_path):
    reg_eval_ICP = pd.read_csv(eval_csv_path)
    print("============================================== Evaluation Table ==============================================")
    print(reg_eval_ICP.to_string(index=False))
    print("============================================================================================================")
else:
    print(f'No file found at: {eval_csv_path}\nPlease insert the .csv file with the obtained evaluation results in the expected path with the expected file name')



# 5.3 Results Analysis
# Notice that after applying our pipeline to a full split, we
# will obtain a table in which each row corresponds to one specific
# alignment. Hence, we can create a new table to present the overall
# results (errors/performance) for all clouds of each scene and of
# the whole split we selected from the dataset. In this way, we can
# obtain a summary of the overall performance of the ICP algorithm.

def assess_results(results):
    """
    Given a registration result table, computes the mean performance of the alignment
    for all clouds of a specific scene and for that whole split we selected from a dataset.

    Args:
        results (pd.DataFrame): Table containing the registration results of the split

    Returns:
        pd.DataFrame: Table with the overall (mean) performance results
    """

    analysis = results.copy()                                                                           # create a copy of results
    analysis = analysis.drop(["Source", "Target", "Transformation"], axis="columns")              # remove unnecessary columns

    analysis = analysis.groupby(["Scene"]).mean().reset_index()                                         # compute the averages of each scene
    analysis = analysis.rename(columns={"Fitness": "Mean Fitness",                                      # rename columns
                                        "Inlier RMSE": "Mean Inlier RMSE"})

    total_mean = pd.DataFrame({"Scene": "TOTAL", "Mean Fitness": analysis["Mean Fitness"].mean(),       # compute total split average
                               "Mean Inlier RMSE": analysis["Mean Inlier RMSE"].mean()}, index=[0])
    analysis = pd.concat([analysis, total_mean], ignore_index=True)                                     # add total split average

    # ensures the evaluation folder is present
    if not os.path.isdir(f"{output_folder}/evaluation"):
        os.makedirs(f"{output_folder}/evaluation")
    
    # save the obtained registration results summary table as a .csv in the output folder
    filename = f"{output_folder}/evaluation/mean_fitness_and_RMSE_table_ICP.csv"
    analysis.to_csv(filename, index=False)
    print(f'Registration results summary table saved at: {filename}')

    return analysis

analysis = assess_results(results)
print("============================================== Analysis Table ==============================================")
print(analysis.to_string(index=False))
print("============================================================================================================")
