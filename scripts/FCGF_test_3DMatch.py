# 1 Setting Up the Environment 
# Follow the steps presented at the `README.md` file to install both PyTorch
# and the Minkowski Engine. And check below if the installation was successful.
import MinkowskiEngine as ME
print(f'MinkowskiEngine version: {ME.__version__}')
import torch
print(f'PyTorch version: {torch.__version__}')

# After that, we just need to import everything we are going to use.
import os
import sys
import copy
import numpy as np
import pandas as pd
import open3d as o3d
import subprocess
from urllib.request import urlretrieve
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

# And set Python to include `source/FCGF` in its search path. Otherwise, since this
# notebooks is in a different folder, we would not be able to import functions and
# other things from the FCGF folder.

# Get the absolute path of the source directory
sys.path.append(os.path.abspath("../source/FCGF"))



# 2 Input Data
# Since we will not retrain the model but use just the pre-trained weights,
# we can download only the test split. If the train split is ever needed,
# you can download it too by uncommenting the last block.

test_path = '../data/FCGF/threedmatch_test'
if not os.path.exists(test_path):
    print(f'Downloading data at {test_path}\n\n=================================================================\n')
    subprocess.run(["bash", "../source/FCGF/scripts/download_3dmatch_test.sh", test_path], check=True)
else:
    print(f'The data is already available at {test_path}')

# train_path = '../data/FCGF/threedmatch_train/'
# if not os.path.exists(train_path):
#     print(f'Downloading data at {train_path}\n=================================================================')
#     subprocess.run(["bash", "../source/FCGF/scripts/download_datasets.sh", train_path], check=True)
# else:
#     print(f'The data is already available at {train_path}')

# Besides that, we also need to download the pre-trained model so we can
# use it to perform the tests.

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



# 3 Testing 
# Here we run the benchmark script to test the FCGF performance in both extraction
# features itself and providing a registration based on these features. To check
# the whole implementation of this method as well as details on how this benchmark
# is performed, please refer to: https://github.com/gabriel-corteletti/FCGF

subset = 3
voxel_size = 0.025
inlier_th = 0.05            # 5 cm --> this must be the same for ICP
model = fcgf_weight_path

run_name = "script_test_subset3"

time = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime('%Y-%m-%d_%H-%M-%S')
output_folder = f"../output/FCGF/{run_name}-{time}"
feature_path = f"{output_folder}/features"

subprocess.run([
    "python", "../source/FCGF/scripts/benchmark_3dmatch.py",
    "--source", test_path,
    "--target", feature_path,
    "--voxel_size", str(voxel_size),
    "--subset", str(subset),
    "--model", model,
    "--extract_features",
    "--evaluate_feature_match_recall",
    "--evaluate_registration"],
    check=True)



# 3.1 Quantitative Analysis
# Now, we have one `.log` file for each scene. To evaluate the registration
# performance in terms of **Recall** and **Preccision**, we must use a MATLAB
# script provided by the author and adapted for our use case (the adapted script
# is available at: github.com/gabriel-corteletti/3dmatch-toolbox

# Therefore, since we cannot run a MATLAB script in a Python environment, we
# must open MATLAB and run it from there using the `.log` files we obtained
# here (stored at `output/FCGF/{run_name}-{time}/registration/logs`)
# However, if we are considering just a subset of the test split, we have to
# adapt also the ground truth files to consider only the results of the same
# alignments we performed, otherwise the computation of registration recall
# and precision will be affected by this inconsistency.

# To do so, given the subset size we are considering, we first compute all
# possible pairs to be aligned, i.e. all non-consecutive pairs, and then we
# create a copy of the original ground truth *.log* and *.info* files where
# we select only the data related to these possible pairs.

# We need to compute all the possible pairs instead of simply considering the
# same pairs inserted in the obtained registration .log files because it might
# happen that the algorithm judges itself not able to perform a certain alignment.

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


def get_scene_reducedGT(log_path, info_path, scene_out_gt_path, matching_pairs, scene):

    if not os.path.isdir(scene_out_gt_path):
        os.makedirs(scene_out_gt_path)

    out_log_path = os.path.join(scene_out_gt_path, 'gt.log')
    out_info_path = os.path.join(scene_out_gt_path, 'gt.info')
    copy = False

    n_pairs = len(matching_pairs[scene])
    num_frag = int((1 + np.sqrt(1 + 8*n_pairs))/2)

    with open(log_path, 'r') as i:
        with open(out_log_path, 'w') as o:
            for idx, line in enumerate(i):

                line = line.replace('\n', '').replace('\t', '').split()

                if idx%5 == 0:
                    if [int(line[0]), int(line[1])] in matching_pairs[scene]:
                        o.write(f"{line[0]}\t {line[1]}\t {num_frag}\n")
                        copy = True
                    else:
                        copy = False
                elif copy:
                    o.write(f"{line[0]}\t {line[1]}\t {line[2]}\t {line[3]}\n")

    with open(info_path, 'r') as i:
        with open(out_info_path, 'w') as o:
            for idx, line in enumerate(i):

                line = line.replace('\n', '').replace('\t', '').split()

                if idx%7 == 0:
                    if [int(line[0]), int(line[1])] in matching_pairs[scene]:
                        o.write(f"{line[0]}\t {line[1]}\t {num_frag}\n")
                        copy = True
                    else:
                        copy = False
                elif copy:
                    o.write(f"{line[0]}\t {line[1]}\t {line[2]}\t {line[3]}\t {line[4]}\t {line[5]}\n")


def get_reducedGT(output_folder, test_path):

    out_gt_path = f"{output_folder}/groundtruth"
    if not os.path.isdir(out_gt_path):
        os.makedirs(out_gt_path)

    matching_pairs = get_matching_pairs(f"{output_folder}/registration/matching_pairs.txt")

    for filename in os.listdir(test_path):
        aux = filename.split('-')
        if aux[-1] == 'evaluation':
            log_path = os.path.join(test_path, filename, 'gt.log')
            info_path = os.path.join(test_path, filename, 'gt.info')
            scene_out_gt_path = os.path.join(out_gt_path, filename)
            scene = '-'.join(aux[:-1])
            get_scene_reducedGT(log_path, info_path, scene_out_gt_path, matching_pairs, scene)
    print(f'Reduced ground truth files saved at: {out_gt_path}')

get_reducedGT(output_folder, test_path)

# Then we run the script evaluate.m (https://github.com/gabriel-corteletti/3dmatch-toolbox/blob/master/evaluation/geometric-registration/evaluate.m)
# to obtain a .csv file containing the registration recall and precision of each scene as
# well as the overall average results. After obtaining this, we can upload it back to this
# environment and present it in a table below.
eval_csv_path = f"{output_folder}/evaluation/registration_evaluation_FCGF.csv"
if os.path.isfile(eval_csv_path):
    reg_eval_FCGF = pd.read_csv(eval_csv_path)
    print("============================================== Evaluation Table ==============================================")
    print(reg_eval_FCGF.to_string(index=False))
    print("============================================================================================================")
else:
    print(f'No file found at: {eval_csv_path}\nPlease insert the .csv file with the obtained evaluation results in the expected path with the expected file name')

# We can also compute the fitness and inlier RMSE to compare it to that of ICP. To do so,
# we need to access the results stored in the output folder of the run. Hence, we define
# a function responsible for retrieving the obtained transformation for all considered
# pairs for all scenes, and store it in a DataFrame.
def get_results_from_folder(output_folder, inlier_th):

    results = pd.DataFrame({
    "Scene": pd.Series(dtype='str'),
    "Target": pd.Series(dtype='int'),
    "Source": pd.Series(dtype='int'),
    "Fitness": pd.Series(dtype='float'),
    "Inlier RMSE": pd.Series(dtype='float'),
    "Transformation": pd.Series(dtype='object')
    })
    
    logs_folder = f'{output_folder}/registration/logs'

    for scene in os.listdir(logs_folder):
        scene_name = scene.split('_FCGF')[0]

        found = -1
        transformation = []

        with open(os.path.join(logs_folder, scene), 'r') as f:
            for idx, line in enumerate(f):
                line = line.replace('\n', '').replace('\t', '').split()
                if idx%5 == 0:
                    tgt_ID = int(line[0])
                    src_ID = int(line[1])
                    found = 0
                elif (found > -1) and (found < 4):
                    transformation.append([float(i) for i in line])
                    found += 1
                    if found == 4:


                        source = o3d.io.read_point_cloud(os.path.join(test_path, scene_name, 'cloud_bin_%s.ply' %src_ID))
                        target = o3d.io.read_point_cloud(os.path.join(test_path, scene_name, 'cloud_bin_%s.ply' %tgt_ID))

                        eval = o3d.pipelines.registration.evaluate_registration(source, target, inlier_th, transformation)


                        new_result = pd.DataFrame({'Scene': scene_name,
                                                   'Target': tgt_ID,
                                                   'Source': src_ID,
                                                   'Fitness': eval.fitness,
                                                   'Inlier RMSE': eval.inlier_rmse,
                                                   'Transformation': [transformation]})
                        results = pd.concat([results, new_result], ignore_index=True)
                        found = -1
                        transformation = []

    # save the obtained table as a .csv in the output folder
    filename = f"{output_folder}/registration/registration_table_FCGF.csv"
    results.to_csv(filename, index=False)
    print(f'Registration results table saved at: {filename}')

    return results

results = get_results_from_folder(output_folder, inlier_th)
print("============================================== Registration Table ==============================================")
print(results.to_string(index=False))
print("================================================================================================================")

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
    filename = f"{output_folder}/evaluation/mean_fitness_and_RMSE_table_FCGF.csv"
    analysis.to_csv(filename, index=False)
    print(f'Registration results summary table saved at: {filename}')

    return analysis

analysis = assess_results(results)
print("============================================== Analysis Table ==============================================")
print(analysis.to_string(index=False))
print("============================================================================================================")
