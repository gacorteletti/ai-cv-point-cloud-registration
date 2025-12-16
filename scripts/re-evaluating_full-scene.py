# # 1 Setup

import pandas as pd
import numpy as np
import open3d as o3d
from typing import Literal
import matplotlib.pyplot as plt
import seaborn as sns
import os

out_folder = "../output/SuctionNet/complete_test-2025-10-07_08-23-06"

GO_full = pd.read_csv(f"{out_folder}/geometric_only-full_scene/registration_table_full.csv")
GO_obj = pd.read_csv(f"{out_folder}/geometric_only-single_object/registration_table_full.csv")
DL_full = pd.read_csv(f"{out_folder}/deep_learning-full_scene/registration_table_full.csv")
DL_obj = pd.read_csv(f"{out_folder}/deep_learning-single_object/registration_table_full.csv")

# # 2 Re-computing fitness and inlier RMSE

def parse_transform_array(col):
    """
    Parse a whole Series of transforms (string or array-like) into a list of 4x4 np.arrays.
    """
    parsed = []
    for T in col:
        if isinstance(T, str):
            T_clean = T.replace('[', '').replace(']', '')
            parsed.append(np.fromstring(T_clean, sep=' ').reshape(4, 4))
        else:
            parsed.append(np.array(T, dtype=float).reshape(4, 4))
    return parsed


def re_evaluate_full_scene_results(df, inlier_th, dataset_dir="../data/SuctionNet"):
    """
    Re-evaluate fitness/RMSE using single-object clouds as targets,
    reusing the transforms that were computed on full-scene clouds.

    Optimizations:
      - cache CAD models per obj_id
      - cache single-object clouds per (scene_id, image_id, obj_id)
      - parse all transforms once outside the loop
      - fill results into NumPy arrays, then assign to df at the end
    """
    # Work on a copy and ensure a clean 0..N-1 index for array indexing
    df = df.copy().reset_index(drop=True)

    n = len(df)

    # Pre-parse all transforms once
    ransac_T_list = parse_transform_array(df['RANSAC: Transformation'])
    icp_T_list    = parse_transform_array(df['ICP: Transformation'])

    # Arrays to hold new metrics
    new_ransac_fit   = np.zeros(n, dtype=float)
    new_icp_fit      = np.zeros(n, dtype=float)
    new_ransac_rmse  = np.zeros(n, dtype=float)
    new_icp_rmse     = np.zeros(n, dtype=float)

    # Caches to avoid re-reading PLY files
    cad_cache = {}          # key: obj_id         -> o3d.geometry.PointCloud
    target_cache = {}       # key: (scene,image,obj) -> o3d.geometry.PointCloud

    for i in range(n):
        row = df.iloc[i]

        scene_id = int(row["Scene"])
        image_id = int(row["Image"])
        obj_id   = int(row["Object"])

        # ---------- SOURCE (CAD model) ----------
        if obj_id not in cad_cache:
            source_path = f'{dataset_dir}/models/ply/{obj_id-1:03d}.ply'
            if not os.path.exists(source_path):
                print(f"[WARN] Missing CAD model: {source_path} – skipping row {i}")
                continue
            cad_cache[obj_id] = o3d.io.read_point_cloud(source_path)
        source = cad_cache[obj_id]

        # ---------- TARGET (single-object cloud) ----------
        key = (scene_id, image_id, obj_id)
        if key not in target_cache:
            target_path = (
                f'{dataset_dir}/acquired/single_object/scene_{scene_id:04d}/'
                f'image_{image_id:04d}/'
                f'scene_{scene_id:04d}_image_{image_id:04d}_object_{obj_id:02d}.ply'
            )
            if not os.path.exists(target_path):
                print(f"[WARN] Missing target: {target_path} – skipping row {i}")
                continue
            target_cache[key] = o3d.io.read_point_cloud(target_path)
        target = target_cache[key]

        # ---------- Transforms (already parsed) ----------
        T_ransac = ransac_T_list[i]
        T_icp    = icp_T_list[i]

        # ---------- Re-evaluate on object-only target ----------
        eval_ransac = o3d.pipelines.registration.evaluate_registration(
            source, target, inlier_th, T_ransac
        )
        eval_icp = o3d.pipelines.registration.evaluate_registration(
            source, target, inlier_th, T_icp
        )

        # Store metrics
        new_ransac_fit[i]  = eval_ransac.fitness
        new_icp_fit[i]     = eval_icp.fitness
        new_ransac_rmse[i] = eval_ransac.inlier_rmse
        new_icp_rmse[i]    = eval_icp.inlier_rmse

    # Assign back in one shot
    df['RANSAC: Fitness']       = new_ransac_fit
    df['ICP: Fitness']          = new_icp_fit
    df['RANSAC: Inlier RMSE']   = new_ransac_rmse
    df['ICP: Inlier RMSE']      = new_icp_rmse

    return df


inlier_th = 0.005

GO_full = re_evaluate_full_scene_results(GO_full, inlier_th)
DL_full = re_evaluate_full_scene_results(DL_full, inlier_th)

GO_full.to_csv(f'{out_folder}/geometric_only-full_scene/registration_table_full_re-evaluated.csv', index=False)
DL_full.to_csv(f'{out_folder}/deep_learning-full_scene/registration_table_full_re-evaluated.csv', index=False)


filt_GO_full = GO_full[GO_full['ICP: Fitness'] > 0.3].copy()
filt_GO_obj = GO_obj[GO_obj['ICP: Fitness'] > 0.3].copy()
filt_DL_full = DL_full[DL_full['ICP: Fitness'] > 0.3].copy()
filt_DL_obj = DL_obj[DL_obj['ICP: Fitness'] > 0.3].copy()

filt_GO_full.to_csv(f'{out_folder}/geometric_only-full_scene/registration_table_filtered_re-evaluated.csv', index=False)
filt_DL_full.to_csv(f'{out_folder}/deep_learning-full_scene/registration_table_filtered_re-evaluated.csv', index=False)


# # 3 Assessing Recall

pct_go_object = 100 * len(filt_GO_obj) / len(GO_obj)
pct_go_full = 100 * len(filt_GO_full) / len(GO_full)
pct_dl_object = 100 * len(filt_DL_obj) / len(DL_obj)
pct_dl_full = 100 * len(filt_DL_full) / len(DL_full)

pct_df = pd.DataFrame(
    {'Setting': ['Single Objects', 'Full Scenes'],
     'Geometric-Only': [pct_go_object, pct_go_full],
     'Deep-Learning': [pct_dl_object, pct_dl_full]}
)

filename = f"{out_folder}/recall_estimated_totals_re-evaluated.csv"
pct_df.to_csv(filename, index=False)
print(f'Estimated Recall totals saved at: {filename}')

print('==================== Estimated Recall ====================')
print(pct_df.to_string(index=False))


# 4 Assessing Alignment Quality

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
    filename = f"{output_folder}/local_evaluation_per_{'object' if groupby_col == 'Object' else 'scene'}_re-evaluated.csv"
    analysis.to_csv(filename, index=False)
    print(f'Local evaluation table saved at: {filename}')

    return analysis


# ## 4.1 Results per Scene

# We begin by summarizing the results of each pipeline and setting across all individual
# scenes, as well as the overall performance.

GO_full_assess = assess_results(filt_GO_full, 'geometric_only', 'full_scene', out_folder)
GO_obj_assess = assess_results(filt_GO_obj, 'geometric_only', 'single_object', out_folder)
DL_full_assess = assess_results(filt_DL_full, 'deep_learning', 'full_scene', out_folder)
DL_obj_assess = assess_results(filt_DL_obj, 'deep_learning', 'single_object', out_folder)

# We then consolidate the overall results ("TOTAL" row) of each pipeline/setting into
# a single table to facilitate comparison of each pipeline’s performance under the different settings.

total_GO_full = GO_full_assess[GO_full_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()
total_GO_obj = GO_obj_assess[GO_obj_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()
total_DL_full = DL_full_assess[DL_full_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()
total_DL_obj = DL_obj_assess[DL_obj_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()

total_comparison = pd.concat([total_GO_full, total_DL_full, total_GO_obj, total_DL_obj])

# Save the obtained registration results summary table as a .csv in the output folder
filename = f"{out_folder}/overall_evaluation_per_scene_totals_re-evaluated.csv"
total_comparison.to_csv(filename, index=False)
print(f'Overall comparison table (totals per pipeline/setting combination) saved at: {filename}')

print('==================== Comparison Table ====================')
print(total_comparison.to_string(index=False))


# ## 4.2 Results per Object

# Similar to the previous analysis, we can also compute the results per object rather
# than per scene. This enables us to identify how different objects perform across
# pipelines, regardless of viewing angle or scene arrangement (i.e., how the
# objects are positioned within a scene). In this way, we can examine whether the
# shape or geometry of each object correlates with the final results.

GO_full_assess_obj = assess_results(filt_GO_full, 'geometric_only', 'full_scene', out_folder, groupby_col="Object")
GO_object_assess_obj = assess_results(filt_GO_obj, 'geometric_only', 'single_object', out_folder, groupby_col="Object")
DL_full_assess_obj = assess_results(filt_DL_full, 'deep_learning', 'full_scene', out_folder, groupby_col="Object")
DL_object_assess_obj = assess_results(filt_DL_obj, 'deep_learning', 'single_object', out_folder, groupby_col="Object")

# Then, we evaluate the recall of each individual object under each pipeline-setting combination.

dfs = [GO_full, DL_full, GO_obj, DL_obj]
filt_dfs = [filt_GO_full, filt_DL_full, filt_GO_obj, filt_DL_obj]
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

filename = f"{out_folder}/objects_recall_detailed_re-evaluated.csv"
obj_recall.to_csv(filename, index=False)
print(f'Per-object recall results on different pipelines/settings table saved at: {filename}')

print('==================== Per-Object Recall Results ====================')
print(obj_recall.to_string(index=False))

# Since the dataset contains many different objects, it is more informative to focus on the top
# three and bottom three cases to better understand their performance metrics and underlying
# behavior. By examining these extreme cases, we can gain insight into how object-specific
# characteristics influence alignment performance across different pipelines and settings.

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

filename = f"{out_folder}/objects_recall_top_bottom_re-evaluated.csv"
summary_df.to_csv(filename, index=False)
print(f'Top and bottom objects\' recall results summary table saved at: {filename}')

print('==================== Top and Bottom Objects\' Recall Summary ====================')
summary_df

# If a comprehensive analysis of all objects is desired, we can focus on the recall
# values alone and visualize them using a heatmap.

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
plt.savefig(f'{out_folder}/recall_heatmap_re-evaluated.png', dpi=300, bbox_inches='tight')
print(f'Recall heatmap saved at: {out_folder}/recall_heatmap_re-evaluated.png')
plt.close()


# 5 Evaluating on easiest and hardest objects

filt_GO_full = pd.read_csv(f'{out_folder}/geometric_only-full_scene/registration_table_filtered_re-evaluated.csv')
filt_DL_full = pd.read_csv(f'{out_folder}/deep_learning-full_scene/registration_table_filtered_re-evaluated.csv')

filt_GO_obj = pd.read_csv(f"{out_folder}/geometric_only-single_object/registration_table_filtered.csv")
filt_DL_obj = pd.read_csv(f"{out_folder}/deep_learning-single_object/registration_table_filtered.csv")

obj_recall = pd.read_csv(f"{out_folder}/objects_recall_detailed_re-evaluated.csv")


## 5.1 Single Object

df = obj_recall[['Object', 'pct GO Single', 'pct DL Single']].copy()
df['dl_edge'] = df['pct DL Single'] - df['pct GO Single']
df = df.sort_values(by='dl_edge', ascending=False).reset_index(drop=True)


### 5.1.1 Top 3 objects

top_df = df.iloc[:3].copy()

# account for only the 3 top DL objects
aux_GO_full = filt_GO_full[filt_GO_full['Object'].isin(top_df['Object'])]
aux_DL_full = filt_DL_full[filt_DL_full['Object'].isin(top_df['Object'])]
aux_GO_obj = filt_GO_obj[filt_GO_obj['Object'].isin(top_df['Object'])]
aux_DL_obj = filt_DL_obj[filt_DL_obj['Object'].isin(top_df['Object'])]

aux_GO_full_assess = assess_results(aux_GO_full, 'geometric_only', 'full_scene', out_folder)
aux_DL_full_assess = assess_results(aux_DL_full, 'deep_learning', 'full_scene', out_folder)
aux_GO_obj_assess = assess_results(aux_GO_obj, 'geometric_only', 'single_object', out_folder)
aux_DL_obj_assess = assess_results(aux_DL_obj, 'deep_learning', 'single_object', out_folder)

aux_total_GO_full = aux_GO_full_assess[aux_GO_full_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()
aux_total_GO_obj = aux_GO_obj_assess[aux_GO_obj_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()
aux_total_DL_full = aux_DL_full_assess[aux_DL_full_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()
aux_total_DL_obj = aux_DL_obj_assess[aux_DL_obj_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()

total_comparison = pd.concat([aux_total_GO_full, aux_total_DL_full, aux_total_GO_obj, aux_total_DL_obj])

filename = f"{out_folder}/overall_evaluation_per_scene_totals_top_3_on_single.csv"
total_comparison.to_csv(filename, index=False)
print(f'Overall comparison table (totals per pipeline/setting combination) saved at: {filename}')

print('==================== Single Object - Top 3 ====================')
print(total_comparison.to_string(index=False))


### 5.1.2 Bottom 3 Objects

bot_df = df.iloc[-3:].copy()

# account for only the 3 top DL objects
aux_GO_full = filt_GO_full[filt_GO_full['Object'].isin(bot_df['Object'])]
aux_DL_full = filt_DL_full[filt_DL_full['Object'].isin(bot_df['Object'])]
aux_GO_obj = filt_GO_obj[filt_GO_obj['Object'].isin(bot_df['Object'])]
aux_DL_obj = filt_DL_obj[filt_DL_obj['Object'].isin(bot_df['Object'])]

aux_GO_full_assess = assess_results(aux_GO_full, 'geometric_only', 'full_scene', out_folder)
aux_DL_full_assess = assess_results(aux_DL_full, 'deep_learning', 'full_scene', out_folder)
aux_GO_obj_assess = assess_results(aux_GO_obj, 'geometric_only', 'single_object', out_folder)
aux_DL_obj_assess = assess_results(aux_DL_obj, 'deep_learning', 'single_object', out_folder)

aux_total_GO_full = aux_GO_full_assess[aux_GO_full_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()
aux_total_GO_obj = aux_GO_obj_assess[aux_GO_obj_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()
aux_total_DL_full = aux_DL_full_assess[aux_DL_full_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()
aux_total_DL_obj = aux_DL_obj_assess[aux_DL_obj_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()

total_comparison = pd.concat([aux_total_GO_full, aux_total_DL_full, aux_total_GO_obj, aux_total_DL_obj])

filename = f"{out_folder}/overall_evaluation_per_scene_totals_bottom_3_on_single.csv"
total_comparison.to_csv(filename, index=False)
print(f'Overall comparison table (totals per pipeline/setting combination) saved at: {filename}')

print('==================== Single Object - Bottom 3 ====================')
print(total_comparison.to_string(index=False))


### 5.1.3 Middle 3 Objects

mid = len(df) // 2
mid_df = df.iloc[mid-1:mid+2].copy()

# account for only the 3 top DL objects
aux_GO_full = filt_GO_full[filt_GO_full['Object'].isin(mid_df['Object'])]
aux_DL_full = filt_DL_full[filt_DL_full['Object'].isin(mid_df['Object'])]
aux_GO_obj = filt_GO_obj[filt_GO_obj['Object'].isin(mid_df['Object'])]
aux_DL_obj = filt_DL_obj[filt_DL_obj['Object'].isin(mid_df['Object'])]

aux_GO_full_assess = assess_results(aux_GO_full, 'geometric_only', 'full_scene', out_folder)
aux_DL_full_assess = assess_results(aux_DL_full, 'deep_learning', 'full_scene', out_folder)
aux_GO_obj_assess = assess_results(aux_GO_obj, 'geometric_only', 'single_object', out_folder)
aux_DL_obj_assess = assess_results(aux_DL_obj, 'deep_learning', 'single_object', out_folder)

aux_total_GO_full = aux_GO_full_assess[aux_GO_full_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()
aux_total_GO_obj = aux_GO_obj_assess[aux_GO_obj_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()
aux_total_DL_full = aux_DL_full_assess[aux_DL_full_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()
aux_total_DL_obj = aux_DL_obj_assess[aux_DL_obj_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()

total_comparison = pd.concat([aux_total_GO_full, aux_total_DL_full, aux_total_GO_obj, aux_total_DL_obj])

filename = f"{out_folder}/overall_evaluation_per_scene_totals_middle_3_on_single.csv"
total_comparison.to_csv(filename, index=False)
print(f'Overall comparison table (totals per pipeline/setting combination) saved at: {filename}')

print('==================== Single Object - Middle 3 ====================')
print(total_comparison.to_string(index=False))


## 5.2 Full Scene

df = obj_recall[['Object', 'pct GO Full', 'pct DL Full']].copy()
df['dl_edge'] = df['pct DL Full'] - df['pct GO Full']
df = df.sort_values(by='dl_edge', ascending=False).reset_index(drop=True)

### 5.2.1 Top 3 objects

top_df = df.iloc[:3].copy()

# account for only the 3 top DL objects
aux_GO_full = filt_GO_full[filt_GO_full['Object'].isin(top_df['Object'])]
aux_DL_full = filt_DL_full[filt_DL_full['Object'].isin(top_df['Object'])]
aux_GO_obj = filt_GO_obj[filt_GO_obj['Object'].isin(top_df['Object'])]
aux_DL_obj = filt_DL_obj[filt_DL_obj['Object'].isin(top_df['Object'])]

aux_GO_full_assess = assess_results(aux_GO_full, 'geometric_only', 'full_scene', out_folder)
aux_DL_full_assess = assess_results(aux_DL_full, 'deep_learning', 'full_scene', out_folder)
aux_GO_obj_assess = assess_results(aux_GO_obj, 'geometric_only', 'single_object', out_folder)
aux_DL_obj_assess = assess_results(aux_DL_obj, 'deep_learning', 'single_object', out_folder)

aux_total_GO_full = aux_GO_full_assess[aux_GO_full_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()
aux_total_GO_obj = aux_GO_obj_assess[aux_GO_obj_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()
aux_total_DL_full = aux_DL_full_assess[aux_DL_full_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()
aux_total_DL_obj = aux_DL_obj_assess[aux_DL_obj_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()

total_comparison = pd.concat([aux_total_GO_full, aux_total_DL_full, aux_total_GO_obj, aux_total_DL_obj])

filename = f"{out_folder}/overall_evaluation_per_scene_totals_top_3_on_full.csv"
total_comparison.to_csv(filename, index=False)
print(f'Overall comparison table (totals per pipeline/setting combination) saved at: {filename}')

print('==================== Full Scene - Top 3 ====================')
print(total_comparison.to_string(index=False))


## 5.2.2 Bottom 3 Objects

bot_df = df.iloc[-3:].copy()

# account for only the 3 top DL objects
aux_GO_full = filt_GO_full[filt_GO_full['Object'].isin(bot_df['Object'])]
aux_DL_full = filt_DL_full[filt_DL_full['Object'].isin(bot_df['Object'])]
aux_GO_obj = filt_GO_obj[filt_GO_obj['Object'].isin(bot_df['Object'])]
aux_DL_obj = filt_DL_obj[filt_DL_obj['Object'].isin(bot_df['Object'])]

aux_GO_full_assess = assess_results(aux_GO_full, 'geometric_only', 'full_scene', out_folder)
aux_DL_full_assess = assess_results(aux_DL_full, 'deep_learning', 'full_scene', out_folder)
aux_GO_obj_assess = assess_results(aux_GO_obj, 'geometric_only', 'single_object', out_folder)
aux_DL_obj_assess = assess_results(aux_DL_obj, 'deep_learning', 'single_object', out_folder)

aux_total_GO_full = aux_GO_full_assess[aux_GO_full_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()
aux_total_GO_obj = aux_GO_obj_assess[aux_GO_obj_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()
aux_total_DL_full = aux_DL_full_assess[aux_DL_full_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()
aux_total_DL_obj = aux_DL_obj_assess[aux_DL_obj_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()

total_comparison = pd.concat([aux_total_GO_full, aux_total_DL_full, aux_total_GO_obj, aux_total_DL_obj])

filename = f"{out_folder}/overall_evaluation_per_scene_totals_bottom_3_on_full.csv"
total_comparison.to_csv(filename, index=False)
print(f'Overall comparison table (totals per pipeline/setting combination) saved at: {filename}')

print('==================== Full Scene - Bottom 3 ====================')
print(total_comparison.to_string(index=False))


### 5.2.3 Middle 3 Objects

mid = len(df) // 2
mid_df = df.iloc[mid-1:mid+2].copy()

# account for only the 3 top DL objects
aux_GO_full = filt_GO_full[filt_GO_full['Object'].isin(mid_df['Object'])]
aux_DL_full = filt_DL_full[filt_DL_full['Object'].isin(mid_df['Object'])]
aux_GO_obj = filt_GO_obj[filt_GO_obj['Object'].isin(mid_df['Object'])]
aux_DL_obj = filt_DL_obj[filt_DL_obj['Object'].isin(mid_df['Object'])]

aux_GO_full_assess = assess_results(aux_GO_full, 'geometric_only', 'full_scene', out_folder)
aux_DL_full_assess = assess_results(aux_DL_full, 'deep_learning', 'full_scene', out_folder)
aux_GO_obj_assess = assess_results(aux_GO_obj, 'geometric_only', 'single_object', out_folder)
aux_DL_obj_assess = assess_results(aux_DL_obj, 'deep_learning', 'single_object', out_folder)

aux_total_GO_full = aux_GO_full_assess[aux_GO_full_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()
aux_total_GO_obj = aux_GO_obj_assess[aux_GO_obj_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()
aux_total_DL_full = aux_DL_full_assess[aux_DL_full_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()
aux_total_DL_obj = aux_DL_obj_assess[aux_DL_obj_assess['Scene'] == 'TOTAL'].drop('Scene', axis=1).copy()

total_comparison = pd.concat([aux_total_GO_full, aux_total_DL_full, aux_total_GO_obj, aux_total_DL_obj])

filename = f"{out_folder}/overall_evaluation_per_scene_totals_middle_3_on_full.csv"
total_comparison.to_csv(filename, index=False)
print(f'Overall comparison table (totals per pipeline/setting combination) saved at: {filename}')

print('==================== Full Scene - Middle 3 ====================')
print(total_comparison.to_string(index=False))
