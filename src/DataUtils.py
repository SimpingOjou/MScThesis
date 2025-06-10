import os
import pandas as pd
import numpy as np
import torch

from collections import defaultdict

STEP_PHASE_COLS = ["Step Phase Forelimb", "Step Phase Hindlimb"]
TIME_COLS = ["Time (s)", "Frame"]
POSE_KEYS = ["pose", "X"]

def load_data_from_directory(data_dir: str) -> dict:
    """
    Loads CSV files from the specified directory into a nested dictionary.
    Assumes file naming convention: dataset_group_mouse_direction_run.csv

    Returns:
        dict: Nested dictionary with structure:
              data[dataset_group][mouse_direction][run] = DataFrame
    """
    data = {}
    directory = os.listdir(data_dir)
    for file in directory:
        if file.endswith('.csv'):
            components = file[:-4].split('_')  # Remove .csv before splitting
            if len(components) != 5:
                continue  # Skip files not matching the convention

            dataset, group, mouse, direction, run = components

            datagroup = f"{dataset}_{group}"
            if datagroup not in data:
                data[datagroup] = {}

            mouse_direction = f"{mouse}_{direction}"
            if mouse_direction not in data[datagroup]:
                data[datagroup][mouse_direction] = {}

            data[datagroup][mouse_direction][run] = pd.read_csv(
                os.path.join(data_dir, file), index_col=0
            )
    return data

def segment_steps_by_phase(df, phase_col="phase"):
    """
    Segments a DataFrame into steps based on the specified phase column. Drops the phase column after segmentation.
    """
    phases = df[phase_col]

    # 1. Find where phase changes
    changes = phases != phases.shift()
    change_points = df.index[changes]

    # 2. Get the phases at these change points
    change_phases = phases.loc[change_points].reset_index(drop=True)

    # 3. Identify "stance" → ... → "swing" → "stance" sequences
    segments = []
    indexes : list[dict[str, tuple[int, int]]] = []
    i = 0
    while i < len(change_phases) - 2: # iterate through change phases to look for "stance"
        if change_phases[i] == "stance":
            swing_found = False

            for j in range(i+1, len(change_phases)): # iterate through next change phases to look for the next "swing"
                if not swing_found and change_phases[j] == "swing":
                    swing_idx = j
                    swing_found = True
                    continue
                if swing_found:
                    start_idx = change_points[i]
                    end_idx = change_points[j] - 1
                    segment = df.loc[start_idx:end_idx].drop(columns=STEP_PHASE_COLS, errors='ignore')
                    segment.reset_index(drop=True, inplace=True)
                    
                    ## Reset x to 0
                    pose_cols = [col for col in segment.columns if all(key in col.lower() for key in POSE_KEYS)]
                    segment[pose_cols] = segment[pose_cols] - segment[pose_cols].iloc[0]
                    segments.append(segment)

                    indexes.append({
                        "stance" : (start_idx, swing_idx -1),
                        "swing" : (swing_idx, end_idx)
                    })
                    i = j  # skip ahead to end of swing
                    break
            else: # else clause of for loop, no swing found
                # no closing stance; go to end
                start_idx = change_points[i]
                segment = df.loc[start_idx:].drop(columns=STEP_PHASE_COLS, errors='ignore')
                segment.reset_index(drop=True, inplace=True)

                assert swing_found, "Swing not found in the above for loop"

                ## Reset x to 0
                pose_cols = [col for col in segment.columns if all(key in col.lower() for key in POSE_KEYS)]
                segment[pose_cols] = segment[pose_cols] - segment[pose_cols].iloc[0]
                segments.append(segment)

                indexes.append({
                        "stance" : (start_idx, swing_idx -1),
                        "swing" : (swing_idx, end_idx)
                    })
                break
        i += 1

    return segments

def segment_all_steps(data):
    """
    Segments all steps in the provided data dictionary by phase for both hindlimb and forelimb.
    Arguments:
        data (dict): Nested dictionary containing DataFrames for each group, mouse, and run.
    Returns:
        tuple: Two lists containing segmented hindlimb and forelimb steps.
            - step: step DataFrame with columns for each pose and other features.
            - group: Group name.
            - mouse: Mouse identifier.
            - run: Run identifier.
    """
    segmented_hindsteps : list[dict[str, pd.DataFrame | str]] = []
    segmented_foresteps : list[dict[str, pd.DataFrame | str]] = []
    for group in data:
        for mouse in data[group]:
            for run in data[group][mouse]:
                df = data[group][mouse][run]
                to_drop = TIME_COLS
                df = df.drop(columns=to_drop, errors='ignore')
                hindsteps = segment_steps_by_phase(df, phase_col=STEP_PHASE_COLS[1])  # Hindlimb phase
                for step_df in hindsteps:
                    segmented_hindsteps.append({
                        "step": step_df,
                        "group": group,
                        "mouse": mouse,
                        "run": run
                    })
                foresteps = segment_steps_by_phase(df, phase_col=STEP_PHASE_COLS[0])  # Forelimb phase
                for step_df in foresteps:
                    segmented_foresteps.append({
                        "step": step_df,
                        "group": group,
                        "mouse": mouse,
                        "run": run
                    })
    return segmented_hindsteps, segmented_foresteps

def steps_to_tensor(step_dicts, scaler):
    """
        Convert a list of step dictionaries to a padded tensor of shape (num_steps, max_length, num_features).

        Returns:
            - A tensor of shape (num_steps, max_length, num_features) containing the scaled step data (B, T, F).
            - A tensor of lengths for each step indicating the actual length of each step (B).
    """
    step_arrays = [scaler.transform(sd["step"].values) for sd in step_dicts]
    lengths = [len(step) for step in step_arrays]
    max_len = max(lengths)
    dim = step_arrays[0].shape[1]

    padded = np.zeros((len(step_arrays), max_len, dim), dtype=np.float32)
    for i, arr in enumerate(step_arrays):
        padded[i, :len(arr)] = arr

    return torch.tensor(padded), torch.tensor(lengths)

def average_run_features_by_mouse(segmented_steps_dicts)->list[dict[str, pd.DataFrame]]:
    """
    Computes masked average over all steps for each mouse.
    
    Returns:
        dict[str, pd.DataFrame|str]: Mouse ID -> averaged time series (T_max, F)
    """
    mouse_steps = defaultdict(lambda: defaultdict(list))
    for entry in segmented_steps_dicts:
        group = entry["group"]
        mouse = entry["mouse"]
        step_df = entry["step"]
        mouse_steps[group][mouse].append(step_df)

    averaged_by_mouse = []
    for group, mouse_dict in mouse_steps.items():
        for mouse, steps in mouse_dict.items():
            # Determine max length
            max_len = max(len(step) for step in steps)
            num_feats = steps[0].shape[1]

            # Initialize padded tensor and mask
            padded = np.full((len(steps), max_len, num_feats), np.nan, dtype=np.float32)
            for i, step in enumerate(steps):
                length = len(step)
                padded[i, :length, :] = step.values

            # Masked mean (ignores NaNs)
            mean_trace = np.nanmean(padded, axis=0)
            averaged_step_df = pd.DataFrame(mean_trace, columns=steps[0].columns)
            averaged_by_mouse.append({
                "step": averaged_step_df,
                "group": group,
                "mouse": mouse,
            })

    return averaged_by_mouse