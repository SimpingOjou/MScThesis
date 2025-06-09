import os
import pandas as pd
import numpy as np
import torch

STEP_PHASE_COLS = ["Step Phase Forelimb", "Step Phase Hindlimb"]
TIME_COLS = ["Time (s)", "Frame"]
POSE_KEY = "pose"

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
    i = 0
    while i < len(change_phases) - 2:
        if change_phases[i] == "stance":
            swing_found = False
            for j in range(i+1, len(change_phases)):
                if change_phases[j] == "swing":
                    swing_found = True
                elif swing_found and change_phases[j] == "stance":
                    # # Skip first and last segment
                    # if i == 0 or j == len(change_phases) - 1:
                    #     i = j - 1
                    #     break
                    start_idx = change_points[i]
                    end_idx = change_points[j] - 1
                    segment = df.loc[start_idx:end_idx].drop(columns=STEP_PHASE_COLS, errors='ignore')
                    
                    ## Reset x to 0
                    pose_cols = [col for col in segment.columns if POSE_KEY in col]
                    segment[pose_cols] = segment[pose_cols] - segment[pose_cols].iloc[0]
                    segments.append(segment)
                    i = j - 1  # skip ahead
                    break
            else:
                # no closing stance; go to end
                start_idx = change_points[i]
                segment = df.loc[start_idx:].drop(columns=STEP_PHASE_COLS, errors='ignore')

                ## Reset x to 0
                pose_cols = [col for col in segment.columns if POSE_KEY in col.lower()]
                segment[pose_cols] = segment[pose_cols] - segment[pose_cols].iloc[0]
                segments.append(segment)
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
    segmented_hindsteps = []
    segmented_foresteps = []
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