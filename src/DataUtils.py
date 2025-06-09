import os
import pandas as pd

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