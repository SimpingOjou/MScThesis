from os import PathLike
from os.path import join
import pandas as pd

def fuse_features(fore_path:PathLike, hind_path:PathLike, output_path:PathLike,
                  fore_keyword:str='forelimb', hind_keyword:str='hindlimb',
                  id_columns:list=['Dataset', 'Mouse']):
    """
    Fuse features from forelimb and hindlimb data into a single CSV file.
    
    Parameters:
    - fore_path: Path to the forelimb features CSV file.
    - hind_path: Path to the hindlimb features CSV file.
    - output_path: Path where the fused features CSV file will be saved.
    """    
    # Load forelimb and hindlimb data
    fore_data = pd.read_csv(fore_path)
    hind_data = pd.read_csv(hind_path)

    # Ensure both datasets have the same number of rows
    if len(fore_data) != len(hind_data):
        raise ValueError("Forelimb and hindlimb datasets must have the same number of rows.")

    rows_fore_in_hind = fore_data[id_columns].isin(hind_data[id_columns]).all(axis=1)
    rows_hind_in_fore = hind_data[id_columns].isin(fore_data[id_columns]).all(axis=1)

    print(f"Rows of forelimb in hindlimb: {rows_fore_in_hind.sum()} / {len(fore_data)}")
    print(f"Rows of hindlimb in forelimb: {rows_hind_in_fore.sum()} / {len(hind_data)}")

    rows_not_in_hind = ~rows_fore_in_hind
    rows_not_in_fore = ~rows_hind_in_fore

    if rows_not_in_hind.any() or rows_not_in_fore.any():
        raise ValueError("Forelimb and hindlimb datasets do not match on the ID columns. "
                         "Ensure that both datasets have the same 'Dataset' and 'Mouse' identifiers.")

    # Combine the datasets
    fused_data = pd.merge(fore_data, hind_data, on=id_columns, how='inner', suffixes=(f' - {fore_keyword}', f' - {hind_keyword}'))

    # Save the fused data to a new CSV file
    fused_data.to_csv(output_path, index=False)
    print(f"Fused features saved to {output_path}")

if __name__ == "__main__":
    data_folder = './csv/Epilepsy/'
    fore_file = 'Post_forelimb_mouse_features_2025-05-24_13-29-36.csv'
    hind_file = 'Post_hindlimb_mouse_features_2025-05-24_13-29-36.csv'
    output_file = 'Post_fused_mouse_features.csv'

    fuse_features(
        fore_path=join(data_folder, fore_file),
        hind_path=join(data_folder, hind_file),
        output_path=join(data_folder, output_file),
    )