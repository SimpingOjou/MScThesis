import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def main():
    data_folder = './csv/LR_separated/'
    group_names = ['A', 'B', 'C', 'D', 'E']
    limbs = ['forelimb', 'hindlimb']
    features_to_analyze = ['step height', 'step length', 'step duration', 'vector angle']

    dm = DataManager(data_folder, group_names, limbs, features_to_analyze)
    dm.initialize_data_dict()

    data:dict[str, dict[str, dict[str, list[float]]]] = dm.get_data_dict()

    print(dm.get_feature_names_from_data())
    
    dm.plot_data()

class DataManager:
    def __init__(self, data_folder:str, 
                 group_names:list[str], limbs:list[str],
                 features_to_compare:list[str] = None):
        self.data:dict[str, dict[str, dict[str, list[float]]]] = dict()
        self.feature_names:list[str] = []

        self.data_folder:str = data_folder
        self.group_names:list[str] = group_names
        self.limbs:list[str] = limbs

        self.features_to_compare:list[str] = features_to_compare

    def initialize_data_dict(self)->dict[str, dict[str, dict[str, list[float]]]]:
        data = dict()
        for group in self.group_names:
            if group not in data:
                data[group] = dict()
                data[group][self.limbs[0]] = dict()
                data[group][self.limbs[1]] = dict()

            # Get all CSV files with group in their name
            for limb in self.limbs:
                for file in os.listdir(self.data_folder):
                    if file.endswith('.csv') and group in file and limb in file:
                        file_path = os.path.join(self.data_folder, file)
                        df = pd.read_csv(file_path)
                        for feature in df.columns:
                            if feature not in data[group][limb]:
                                data[group][limb][feature] = dict()
                            data[group][limb][feature] = df[feature].tolist()
        self.data = data

    def get_data_dict(self)->dict[str, dict[str, dict[str, list[float]]]]:
        return self.data

    def get_feature_names_from_data(self)->dict[str, dict[str, dict[str, list[float]]]]:
        first_group = next(iter(self.data.values()))
        first_limb = next(iter(first_group.values()))

        return first_limb.keys()

if __name__ == '__main__':
    main()