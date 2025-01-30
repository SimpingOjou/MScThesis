import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def main():
    data_folder = './csv/LR_separated/'
    batch_names = ['A', 'B', 'C', 'D', 'E']
    limbs = ['forelimb']#, 'hindlimb']
    features_to_analyze = ['step height', 'step length', 'step duration', 'vector angle']

    dm = DataManager(data_folder, batch_names, limbs, features_to_analyze)

    data:dict[str, dict[str, dict[str, list[float]]]] = dm.get_data_dict()
    print(data)
    print(dm.get_feature_names_from_data())

class DataManager:
    def __init__(self, data_folder:str, 
                 batch_names:list[str], limbs:list[str],
                 features_to_compare:list[str] = None):
        self.data:dict[str, dict[str, dict[str, list[float]]]] = dict()
        self.feature_names:list[str] = []

        self.data_folder:str = data_folder
        self.batch_names:list[str] = batch_names
        self.limbs:list[str] = limbs

        self.features_to_compare:list[str] = features_to_compare
        self.use_run:bool = None

        self._initialize_data_dict()

    def _initialize_data_dict(self)->dict[str, dict[str, dict[str, list[float]]]]:
        data = dict()

        for file in os.listdir(self.data_folder):
            if file.endswith('.csv') and any(limb in file for limb in self.limbs):
                file_path = os.path.join(self.data_folder, file)
                df = pd.read_csv(file_path)

                column_list = df.columns.tolist()
                dataset_names = df[column_list[0]].tolist()
                mouse_names = df[column_list[1]].tolist()
                self.use_run = 'run' in column_list[2] if self.use_run is None else self.use_run
                run_numbers = df[column_list[2]].tolist() if self.use_run else None

                for dataset in dataset_names:
                    if dataset not in data:
                        data[dataset] = dict()
                    
                    for mouse in mouse_names:
                        if mouse not in data[dataset]:
                            data[dataset][mouse] = dict()

                        if run_numbers:
                            for run in run_numbers:
                                if run not in data[dataset][mouse]:
                                    data[dataset][mouse][run] = dict()
                                for feature in column_list[3:]:
                                    if feature not in data[dataset][mouse][run]:
                                        data[dataset][mouse][run][feature] = dict()
                                    data[dataset][mouse][run][feature] = df[feature].tolist()
                        else:
                            for feature in column_list[2:]:
                                if feature not in data[dataset][mouse]:
                                    data[dataset][mouse][feature] = dict()
                                data[dataset][mouse][feature] = df[feature].tolist()
        self.data = data

    def get_data_dict(self)->dict[str, dict[str, dict[str, list[float]]]]:
        '''
            Returns the data dictionary

            structure:
            - dataset_name
            - mouse_name
            - run_number | optional
            - feature_name
                - list of values
        '''
        return self.data

    def get_feature_names_from_data(self)->dict[str, dict[str, dict[str, list[float]]]]:
        first_dataset = next(iter(self.data))
        first_mouse = next(iter(self.data[first_dataset]))
        first_run = next(iter(self.data[first_dataset][first_mouse])) if self.use_run else None
        first_feature = self.data[first_dataset][first_mouse][first_run] if self.use_run else self.data[first_dataset][first_mouse]


        return first_feature.keys()

if __name__ == '__main__':
    main()