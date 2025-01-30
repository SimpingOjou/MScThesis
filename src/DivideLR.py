import os
from tqdm import tqdm
from multiprocessing import Pool

class MultiprocessCopy():
    def __init__(self, batches, datasets,
                 input_path, output_path,
                 sideview_key, ventralview_key,left_right_key, 
                 data_format, video_format,
                 output_sideview_folder, output_ventralview_folder, output_video_folder):
        self.input_path = input_path
        self.output_path = output_path
        
        self.batches = batches
        self.datasets = datasets

        self.sideview_key = sideview_key
        self.ventralview_key = ventralview_key
        self.left_right_key = left_right_key
        self.data_format = data_format
        self.video_format = video_format
        self.output_sideview_folder = output_sideview_folder
        self.output_ventralview_folder = output_ventralview_folder
        self.output_video_folder = output_video_folder

    def copy_data(self, data):
        file, ext = data

        input_file = file
        file = file.split('/')[-1]
        output_batch = None
        output_dataset = None
        output_view = None

        # Separate the filename by _
        element_list = set(file.split('_'))
    
        current_batch = [key for key in element_list if key in self.batches]
        current_batch = current_batch[0] if len(current_batch) > 0 else None
        current_dataset = [key for key in element_list if key in self.datasets]
        current_dataset = current_dataset[0] if len(current_dataset) > 0 else None

        ## Batch 
        if current_batch is None:
            print(f'Batch: {current_batch}')
            return False, file+ext
        
        if any(self.left_right_key[0] in element for element in element_list):
            output_batch = current_batch + '_left'
        elif any(self.left_right_key[1] in element for element in element_list):
            output_batch = current_batch + '_right'
        
        ## Datast
        if current_dataset is None:
            print(f'Dataset: {current_dataset}')
            return False, file+ext

        if any(current_dataset in element for element in element_list):
            output_dataset = current_dataset

        ## view and format
        if self.data_format == ext:
            if any(self.sideview_key in element for element in element_list):
                output_view = self.output_sideview_folder
            elif any(self.ventralview_key in element for element in element_list):
                output_view = self.output_ventralview_folder

            if output_view is None:
                print(f'View: {output_view}')
                return False, file+ext
        
        if ext == self.video_format: 
            output_view = self.output_video_folder

            if output_batch is None or output_dataset is None or output_view is None:
                print(f'View: {output_view}')
                return False, file+ext

        output_folder = os.path.join(self.output_path, output_batch, output_dataset, output_view)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        input_file = input_file + ext
        output_file = os.path.join(output_folder, file + ext)

        try:
            os.system(f'cp {input_file} {os.path.join(output_file)}')
        except Exception as e:
            print(f'Error: {e}')
            print(f'File: {file + ext} not copied!')

        return True, file+ext

def main():
    input_folder = './Data/by_group'
    input_path = os.path.join(os.getcwd(), input_folder)

    output_folder = 'Data/Export/DividedLR'
    output_path = os.path.join(os.getcwd(), output_folder)

    data_format = '.csv'
    video_format = '.mp4'
    sideview_key = 'sideview'
    ventralview_key = 'ventralview'
    left_right_key = ('left', 'right')
    output_sideview_folder = 'side_view_analysis'
    output_ventralview_folder = 'ventral_view_analysis'
    output_video_folder = 'Video'

    copy_files(get_all_files(input_path), output_path, input_path,
               sideview_key, ventralview_key,
               left_right_key, data_format, video_format,
               output_sideview_folder, output_ventralview_folder, output_video_folder)

def get_all_files(path)->list:
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def get_all_batches(path)->list:
    dir_list = [dir for dir in next(os.walk(path))[1]]
    return sorted(dir_list)

def get_all_datasets(path)->list:
    batches = get_all_batches(path)
    dataset_list = []
    for batch in batches:
        batch_path = os.path.join(path, batch)
        for dir in next(os.walk(batch_path))[1]:
            dataset_list.append(dir)

    return sorted(set(dataset_list))

def copy_files(file_list, output_path, input_path,
               sideview_key, ventralview_key,left_right_key, 
               data_format, video_format,
               output_sideview_folder, output_ventralview_folder, output_video_folder)->None:
    batches = get_all_batches(input_path)
    datasets = get_all_datasets(input_path)

    file_list = [file for file in file_list]
    coupled_list = [os.path.splitext(file) for file in file_list]

    print('-'*20)
    print(f'Found batches: {batches}.')
    print(f'Found datasets: {datasets}.')
    print(f'Found {len(file_list)} files.')
    print('-'*20)
    print('Continue? (Y/n)')
    choice = input()
    if choice.lower() != 'y' and choice.lower() != '':
        return
    print('-'*20)

    function_input = (batches, datasets, input_path, output_path,
                        sideview_key, ventralview_key, left_right_key, 
                        data_format, video_format,
                        output_sideview_folder, output_ventralview_folder, output_video_folder)
    
    m_copy = MultiprocessCopy(*function_input)
    p = Pool()

    for no_error, filename in tqdm(p.imap(m_copy.copy_data, coupled_list), desc='Copying files', unit='it', total=len(coupled_list)):
        if not no_error:
            print('Error copying file: ', filename)

    p.close()
    p.join()

    print('-'*20)
    print('Files copied!')
    print('-'*20)

if __name__ == '__main__':
    main()