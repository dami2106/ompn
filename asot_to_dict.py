import numpy as np 
import os 
import glob 
from enum import Enum
import pickle
from gym_psketch.env.craft import Actions

def convert_array_to_enum(arr):
    if arr.ndim != 1:
        raise ValueError("Input array must be 1-dimensional")
    return [Actions(int(val)) for val in arr]

def get_file_names(directory):
    file_list = glob.glob(os.path.join(directory, '*'))
    plain_names = [os.path.splitext(os.path.basename(file))[0] for file in file_list]
    return plain_names

#folder name is like mixed_static/mixed_static_pixels
#features name is like pca_features or symbolic_obs
def convert_data(folder_name):
    directory_path = os.path.join('Traces', folder_name)

    if 'pixels' in folder_name:
        features_name = 'pca_features'
    else:
        features_name = 'symbolic_obs'

    file_names = get_file_names(directory_path + '/' + features_name)

    mapping_file = os.path.join(directory_path, 'mapping/mapping.txt')
    with open(mapping_file, 'r') as f:
        lines = f.readlines()
        mapping = {}
        for line in lines:
            key, value = line.strip().split(' ')
            mapping[key] = value

    #Reverse the mapping
    mapping = {v: int(k) for k, v in mapping.items()}
    print(mapping)

    dataset = {
        'trajs' : [],
        'env_id' : -1,
    }

    for file_name in file_names:
        state = np.load(os.path.join(directory_path, features_name, file_name + '.npy'), allow_pickle=True)
        actions = np.load(os.path.join(directory_path, 'actions', file_name + '.npy'), allow_pickle=True)

        actions = convert_array_to_enum(actions)

        #Load the ground truth text file 
        with open(os.path.join(directory_path, 'groundTruth', file_name), 'r') as f:
            lines = f.readlines()
            ground_truth = [line.strip() for line in lines]

        ground_truth.append(ground_truth[-1])

        #Convert the ground truth to a list of integers
        ground_truth = [mapping[gt] for gt in ground_truth]


        episode = {
            'features' : state,
            'action' : actions,
            'reward' : np.zeros(len(actions)),
            'groundTruth' : ground_truth,
        }
        
        dataset['trajs'].append(episode)

    save_path = os.path.join('dataset', folder_name.split('/')[0])
    os.makedirs(save_path, exist_ok=True)
    
    output_path = os.path.join(save_path, f'{folder_name.split("/")[1]}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)



if __name__ == '__main__':

    #get a list of folders in Traces/
    trace_folders = [f for f in os.listdir('Traces') if os.path.isdir(os.path.join('Traces', f))]
    types = ['pixels', 'symbolic']
    sizes = ['', 'big']
    for folder in trace_folders:
        for type in types:
            for size in sizes:
                data_folder = folder + '/' +(folder + '_' + type) + (('_' + size) if size else '')
                convert_data(data_folder)
                print(f"Converted data for {data_folder} and saved to dataset/{data_folder.split('/')[0]}")
