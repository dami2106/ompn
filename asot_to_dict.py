import numpy as np 
import os 
import glob 
from enum import Enum

#Improt the actions enum from 
from gym_psketch.env.craft import Actions

def convert_array_to_enum(arr):
    if arr.ndim != 1:
        raise ValueError("Input array must be 1-dimensional")
    return [Actions(int(val)) for val in arr]

def get_file_names(directory):
    file_list = glob.glob(os.path.join(directory, '*'))
    plain_names = [os.path.splitext(os.path.basename(file))[0] for file in file_list]
    return plain_names

# Example usage
directory_path = 'data/stone_pick_static_symbolic_big'
file_names = get_file_names(directory_path + '/symbolic_obs')


#Load the mapping.txt
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
    state = np.load(os.path.join(directory_path, 'symbolic_obs', file_name + '.npy'), allow_pickle=True)
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


#Save the dataset to a pkl file 
import pickle
output_path = 'dataset/stone_pick_dataset.pkl'  # You can change the path/filename as needed
with open(output_path, 'wb') as f:
    pickle.dump(dataset, f)