import pickle as pkl 
import argparse
import os



pkl_files = [f for f in os.listdir('dataset') if f.endswith('.pkl')]

for file in pkl_files:
    #Load in the datafile from dataset
    with open('dataset/' + file, 'rb') as f:
        data = pkl.load(f)


    print("KEYS: ", data.keys())
    print("Dataset Length: ", len(data['trajs']))

    lengths = []

    for ep in data['trajs']:
        state = ep['features']
        action = ep['action']
        ground_truth = ep['groundTruth']

        assert len(state) == len(action), f"State and action lengths do not match: {len(state)} != {len(action)}"
        assert len(state) == len(ground_truth), f"State and ground truth lengths do not match: {len(state)} != {len(ground_truth)}"
        assert len(action) == len(ground_truth), f"Action and ground truth lengths do not match: {len(action)} != {len(ground_truth)}"

        lengths.append(len(state))

    print("Average Length: ", sum(lengths)/len(lengths))
    print("Verified: ", file)
    print("--------------------------------------------------")