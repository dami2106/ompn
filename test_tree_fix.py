#!/usr/bin/env python3
"""
Test script to verify the new tree generation approach using p_avg values
"""
import numpy as np
import torch
from gym_psketch.visualize import distance2ctree, tree_to_str

def test_p_avg_tree_generation():
    print("Testing p_avg-based tree generation...")
    
    # Simulate p_hats (model probability outputs)
    seq_len = 10
    nb_slots = 4
    p_hats = torch.randn(seq_len, 1, nb_slots + 1)
    p_hats = torch.softmax(p_hats, dim=-1)  # Make them probabilities
    
    # Calculate p_avg like in the original approach
    p_vals = torch.arange(nb_slots + 1)
    p_avg = (p_vals * p_hats.squeeze(1)).sum(-1)  # [seq_len]
    
    print(f"p_avg values: {p_avg.numpy()}")
    
    # Create depth information from p_avg
    depths = p_avg[:-1]  # Remove last element
    depths = (depths - depths.min()) / (depths.max() - depths.min() + 1e-8)
    depths = np.digitize(depths.numpy(), np.linspace(0, 1, 5))
    
    print(f"Depth values: {depths}")
    
    # Convert actions to vocabulary strings
    actions = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    action_strs = [f"act_{idx}" for idx in actions]
    
    # Use distance2ctree with depth information
    tree = distance2ctree(depths, action_strs, binary=False)
    tree_str = tree_to_str(tree)
    
    print(f"Tree structure: {tree}")
    print(f"Tree string: {tree_str}")
    
    # Test with more structured p_avg values
    print("\n=== Testing with structured p_avg values ===")
    structured_p_avg = torch.tensor([0.1, 0.2, 0.8, 0.9, 0.3, 0.4, 0.7, 0.8, 0.2, 0.1])
    structured_depths = structured_p_avg[:-1]
    structured_depths = (structured_depths - structured_depths.min()) / (structured_depths.max() - structured_depths.min() + 1e-8)
    structured_depths = np.digitize(structured_depths.numpy(), np.linspace(0, 1, 5))
    
    print(f"Structured p_avg: {structured_p_avg.numpy()}")
    print(f"Structured depths: {structured_depths}")
    
    structured_tree = distance2ctree(structured_depths, action_strs, binary=False)
    structured_tree_str = tree_to_str(structured_tree)
    
    print(f"Structured tree: {structured_tree}")
    print(f"Structured tree string: {structured_tree_str}")

if __name__ == "__main__":
    test_p_avg_tree_generation() 