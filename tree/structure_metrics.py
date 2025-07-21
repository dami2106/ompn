from collections import defaultdict
from typing import Dict, List, Union, Tuple
import hashlib

# Define tree node type - updated for new structure
TreeNode = Dict[str, Union[str, List["TreeNode"]]]

def compute_hierarchy_depth(tree: TreeNode) -> int:
    if "children" not in tree:
        return 1
    return 1 + max(compute_hierarchy_depth(child) for child in tree["children"])

def count_nodes(tree: TreeNode) -> int:
    if "children" not in tree:
        return 1
    return 1 + sum(count_nodes(child) for child in tree["children"])

def compute_branching_factor(tree: TreeNode) -> Tuple[float, int]:
    if "children" not in tree:
        return 0.0, 0
    
    total_branches = 0
    internal_nodes = 0
    max_branching = 0

    def traverse(node):
        nonlocal total_branches, internal_nodes, max_branching
        if "children" in node:
            num_children = len(node["children"])
            total_branches += num_children
            max_branching = max(max_branching, num_children)
            internal_nodes += 1
            for child in node["children"]:
                traverse(child)

    traverse(tree)
    avg_branching = total_branches / internal_nodes if internal_nodes > 0 else 0.0
    return avg_branching, max_branching

def compute_subtree_hash(node: TreeNode) -> str:
    """Hashes the structure and content of the subtree for reuse/modularity metrics."""
    if "children" not in node:
        return f"leaf:{node['symbol']}"
    child_hashes = tuple(compute_subtree_hash(child) for child in node["children"])
    return f"symbol:{node['symbol']}|{hash(child_hashes)}"

def compute_reuse_score(tree: TreeNode) -> float:
    """Reuse score = unique_subtrees / total_subtree_occurrences"""
    counter = defaultdict(int)

    def traverse(node):
        h = compute_subtree_hash(node)
        counter[h] += 1
        if "children" in node:
            for child in node["children"]:
                traverse(child)

    traverse(tree)
    total = sum(counter.values())
    unique = len(counter)
    return unique / total if total > 0 else 1.0

def compute_modularity_score(tree: TreeNode) -> float:
    """Approximate modularity by measuring how many top-level subtrees can stand independently."""
    if "children" not in tree:
        return 1.0
    root_children = tree["children"]
    independent_count = 0
    for child in root_children:
        reused = compute_reuse_score(child)
        if reused > 0.9:  # high reuse means it's likely a standalone, reusable module
            independent_count += 1
    return independent_count / len(root_children) if root_children else 1.0

def compute_structure_metrics(tree: TreeNode) -> Dict[str, Union[int, float]]:
    """Compute various structure metrics for the given tree."""
    depth = compute_hierarchy_depth(tree)
    size = count_nodes(tree)
    avg_branching, max_branching = compute_branching_factor(tree)
    reuse = compute_reuse_score(tree)
    modularity = compute_modularity_score(tree)

    return {
        "depth": depth,
        "size": size,
        "avg_branching": avg_branching,
        "max_branching": max_branching,
        "reuse": reuse,
        "modularity": modularity
    }


def compute_average_structure_metrics(trees: List[TreeNode]) -> Dict[str, float]:
    if not trees:
        return {}

    sums = defaultdict(float)
    count = 0

    for tree in trees:
        metrics = compute_structure_metrics(tree)
        for k, v in metrics.items():
            sums[k] += v
        count += 1

    return {k: (sums[k] / count) for k in sums}

def serialize_tree_ignoring_root(tree: TreeNode) -> str:
    def serialize(node):
        if "children" not in node:
            return f"leaf:{node['symbol']}"
        child_strs = [serialize(child) for child in node["children"]]
        return f"[{','.join(child_strs)}]"
    
    # Skip root symbol
    if "children" not in tree:
        return serialize(tree)
    return serialize({"children": tree["children"]})

def count_unique_trees(trees: List[TreeNode]) -> int:
    seen = set()
    for tree in trees:
        tree_str = serialize_tree_ignoring_root(tree)
        seen.add(tree_str)
    return len(seen)

if __name__ == "__main__":
    import json 
        # Load the tree from my_tree.json
    with open('my_tree.json', 'r') as f:
        tree = json.load(f)
    
    # Compute metrics
    metrics = compute_structure_metrics(tree)
    
    print("Tree structure metrics:")
    print("=" * 30)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    print("\nTree structure:")
    print(json.dumps(tree, indent=2))
