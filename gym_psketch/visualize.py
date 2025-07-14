import numpy as np

from gym_psketch import ACTION_VOCAB

CHARS = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]


def visual(val, max_val):
    val = np.clip(val, 0, max_val)
    if abs(val) == max_val:
        step = len(CHARS) - 1
    else:
        step = int(abs(float(val) / max_val) * len(CHARS))
    colourstart = ""
    colourend = ""
    if val < 0:
        colourstart, colourend = '\033[90m', '\033[0m'
    return colourstart + CHARS[step] + colourend


def idxpos2tree(idxs, pos):
    viz = pos.transpose(1, 0).cpu().numpy()
    string = np.array2string(viz, formatter={'float_kind': lambda x: visual(x, 1)},
                             max_line_width=5000)
    lines = [line[2:-1].replace(']', '') for line in string.split('\n')]
    lines.append(' '.join([ACTION_VOCAB[idx] for idx in idxs]))
    return '\n'.join(lines)


def distance2ctree(depth, sen, binary=False):
    assert len(depth) == len(sen) - 1
    if len(sen) == 1:
        parse_tree = sen[0]
    else:
        max_depth = max(depth)
        parse_tree = []
        sub_depth = []
        sub_sen = []
        for d, w in zip(depth, sen[:-1]):
            sub_sen.append(w)
            if d == max_depth:
                parse_tree.append(distance2ctree(sub_depth, sub_sen, binary))
                sub_depth = []
                sub_sen = []
            else:
                sub_depth.append(d)
        sub_sen.append(sen[-1])
        parse_tree.append(distance2ctree(sub_depth, sub_sen, binary))
    if len(parse_tree) > 2 and binary:
        bin_tree = parse_tree.pop(-1)
        while len(parse_tree) > 0:
            bin_tree = [parse_tree.pop(-1), bin_tree]
        return bin_tree
    return parse_tree


def tree_to_str(parse_tree):
    if isinstance(parse_tree, list):
        res = [tree_to_str(ele) for ele in parse_tree]
        return '(' + ' '.join(res) + ')'
    elif isinstance(parse_tree, str):
        return parse_tree


def predicted_data_to_tree(actions, boundaries, subtask_seq, binary=False):
    """
    Convert predicted segmentation data to a tree representation.
    
    Args:
        actions: List/array of action indices
        boundaries: List of boundary positions 
        subtask_seq: List of subtask labels for each segment
        binary: Whether to create binary tree structure
        
    Returns:
        Tree structure that can be converted to string with tree_to_str()
    """
    if len(actions) == 0:
        return []
    
    # Convert actions to vocabulary strings
    action_strs = [ACTION_VOCAB[idx] if idx < len(ACTION_VOCAB) else f"act_{idx}" for idx in actions]
    
    if len(boundaries) == 0 or len(subtask_seq) == 0:
        # No segmentation, return flat action sequence
        return action_strs
    
    # Create segments based on boundaries
    segments = []
    start = 0
    
    for i, boundary in enumerate(boundaries):
        if boundary > len(action_strs):
            boundary = len(action_strs)
        
        segment_actions = action_strs[start:boundary]
        if len(segment_actions) > 0:
            # Create subtask representation
            subtask_label = f"subtask_{subtask_seq[i]}" if i < len(subtask_seq) else f"subtask_{i}"
            
            if len(segment_actions) == 1:
                # Single action segment
                segments.append([subtask_label, segment_actions[0]])
            else:
                # Multiple actions in segment
                segments.append([subtask_label] + segment_actions)
        
        start = boundary
    
    # Handle any remaining actions after last boundary
    if start < len(action_strs):
        remaining_actions = action_strs[start:]
        subtask_idx = len(boundaries)
        subtask_label = f"subtask_{subtask_seq[subtask_idx]}" if subtask_idx < len(subtask_seq) else f"subtask_{subtask_idx}"
        
        if len(remaining_actions) == 1:
            segments.append([subtask_label, remaining_actions[0]])
        else:
            segments.append([subtask_label] + remaining_actions)
    
    # If binary tree requested, convert to binary structure
    if binary and len(segments) > 2:
        bin_tree = segments.pop(-1)
        while len(segments) > 0:
            bin_tree = [segments.pop(-1), bin_tree]
        return bin_tree
    
    return segments


def boundaries_to_depth(boundaries, seq_length):
    """
    Convert boundary positions to depth representation for use with distance2ctree.
    
    Args:
        boundaries: List of boundary positions
        seq_length: Total length of the sequence
        
    Returns:
        List of depths for each position (length = seq_length - 1)
    """
    if seq_length <= 1:
        return []
    
    depths = []
    boundary_set = set(boundaries)
    
    for i in range(seq_length - 1):
        if (i + 1) in boundary_set:
            depths.append(1)  # High depth at boundaries
        else:
            depths.append(0)  # Low depth within segments
    
    return depths


def predicted_data_to_hierarchical_tree(actions, boundaries, binary=False):
    """
    Convert predicted data to hierarchical tree using the existing distance2ctree function.
    
    Args:
        actions: List/array of action indices
        boundaries: List of boundary positions
        binary: Whether to create binary tree structure
        
    Returns:
        Tree structure created by distance2ctree
    """
    if len(actions) == 0:
        return []
    
    # Convert actions to vocabulary strings
    action_strs = [ACTION_VOCAB[idx] if idx < len(ACTION_VOCAB) else f"act_{idx}" for idx in actions]
    
    if len(boundaries) == 0:
        # No boundaries, return flat structure
        if len(action_strs) == 1:
            return action_strs[0]
        return action_strs
    
    # Generate depth array from boundaries
    depths = boundaries_to_depth(boundaries, len(actions))
    
    if len(depths) == 0:
        return action_strs[0] if len(action_strs) == 1 else action_strs
    
    # Use existing distance2ctree function
    return distance2ctree(depths, action_strs, binary=binary)
