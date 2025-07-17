import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform

from metrics import pred_to_gt_match, filter_exclusions

import json
import re
from graphviz import Digraph

def plot_segmentation_gt(gt, pred, mask, gt_uniq=None, pred_to_gt=None, name='', comparison_name = 'CMPLE'):
    colors = {}

    # pred_, gt_ = filter_exclusions(pred[mask].cpu().numpy(), gt[mask].cpu().numpy())
    pred_ = pred[mask].cpu().numpy()
    gt_ = gt[mask].cpu().numpy()
    if pred_to_gt is None:
        pred_opt, gt_opt = pred_to_gt_match(pred_, gt_)
    else:
        pred_opt, gt_opt = zip(*pred_to_gt.items())
    for pr_lab, gt_lab in zip(pred_opt, gt_opt):
        pred_[pred_ == pr_lab] = gt_lab
    n_frames = len(pred_)

    # add colors for predictions which do not match to a gt class

    if gt_uniq is None:
        gt_uniq = np.unique(gt_)
    pred_not_matched = np.setdiff1d(pred_opt, gt_uniq)
    if len(pred_not_matched) > 0:
        gt_uniq = np.concatenate((gt_uniq, pred_not_matched))

    n_class = len(gt_uniq)
    if n_class <= 20:
        cmap = plt.get_cmap('tab20')
    else:  # up to 40 classes
        cmap1 = plt.get_cmap('tab20')
        cmap2 = plt.get_cmap('tab20b')
        cmap = lambda x: cmap1(round(x * n_class / 20., 2)) if x <= 19. / n_class else cmap2(round((x - 20 / n_class) * n_class / 20, 2))

    for i, label in enumerate(gt_uniq):
        if label == -1:
            colors[label] = (0, 0, 0)
        else:
            colors[label] = cmap(i / n_class)

    fig = plt.figure(figsize=(16, 4))
    plt.axis('off')
    plt.title(name, fontsize=45, pad=20)

    # plot gt segmentation

    ax = fig.add_subplot(2, 1, 1)
    ax.set_ylabel('TRUTH', fontsize=22, rotation=0, labelpad=40, verticalalignment='center')
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    gt_segment_boundaries = np.where(gt_[1:] - gt_[:-1])[0] + 1
    gt_segment_boundaries = np.concatenate(([0], gt_segment_boundaries, [len(gt_)]))

    for start, end in zip(gt_segment_boundaries[:-1], gt_segment_boundaries[1:]):
        label = gt_[start]
        ax.axvspan(start / n_frames, end / n_frames, facecolor=colors[label], alpha=1.0)
        ax.axvline(start / n_frames, color='black', linewidth=3)
        ax.axvline(end / n_frames, color='black', linewidth=3)

    # plot predicted segmentation after matching to gt labels w/Hungarian

    ax = fig.add_subplot(2, 1, 2)
    ax.set_ylabel(f'{comparison_name}', fontsize=22, rotation=0, labelpad=60, verticalalignment='center')
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    pred_segment_boundaries = np.where(pred_[1:] - pred_[:-1])[0] + 1
    pred_segment_boundaries = np.concatenate(([0], pred_segment_boundaries, [len(pred_)]))

    for start, end in zip(pred_segment_boundaries[:-1], pred_segment_boundaries[1:]):
        label = pred_[start]
        ax.axvspan(start / n_frames, end / n_frames, facecolor=colors[label], alpha=1.0)
        ax.axvline(start / n_frames, color='black', linewidth=3)
        ax.axvline(end / n_frames, color='black', linewidth=3)

    fig.tight_layout()
    return fig


def plot_segmentation(pred, mask, name=''):
    colors = {}
    cmap = plt.get_cmap('tab20')
    uniq = np.unique(pred[mask].cpu().numpy())
    n_frames = len(pred)

    # add colors for predictions which do not match to a gt class

    for i, label in enumerate(uniq):
        if label == -1:
            colors[label] = (0, 0, 0)
        else:
            colors[label] = cmap(i / len(uniq))

    fig = plt.figure(figsize=(16, 2))
    plt.axis('off')
    plt.title(name, fontsize=30, pad=20)

    # plot gt segmentation

    ax = fig.add_subplot(1, 1, 1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    pred_segment_boundaries = np.where(pred[mask].cpu().numpy()[1:] - pred[mask].cpu().numpy()[:-1])[0] + 1
    pred_segment_boundaries = np.concatenate(([0], pred_segment_boundaries, [len(pred)]))

    for start, end in zip(pred_segment_boundaries[:-1], pred_segment_boundaries[1:]):
        label = pred[mask].cpu().numpy()[start]
        ax.axvspan(start / n_frames, end / n_frames, facecolor=colors[label], alpha=1.0)
        ax.axvline(start / n_frames, color='black', linewidth=3)
        ax.axvline(end / n_frames, color='black', linewidth=3)

    fig.tight_layout()
    return fig


def plot_matrix(mat, gt=None, colorbar=True, title=None, figsize=(10, 5), ylabel=None, xlabel=None):

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot1 = ax.matshow(mat)
    if gt is not None: # plot gt segment boundaries
        gt_change = np.where((np.diff(gt) != 0))[0] + 1
        for ch in gt_change:
            ax.axvline(ch, color='red')
    if colorbar:
        plt.colorbar(plot1, ax=ax)
    if title:
        ax.set_title(f'{title}')
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=36)
        ax.tick_params(axis='x', labelsize=24)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=36)
        ax.tick_params(axis='y', labelsize=24)
    ax.set_aspect('auto')
    fig.tight_layout()
    return fig


def parse_tree_string(s):
    s = s.strip()
    idx = 0
    counter = {'n': 0}

    def skip_ws():
        nonlocal idx
        while idx < len(s) and s[idx].isspace():
            idx += 1

    def parse_node(is_root=False):
        nonlocal idx
        assert s[idx] == '(', f"Expected '(', got {s[idx:]}"
        idx += 1
        vals = []
        kids = []
        while True:
            skip_ws()
            if idx >= len(s):
                raise ValueError("Unmatched '('")
            if s[idx] == '(':
                kids.append(parse_node(is_root=False))
            elif s[idx] == ')':
                idx += 1
                break
            else:
                # parse an integer
                m = re.match(r'[+-]?\d+', s[idx:])
                if not m:
                    raise ValueError(f"Invalid token at {s[idx:]}")
                token = m.group(0)
                vals.append(int(token))
                idx += len(token)

        # choose a label
        if kids and not vals:
            if is_root:
                label = 'root'
            else:
                counter['n'] += 1
                label = f"N{counter['n']}"
        elif vals:
            label = " ".join(str(v) for v in vals)
        else:
            # empty node?  Treat like a pure‑children node.
            if is_root:
                label = 'root'
            else:
                counter['n'] += 1
                label = f"N{counter['n']}"

        node = {'symbol': label}
        if kids:
            node['children'] = kids
        return node

    # kick off the parse, telling the top level it's the root
    skip_ws()
    if not s or s[0] != '(':
        raise ValueError("Input must start with '('")
    tree = parse_node(is_root=True)
    skip_ws()
    if idx != len(s):
        raise ValueError(f"Extra text after tree: {s[idx:]}")
    return tree


def save_tree_to_json(tree_dict, filename):
    """
    Save a parsed tree dict to a JSON file.

    :param tree_dict: The nested dict returned by parse_tree_string()
    :param filename:   Path to write, e.g. "my_tree.json"
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(tree_dict, f, indent=2, ensure_ascii=False)
    # print(f"Tree written to {filename}")


def visualize_tree(tree_dict, filename_base, fmt="png", root_label=None, label_mapping=None):
    """
    tree_dict: nested dict with keys
      - 'production' → int
      - OR 'symbol' → str
      and optional 'children': [ … ]
    filename_base: e.g. "seq_tree_0"  (no extension)
    fmt: "png" or "pdf"
    root_label: if given, use this string as the label for the very top node
    label_mapping: optional dict mapping grammar labels to readable strings
    """
    dot = Digraph(format=fmt)
    dot.attr(rankdir="TB")     # top→bottom
    dot.attr("node", shape="oval", fontsize="12")

    def recurse(node, parent_id=None, is_root=False):
        nid = str(id(node))
        # if this is the root of the entire tree and a custom label was supplied, use it:
        if is_root and root_label is not None:
            label = root_label
        else:
            if "production" in node:
                label = f"R{node['production']}"
            else:
                # Use mapping if available, otherwise use original symbol
                label = label_mapping.get(node["symbol"], node["symbol"]) if label_mapping else node["symbol"]

        dot.node(nid, label)
        if parent_id is not None:
            dot.edge(parent_id, nid)

        for child in node.get("children", []):
            recurse(child, nid)

    # kick off recursion marking the first call as the root
    recurse(tree_dict, parent_id=None, is_root=True)

    outpath = dot.render(filename_base, cleanup=True)
    # print(f"Written {outpath}")
