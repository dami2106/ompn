import torch
# import gym
# import gym_psketch
from gym_psketch import DictList, ID2SKETCHS
from gym_psketch import DictList
import compile
from gym_psketch.evaluate import f1, get_subtask_seq, get_ta_lines
from absl import flags, logging
import numpy as np
import os
from tensorboardX import SummaryWriter
from sklearn.cluster import KMeans
from metrics import *
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from visualisation import plot_segmentation_gt
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import copy

FLAGS = flags.FLAGS
flags.DEFINE_integer('compile_train_steps', default=4000, help='train steps')
flags.DEFINE_integer('compile_eval_freq', default=20, help='evaluation frequency')
flags.DEFINE_float('compile_lr', default=0.001, help='learning rate')
flags.DEFINE_integer('compile_batch_size', default=256, help='learning rate')
flags.DEFINE_integer('compile_max_segs', default=7, help='num of segment')
flags.DEFINE_integer('compile_skills', default=5, help='num of skills')
flags.DEFINE_float('compile_beta_z', default=0.1, help='weight Z kl loss')
flags.DEFINE_float('compile_beta_b', default=0.1, help='weight b kl loss')
flags.DEFINE_float('compile_prior_rate', default=3, help='possion distribution. avg length of each seg')
flags.DEFINE_enum('compile_latent', enum_values=['gaussian', 'concrete'], default='gaussian',
                  help='Latent type')
flags.DEFINE_integer('compile_state_size', default=1087, help='State Space size')
flags.DEFINE_integer('compile_action_size', default=16, help='Action Space size')


def get_subtask_ordering(truths, boundaries):
    segment_labels = []
    start = 0
    for end in boundaries:
        segment = truths[start:end]
        if len(segment) == 0:
            continue
        # Use the first label (or majority if needed)
        label = segment[0]  # or np.bincount(segment).argmax()
        segment_labels.append(label)
        start = end
    return segment_labels

def main(training_folder):
    print('Start compile...')
    n_feature, action_size = FLAGS.compile_state_size, FLAGS.compile_action_size
    model = compile.CompILE(vec_size=n_feature,
                            hidden_size=FLAGS.hidden_size,
                            action_size=action_size,
                            env_arch=FLAGS.env_arch,
                            max_num_segments=FLAGS.compile_max_segs,
                            latent_dist=FLAGS.compile_latent,
                            beta_b=FLAGS.compile_beta_b,
                            beta_z=FLAGS.compile_beta_z,
                            prior_rate=FLAGS.compile_prior_rate)

    #Load in the model
    # 1. Load the pickled model
    # with open('experiment/compile_rGgNhBScvt/bot_best.pkl', "rb") as f:
    #     model = torch.load(f, map_location=torch.device("cpu"))  # or "cuda" as needed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Model loaded to device: ", device)

    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.compile_lr)

    # Dataset
    dataloader = compile.Dataloader(env_names=FLAGS.envs,
                                    val_ratio=0.05)
    train_steps = 0
    writer = SummaryWriter(training_folder)
    train_iter = dataloader.train_iter(batch_size=FLAGS.compile_batch_size)
    nb_frames = 0
    train_stats = DictList()
    best_model = None
    best_loss = float('inf')

    while train_steps < FLAGS.compile_train_steps:
        model.train()
        train_batch, train_lengths = train_iter.__next__()

        train_batch.apply(lambda _t: _t.to(device))
        train_lengths = train_lengths.to(device)

        train_outputs, _ = model.forward(train_batch, train_lengths)

        optimizer.zero_grad()
        train_outputs['loss'].backward()
        optimizer.step()
        train_steps += 1
        nb_frames += train_lengths.sum().item()

        train_outputs = DictList(train_outputs)
        train_outputs.apply(lambda _t: _t.item())
        train_stats.append(train_outputs)

        if train_steps % FLAGS.compile_eval_freq == 0:
            train_stats.apply(lambda _tensors: np.mean(_tensors))
            logger_str = ['[TRAIN] steps={}'.format(train_steps)]


            for k, v in train_stats.items():
                logger_str.append("{}: {:.4f}".format(k, v))
                writer.add_scalar('train/' + k, v, global_step=nb_frames)

                if k == 'loss':
                    if v < best_loss:
                        best_loss = v
                        best_model = copy.deepcopy(model)
                        print("Best model saved, loss: ", best_loss)
                        torch.save(best_model, os.path.join(training_folder, 'bot_best.pkl'))

            print('\t'.join(logger_str))
            train_stats = DictList()
            writer.flush()

    print("Training finished, start collecting segments")

    model = best_model if best_model is not None else model

    model.eval()
    model = model.to(device)

    all_z = []  # will hold arrays of shape (batch_size, latent_dim)
    all_sample_bs = []  # will hold arrays of shape (batch_size, T)
    all_lengths = [] #Will hold arrays of shape (batch_size, latent_dim)
    all_actions = [] #Will hold arrays of shape (batch_size, T)
    all_truths = []

    all_train_trajs = []
    for env_name in FLAGS.envs:
        all_train_trajs += dataloader.data['train'][env_name]
        all_train_trajs += dataloader.data['val'][env_name]


    for batch, lengths in dataloader.batch_iter(all_train_trajs,
                                                batch_size=1,
                                                shuffle=False,
                                                epochs=1):
        # if FLAGS.cuda:
        #     batch.apply(lambda t: t.cuda())
        #     lengths = lengths.cuda()

        batch.apply(lambda _t: _t.to(device))
        lengths = lengths.to(device)

        (_, extra_info) = model.forward(batch, lengths)
        b_samples = extra_info['segment']
        z_samples = extra_info['all_z']['samples']

        curr_seg_latents = []
        curr_seg_boundaries = []

        for seg_id in range(len(b_samples)):
            boundary_array = b_samples[seg_id]\
                                .argmax(dim=1)\
                                .cpu()\
                                .numpy()[0]

            latent_encs = z_samples[seg_id]\
                            .detach()\
                            .cpu()\
                            .numpy()[0]

            curr_seg_boundaries.append(boundary_array)
            curr_seg_latents.append(latent_encs)

        all_sample_bs.append(curr_seg_boundaries)
        all_z.append(curr_seg_latents)
        all_lengths.append(lengths.cpu().numpy()[0])
        all_actions.append(batch.action[0].cpu().numpy())
        all_truths.append(batch.groundTruth[0].cpu().numpy())

    print("Accumulated all segments and latents")
    all_z = np.array(all_z)
    all_sample_bs = np.array(all_sample_bs)

    print("All Latents Shape", all_z.shape)
    print("All Boundaries Shape", all_sample_bs.shape)

    flat_z = all_z.reshape(-1, all_z.shape[-1])
    flat_z_scaled = StandardScaler().fit_transform(flat_z)
    del flat_z

    # Define and fit the GMM model
    gmm = GaussianMixture(
        n_components=FLAGS.compile_skills,
        covariance_type='full',       # Allow full covariance to capture complex skill dependencies
        reg_covar=1e-6,               # Small regularization to improve stability
        n_init=5,                     # Multiple initializations for robustness
        random_state=0
    )
    gmm.fit(flat_z_scaled)

    # Predict the cluster labels and reshape
    labels = gmm.predict(flat_z_scaled).reshape(all_sample_bs.shape)
    del  flat_z_scaled
    
    # labels = np.zeros_like(all_sample_bs)

    print("Labels Shape", labels.shape)
    print("Trained Gaussian Mixture clustering model")

    episode_data = []
    preds = []
    preds_static = []

    for ep, model_out in enumerate(zip(all_z, all_sample_bs, labels, all_lengths, all_actions, all_truths)):
        lat, bound, curr_label, lens, acts, truths  = model_out

        assert lens == len(acts), "Length of actions and length of episode do not match"
        assert lens == len(truths), "Length of actions and length of truths do not match"

        bound.sort()
        segments = []
        pred_truth = ""
        prev = 0
        buffer_bound = np.array(bound) + 1
        for b, z, l in zip(buffer_bound, lat, curr_label):
            # only accept a boundary if it moves us forward
            if b <= prev:
                continue

            # clamp in case b > L
            end = min(b, lens)
            #(prev, end, z, l)

            pred_truth +=( str(l) * (end - prev))

            segments.append({
                'start': prev,
                'end': end,
                'latent': z,
                'label': l
            })

            prev = b
            # once we hit the true end, there are no more segments
            if b == lens:
                break

        # if prev < lens:
        #     pred_truth += str(curr_label[-1]) * (lens - prev)

        pred_truth_list = [int(c) for c in pred_truth]

        look_for = np.arange(5, 18)
        gt_segments = np.where(np.isin(acts, look_for))[0]

        if len(gt_segments) < FLAGS.compile_max_segs:
            #Extend gt_segments by padding wit hthe last segment
            # Pad gt_segments with the last value until it reaches compile_max_segs length
            if len(gt_segments) > 0:
                pad_value = gt_segments[-1]
                gt_segments = np.pad(gt_segments, (0, FLAGS.compile_max_segs - len(gt_segments)), mode='constant', constant_values=pad_value)
            else:
                # If gt_segments is empty, pad with zeros
                gt_segments = np.zeros(FLAGS.compile_max_segs, dtype=int)

        subtask_order = get_subtask_ordering(truths, gt_segments + 1)


        print("Ground Truth:            ", truths)
        print("Ground Truth Boundaries: ", gt_segments)
        print("Predicted Boundaries:    ", bound)
        print("Predicted Truths:        ", pred_truth_list)
        print("Actions:                 ", acts)
        print("Subtask Order:           ", subtask_order)
        print()

        if len(subtask_order) < FLAGS.compile_max_segs:
            # Pad subtask_order with the last value until it reaches compile_max_segs length
            pad_value = subtask_order[-1] if len(subtask_order) > 0 else 0
            subtask_order = np.pad(subtask_order, (0, FLAGS.compile_max_segs - len(subtask_order)), mode='constant', constant_values=pad_value)

        decoded_subtask = get_subtask_seq(torch.from_numpy(acts), subtask=subtask_order, use_ids=bound)


        episode_data.append({
            'episode': ep,
            'length': lens,
            'skill_info': segments,
            'actions': acts.tolist(), #Remove actions padding
            'predicted_skills': pred_truth_list,
            'predicted_skills_static': decoded_subtask.tolist(),
            'ground_truth': truths,
            'ground_truth_boundaries': gt_segments,
            'predicted_boundaries': bound
        })

        preds.append(pred_truth_list)
        preds_static.append(decoded_subtask)


    #Save the episode data to a file
    with open(os.path.join(training_folder, 'episode_data.pkl'), 'wb') as f:
        torch.save(episode_data, f)

    #Save the model
    with open(os.path.join(training_folder, 'full_model.pkl'), 'wb') as f:
        torch.save(model, f)

    print("Saved model and episode data to: ", training_folder)
    print("Finished collecting segments")

    for ep in episode_data:
        print("Episode: ", ep['episode'])
        print("Pred. Boundaries: ", ep['predicted_boundaries'].tolist())
        print("GT. Boundaries: ", ep['ground_truth_boundaries'].tolist())
        print("Predicted Truths: ", ep['predicted_skills'])
        print("Ground Truth:     ", ep['ground_truth'].tolist())
        print("Predicted Static: ", ep['predicted_skills_static'])
        print("Actions:          ", ep['actions'])
        print("--------------------------------------------------")

    clustering_metrics = get_all_metrics(preds, all_truths)
    static = get_all_metrics(preds_static, all_truths)

    print("Clustering Results (ASOT):")
    print("----------------------------------")
    for name, val in clustering_metrics.items():
        print(f"{name:<15}{val:.4f}")
    print("----------------------------------\n")


    print("Clustering Results (OMPN STATIC):")
    print("----------------------------------")
    for name, val in static.items():
        print(f"{name:<15}{val:.4f}")
    print("----------------------------------\n")


    if not FLAGS.debug:
        print("Plotting results...")
        pred_batch, gt_batch, mask = make_batch(preds_static, all_truths, pad_value=-1)
        visualisation_dir = os.path.join(training_folder, 'visualisation_static')
        os.makedirs(visualisation_dir, exist_ok=True)

        for i, data_batch in enumerate(zip(pred_batch, gt_batch, mask)):
            p, g, m = data_batch
            fig = plot_segmentation_gt(g, p, m)

            #Save it
            fig.savefig(os.path.join(visualisation_dir, f"segmentation_{i}.png"), dpi=300)
            plt.close(fig)

        print("Plotting results...")
        pred_batch, gt_batch, mask = make_batch(preds, all_truths, pad_value=-1)
        visualisation_dir = os.path.join(training_folder, 'visualisation_clustering')
        os.makedirs(visualisation_dir, exist_ok=True)

        for i, data_batch in enumerate(zip(pred_batch, gt_batch, mask)):
            p, g, m = data_batch
            fig = plot_segmentation_gt(g, p, m)

            #Save it
            fig.savefig(os.path.join(visualisation_dir, f"segmentation_{i}.png"), dpi=300)
            plt.close(fig)

        print("Finished plotting results")
