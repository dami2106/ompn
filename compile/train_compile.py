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

FLAGS = flags.FLAGS
flags.DEFINE_integer('compile_train_steps', default=4000, help='train steps')
flags.DEFINE_integer('compile_eval_freq', default=20, help='evaluation frequency')
flags.DEFINE_float('compile_lr', default=0.001, help='learning rate')
flags.DEFINE_integer('compile_batch_size', default=256, help='learning rate')
flags.DEFINE_integer('compile_max_segs', default=4, help='num of segment')
flags.DEFINE_integer('compile_skills', default=4, help='num of skills')
flags.DEFINE_float('compile_beta_z', default=0.1, help='weight Z kl loss')
flags.DEFINE_float('compile_beta_b', default=0.1, help='weight b kl loss')
flags.DEFINE_float('compile_prior_rate', default=3, help='possion distribution. avg length of each seg')
flags.DEFINE_enum('compile_latent', enum_values=['gaussian', 'concrete'], default='gaussian',
                  help='Latent type')



def main(training_folder):
    print('Start compile...')
    # first_env = gym.make(FLAGS.envs[0])
    n_feature, action_size = 1075, 7
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
    with open('experiment/compile_rGgNhBScvt/bot_best.pkl', "rb") as f:
        model = torch.load(f, map_location=torch.device("cpu"))  # or "cuda" as needed

    device = torch.device("cpu")
    print("Model: ")
    if FLAGS.cuda:
        model = model.to(device)

    print("Model loaded")
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.compile_lr)

    # Dataset
    dataloader = compile.Dataloader(env_names=FLAGS.envs,
                                    val_ratio=0.2)
    train_steps = 0
    writer = SummaryWriter(training_folder)
    train_iter = dataloader.train_iter(batch_size=FLAGS.compile_batch_size)
    nb_frames = 0
    train_stats = DictList()

    # env_id_subtask = None
    env_id_subtask = [7, 0, 4, 5]

    while train_steps < FLAGS.compile_train_steps:

        #Evalaute on validation set 
        if train_steps % FLAGS.compile_eval_freq == 0:
            # print("Evaluating on validation set")
            model.eval()
            env_name = FLAGS.envs[0]    

            val_iter = dataloader.val_iter(batch_size=FLAGS.eval_episodes, env_names=[env_name], shuffle=True)
            val_batch, val_lengths = val_iter.__next__()

            if FLAGS.cuda:
                val_batch.apply(lambda _t: _t.cuda())
                val_lengths = val_lengths.cuda()


            with torch.no_grad():
                _, extra_info = model.forward(val_batch, val_lengths)

            seg = torch.stack(extra_info['segment'], dim=1).argmax(-1)
       
            task_corrects = []  # Reset per validation pass

            for _action, _seg, _length, _env_id in zip(val_batch.action, seg, val_lengths, val_batch.env_id):
                __action = _action[:_length]
                use_id = ((__action == 4).nonzero()).view(-1).cpu()  # Ground truth segments 
                _seg = _seg[_seg.sort()[1]].cpu()  # Predicted segments 

                env_id_subtask = ID2SKETCHS[_env_id[0].item()]
                gt_subtask = get_subtask_seq(__action, subtask=env_id_subtask, use_ids=use_id)
                decoded_subtask = get_subtask_seq(__action, subtask=env_id_subtask, use_ids=_seg)

                # Compute per-episode accuracy
                correct = (gt_subtask == decoded_subtask).float().sum().item()
                total = len(gt_subtask)
                episode_accuracy = correct / total if total > 0 else 0.0

                task_corrects.append(episode_accuracy)

            mean_accuracy = sum(task_corrects) / len(task_corrects)
            print(f"[VALID] Mean Ep. Accuracy: {mean_accuracy}")
      

        model.train()
        train_batch, train_lengths = train_iter.__next__()
        if FLAGS.cuda:
            train_batch.apply(lambda _t: _t.cuda())
            train_lengths = train_lengths.cuda()
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
            print('\t'.join(logger_str))
            train_stats = DictList()
            writer.flush()
        

    print("Training finished, start collecting segments")

    model.eval()
    if FLAGS.cuda:
        model = model.cuda()

    all_z = []  # will hold arrays of shape (batch_size, latent_dim)
    all_sample_bs = []  # will hold arrays of shape (batch_size, T)
    all_lengths = [] #Will hold arrays of shape (batch_size, latent_dim)
    all_actions = [] #Will hold arrays of shape (batch_size, T)
    all_truths = []

    all_train_trajs = []
    for env_name in FLAGS.envs:
        all_train_trajs += dataloader.data['train'][env_name]
        all_train_trajs += dataloader.data['val'][env_name]

    all_z = []
    all_sample_bs = []
    all_lengths = []
    all_actions = []

    for batch, lengths in dataloader.batch_iter(all_train_trajs,
                                                batch_size=1,
                                                shuffle=False,
                                                epochs=1):
        if FLAGS.cuda:
            batch.apply(lambda t: t.cuda())
            lengths = lengths.cuda()

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

    print("Accumulated all segments and latents")
    all_z = np.array(all_z)
    all_sample_bs = np.array(all_sample_bs)

    print("All Latents Shape", all_z.shape)
    print("All Boundaries Shape", all_sample_bs.shape)

    flat_z = all_z.reshape(-1, all_z.shape[-1])
    flat_z_scaled = StandardScaler().fit_transform(flat_z)
    kmeans = KMeans(n_clusters=FLAGS.compile_skills, random_state=0).fit(flat_z)
    labels = kmeans.labels_.reshape(all_sample_bs.shape)
    del flat_z
    print("Trained KMeans clustering model")
 
    episode_data = []
    preds = []
    truths = []
    static_preds = []

    for ep, model_out in enumerate(zip(all_z, all_sample_bs, labels, all_lengths, all_actions)):
        lat, bound, curr_label, lens, acts = model_out
        acts_tensor = torch.tensor(acts, dtype=torch.int64)

        use_id = ((acts_tensor == 4).nonzero()).view(-1).cpu()
        gt_subtask = get_subtask_seq(acts_tensor, subtask=env_id_subtask, use_ids=use_id)
        decoded_subtask = get_subtask_seq(acts_tensor, subtask=env_id_subtask,
                                                    use_ids=bound)

        assert lens == len(acts), "Length of actions and length of episode do not match"
     
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
        episode_data.append({
            'episode': ep,
            'length': lens,
            'skill_info': segments,
            'actions': acts.tolist(), #Remove actions padding 
            'predicted_truths': pred_truth_list,
            'ground_truth': gt_subtask.cpu().tolist(),
            'predicted_static' : decoded_subtask.cpu().tolist(),
            'boundaries': bound,
        })

        preds.append(pred_truth_list)
        truths.append(gt_subtask.cpu().tolist())
        static_preds.append(decoded_subtask.cpu().tolist())
    
 
    #Save the episode data to a file
    with open(os.path.join(training_folder, 'episode_data.pkl'), 'wb') as f:
        torch.save(episode_data, f)

    #Save the model 
    with open(os.path.join(training_folder, 'bot_best.pkl'), 'wb') as f:
        torch.save(model, f)

    print("Saved model and episode data to: ", training_folder)
    print("Finished collecting segments")

    for ep in episode_data:
        print("Episode: ", ep['episode'])
        print("Pred. Boundaries: ", ep['boundaries'].tolist())
        print("Predicted Truths: ", ep['predicted_truths'])
        print("Predicted Static: ", ep['predicted_static'])
        print("Ground Truth:     ", ep['ground_truth'])
        print("Actions:          ", ep['actions'])
        print("--------------------------------------------------")

    clustering_metrics = get_all_metrics(preds, truths)
    static_metrics = get_all_metrics(static_preds, truths)  

    print("Clustering Results (ASOT):")
    print("----------------------------------")
    for name, val in clustering_metrics.items():
        print(f"{name:<15}{val:.4f}")
    print("----------------------------------")
    print()
    print("Static Results (OMPN Assumption):")
    print("----------------------------------")
    for name, val in static_metrics.items():
        print(f"{name:<15}{val:.4f}")
    print("----------------------------------")

