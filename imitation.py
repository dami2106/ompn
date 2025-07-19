"""
imitation
"""
from absl import flags, logging
from gym_psketch import *
from gym_psketch import bots
import gym
import torch
import time
import os
from typing import Dict
from tensorboardX import SummaryWriter
import numpy as np
import random

from gym_psketch.evaluate import evaluate_loop, parsing_loop, get_boundaries, get_use_ids, get_subtask_seq, f1
from gym_psketch.visualize import distance2ctree, tree_to_str

from sklearn.cluster import KMeans
from metrics import *

from visualisation import plot_segmentation_gt, parse_tree_string, save_tree_to_json, visualize_tree
import matplotlib.pyplot as plt
import json
from gym_psketch.visualize import predicted_data_to_tree, tree_to_str, predicted_data_to_hierarchical_tree


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

FLAGS = flags.FLAGS



# Misc
flags.DEFINE_bool('il_demo_from_model', default=False, help='Whether to use demo from bot')
flags.DEFINE_integer('il_train_steps', default=50, help='Trainig steps')
flags.DEFINE_integer('il_eval_freq', default=20, help='Evaluation frequency')
flags.DEFINE_integer('il_save_freq', default=200, help='Save freq')
flags.DEFINE_bool('il_no_done', default=False, help='Whether or not to use done during IL')

# Data
flags.DEFINE_float('il_val_ratio', default=0.2,
                   help='Validation')
flags.DEFINE_integer('il_batch_size', default=128,
                     help='batch size')

# Optimize
flags.DEFINE_integer('il_recurrence', default=30, help='bptt length')
flags.DEFINE_float('il_lr', default=5e-4, help="Learning rate")
flags.DEFINE_float('il_clip', default=0.2, help='RNN clip')


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

def run_batch(batch: DictList,
              batch_lengths,
              bot: ModelBot,
              mode='train') \
        -> (DictList, torch.Tensor):
    """
    :param batch: DictList object [bsz, seqlen]
    :param bot:  A model Bot
    :param mode: 'train' or 'eval'
    :return:
        stats: A DictList of bsz, mem_size
    """
    bsz, seqlen = batch.action.shape[0], batch.action.shape[1]
    env_ids = batch.env_id[:, 0] #Shape is [batch_size] since we stack them for the batch
    final_outputs = DictList({})
    mems = None

    # print("Env IDs: ", env_ids)
    # print("Len of env_ids: ", len(env_ids))
    if bot.is_recurrent:
        # print("Bot is recurrent")
        mems = bot.init_memory(env_ids=env_ids)


    for t in range(seqlen):
        curr_transition = batch[:, t]
        final_output = DictList({})
        model_output = bot.forward(curr_transition, curr_transition.env_id, mems)

        # print('model_output', model_output.keys())

        logits = model_output.dist.logits
        targets = curr_transition.action
        if FLAGS.il_no_done:
            final_output.ce_loss = torch.nn.functional.cross_entropy(input=logits, target=targets,
                                                                     reduction='none',
                                                                     ignore_index=bot.done_id)
            # print("ignore_index: ", bot.done_id)
        else:
            #We are here
            # print("DONT ignore_index: ", bot.done_id)
            final_output.ce_loss = torch.nn.functional.cross_entropy(input=logits, target=targets,
                                                                     reduction='none')
            # print("dont ignore ", final_output.ce_loss)
        preds = logits.argmax(-1)

        # print("Predictions: ", preds)
        # print("Targets: ", targets)
        # print("Targets: ", targets)
        # print("Preds: ", preds)

        final_output.acc = (preds == targets).float()
        final_outputs.append(final_output)

        # Update memory
        next_mems = None
        if bot.is_recurrent:
            next_mems = model_output.mems
            if (t+1) % FLAGS.il_recurrence == 0 and mode =='train':
                next_mems = next_mems.detach()
        mems = next_mems

    # Stack on time dim
    final_outputs.apply(lambda _tensors: torch.stack(_tensors, dim=1))
    sequence_mask = torch.arange(batch_lengths.max().item(),
                                 device=batch_lengths.device)[None, :] < batch_lengths[:, None]
    final_outputs.apply(lambda _t: _t.masked_fill(~sequence_mask, 0.))
    return final_outputs


def evaluate_on_envs(bot, dataloader):
    val_metrics = {}
    bot.eval()
    envs = dataloader.env_names
    for env_name in envs:
        val_iter = dataloader.val_iter(batch_size=FLAGS.il_batch_size,
                                       env_names=[env_name])
        output = DictList({})
        total_lengths = 0
        for batch, batch_lens in val_iter:
            if FLAGS.cuda:
                batch.apply(lambda _t: _t.cuda())
                batch_lens = batch_lens.cuda()

            # Initialize memory
            with torch.no_grad():
                batch_results = run_batch(batch, batch_lens, bot, mode='val')
            batch_results.apply(lambda _t: _t.sum().item())
            output.append(batch_results)
            total_lengths += batch_lens.sum().item()
        output.apply(lambda _t: torch.tensor(_t).sum().item() / total_lengths)
        val_metrics[env_name] = {k: v for k, v in output.items()}

    # Parsing
    if 'om' in FLAGS.arch:
        with torch.no_grad():
            parsing_stats, parsing_lines = parsing_loop(bot, dataloader=dataloader)
        for env_name in parsing_stats:
            parsing_stats[env_name].apply(lambda _t: np.mean(_t))
            val_metrics[env_name].update(parsing_stats[env_name])
        print('Get parsing result')
        print('\n' + '\n'.join(parsing_lines))

    # evaluate on free run env
    val_metrics = evaluate_loop(bot, val_metrics)
    return val_metrics


def logging_metric(nb_frames, steps, metrics, writer, prefix):
    # Logger
    for env_name, metric in metrics.items():
        line = ['[{}][{}] steps={}'.format(prefix, env_name, steps)]
        for k, v in metric.items():
            line.append('{}: {:.4f}'.format(k, v))
        print('\t'.join(line))
    mean_val_metric = DictList()
    for metric in metrics.values():
        mean_val_metric.append(metric)
    mean_val_metric.apply(lambda t: torch.mean(torch.tensor(t)))
    for k, v in mean_val_metric.items():
        writer.add_scalar(prefix + '/' + k, v.item(), nb_frames)
    writer.flush()


def main_loop(bot, dataloader, opt, training_folder, test_dataloader=None):
    # Prepare
    train_steps = 0
    writer = SummaryWriter(training_folder)
    train_iter = dataloader.train_iter(batch_size=FLAGS.il_batch_size)
    nb_frames = 0
    train_stats = DictList()
    curr_best = 0
    while True:
        # print('Training steps: {}'.format(train_steps))
        if train_steps > FLAGS.il_train_steps:
            print('Reaching maximum steps')
            break

        if train_steps % FLAGS.il_save_freq == 0:
            with open(os.path.join(training_folder, 'bot{}.pkl'.format(train_steps)), 'wb') as f:
                torch.save(bot, f)

        # if train_steps % FLAGS.il_eval_freq == 0:
        #     print('Evaluating...')
        #     # testing on valid
        #     val_metrics = evaluate_on_envs(bot, dataloader)
        #     logging_metric(nb_frames, train_steps, val_metrics, writer, prefix='val')

        #     # testing on test env
        #     if test_dataloader is not None:
        #         test_metrics = evaluate_on_envs(bot, test_dataloader)
        #         logging_metric(nb_frames, train_steps, test_metrics, writer, prefix='test')

        #     avg_ret = [val_metrics[env_name]['ret'].item() for env_name in val_metrics]
        #     avg_ret = np.mean(avg_ret)

        #     if avg_ret > curr_best:
        #         curr_best = avg_ret
        #         print('Save Best with return: {}'.format(avg_ret))

        #         # Save the checkpoint
        #         with open(os.path.join(training_folder, 'bot_best.pkl'), 'wb') as f:
        #             torch.save(bot, f)

        # Forward/Backward
        bot.train()
        train_batch, train_lengths = train_iter.__next__()
        if FLAGS.cuda:
            train_batch.apply(lambda _t: _t.cuda())
            train_lengths = train_lengths.cuda()

        start = time.time()
        train_batch_res = run_batch(train_batch, train_lengths, bot)
        train_batch_res.apply(lambda _t: _t.sum() / train_lengths.sum())
        batch_time = time.time() - start
        loss = train_batch_res.ce_loss
        opt.zero_grad()
        loss.backward()
        params = [p for p in bot.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters=params, max_norm=FLAGS.il_clip)
        opt.step()
        train_steps += 1
        nb_frames += train_lengths.sum().item()
        fps = train_lengths.sum().item() / batch_time

        stats = DictList()
        stats.grad_norm = grad_norm
        stats.acc = train_batch_res.acc.detach()
        stats.ce_loss = train_batch_res.ce_loss.detach()
        stats.fps = torch.tensor(fps)
        train_stats.append(stats)

        if train_steps % FLAGS.il_eval_freq == 0:
            train_stats.apply(lambda _tensors: torch.stack(_tensors).mean().item())
            logger_str = ['[TRAIN] steps={}'.format(train_steps)]
            for k, v in train_stats.items():
                logger_str.append("{}: {:.4f}".format(k, v))
                writer.add_scalar('train/' + k, v, global_step=nb_frames)
            print('\t'.join(logger_str))
            train_stats = DictList()
            writer.flush()

    # === APPENDED: extract and print skill IDs per episode ===
    print("\nExtracting skill assignments across validation episodes...")
    bot.eval()

    episode_details = []
    all_predicted_subtask = []
    all_gt_subtask = []
    for env_name in dataloader.env_names:
        print(f"Environment: {env_name}")
    
        
        # Get all trajectories for this environment
        all_trajs = dataloader.data['train'][env_name] + dataloader.data['val'][env_name]
        print(f"Number of trajectories to eval: {len(all_trajs)}")
        # random.shuffle(all_trajs)
        
        for ep_idx, traj in enumerate(all_trajs):
            # Convert trajectory to batch format
            batch = DictList(traj)
            batch.apply(lambda _t: torch.tensor(_t)[:-1].unsqueeze(0))  # Remove done

            # batch.apply(lambda _t: torch.tensor(_t)[:-1].unsqueeze(0))  # Remove done
            if FLAGS.cuda:
                batch.apply(lambda _t: _t.cuda())
            
            seq_len = batch.action.shape[1]
            env_ids = batch.env_id[:, 0]
            mems = bot.init_memory(env_ids)
            actions = batch.action
            mem_trace = []
            
            for t in range(seq_len):
                transition = batch[:, t]
                with torch.no_grad():
                    out = bot.forward(transition, transition.env_id, mems)
                p_hat = out.p  # shape [1, nb_slots+1]
                p_slots = p_hat[:, 1:]  # drop end logit
                p_dist = torch.nn.functional.normalize(p_slots, dim=1)
                skill_id = p_dist.argmax(dim=1).item()
                mem_trace.append(skill_id)
                mems = out.mems
            # print(f"  Episode {ep_idx}: mems : {mem_trace}")
            # print(f"  Episode {ep_idx}: actions: {actions.squeeze().tolist()}")

            # Compute and print boundaries
            p_hats = []
            mems = bot.init_memory(env_ids)
            for t in range(seq_len):
                transition = batch[:, t]
                with torch.no_grad():
                    out = bot.forward(transition, transition.env_id, mems)
                p_hats.append(out.p)
                mems = out.mems
            p_hats = torch.stack(p_hats, dim=0)  # [seq_len, 1, nb_slots+1]
            

            look_for = np.arange(5, 18)
            actions_cpu = actions.cpu().squeeze().numpy()
            
            gt_segments = np.where(np.isin(actions_cpu, look_for))[0] 
        
            subtask_order = get_subtask_ordering(batch.groundTruth[0].cpu().numpy(), gt_segments + 1)
            if not FLAGS.debug:
                print(f"{'Subtask Sequence:':20} {subtask_order}")
                print(f"{'GT Segments:':20} {gt_segments}")

            boundaries = get_boundaries(p_hats.squeeze(1), bot.nb_slots, threshold=0.5, nb_boundaries=len(subtask_order))

            # ground_truth = get_use_ids(actions_cpu.squeeze(), env_name)
            predicted = np.array(boundaries)
            
            #Increase all elements of predicted by 1 except the last one
            predicted[:-1] += 1

            if len(predicted) < len(gt_segments):
                predicted = np.pad(predicted, (0, len(gt_segments) - len(predicted)), constant_values=predicted[-1])

            if not FLAGS.debug:
                print(f"{'Predicted:':20} {predicted}")
                print()
            # Get decoded subtasks for both ground truth and predicted
            _action = torch.from_numpy(actions_cpu.squeeze()) 
            _decoded_subtask = get_subtask_seq(_action, 
                                             subtask=subtask_order,
                                             use_ids=predicted)
            
            
            _gt_subtask = get_subtask_seq(_action,
                                         subtask=subtask_order,
                                         use_ids=gt_segments)

            if not FLAGS.debug:
                print(f"{'Decoded Subtask:':20} {_decoded_subtask}")
                print(f"{'OURS GT Subtask:':20} {batch.groundTruth[0].cpu().numpy()}")
                print(f"{'GT Subtask:':20} {_gt_subtask}")
                print()

            # Generate tree representation from predicted data
            pred_tree = predicted_data_to_tree(actions_cpu.squeeze(), 
                                             predicted, 
                                             _decoded_subtask.tolist())
            tree_str = tree_to_str(pred_tree)

            # predicted_tree = parse_tree_string(tree_str)
            


            # Generate tree representation from ground truth data  
            gt_tree = predicted_data_to_tree(actions_cpu.squeeze(),
                                           gt_segments,
                                           _gt_subtask.tolist())
            gt_tree_str = tree_to_str(gt_tree)
            # gt_tree = parse_tree_string(gt_tree_str)
            if not FLAGS.debug:
                print(f"{'Predicted Tree:':20} {tree_str}")
                print(f"{'Ground Truth Tree:':20} {gt_tree_str}")
                print("="*48)



            episode_details.append({
                # 'mem_trace': mem_trace,
                'ep_idx': ep_idx,
                'actions': actions.squeeze().tolist(),
                'boundaries': boundaries,
                'gt_boundaries': gt_segments,
                'subtask_order': subtask_order,
                'decoded_subtask': _decoded_subtask.tolist(),
                'gt_subtask': _gt_subtask.tolist(),
                'predicted_tree': tree_str,
                'gt_tree': gt_tree_str
            })

            all_predicted_subtask.append(_decoded_subtask.tolist())
            all_gt_subtask.append( _gt_subtask.tolist())



        print(f"Evaluating environment {env_name} done.")


        static = get_all_metrics(all_predicted_subtask, all_gt_subtask)

        print("OMPN Results (OMPN STATIC):")
        print("----------------------------------")
        for name, val in static.items():
            print(f"{name:<15}{val:.4f}")
        print("----------------------------------\n")


        if not FLAGS.debug:
            print("Plotting results...")
            pred_batch, gt_batch, mask = make_batch(all_predicted_subtask, all_gt_subtask, pad_value=-1)
            visualisation_dir = os.path.join(training_folder, 'visualisation')
            os.makedirs(visualisation_dir, exist_ok=True)

            for i, data_batch in enumerate(zip(pred_batch, gt_batch, mask)):
                p, g, m = data_batch
                fig = plot_segmentation_gt(g, p, m, comparison_name='OMPN')

                #Save it
                fig.savefig(os.path.join(visualisation_dir, f"segmentation_{i}.png"), dpi=300)
                plt.close(fig)

            print("Finished plotting results")

            # Save and visualize trees
            print("Saving and visualizing trees...")
            trees_dir = os.path.join(training_folder, 'episode_data')
            os.makedirs(trees_dir, exist_ok=True)
            
            # Save all tree data to a JSON file for later analysis
            data_details = {
                'episode_details': episode_details,
                'environment': env_name,
                'total_episodes': len(episode_details)
            }
            
            with open(os.path.join(trees_dir, f'data_details_{env_name}.json'), 'w') as f:
                json.dump(data_details, f, indent=2)
            
            # Create text file with formatted tree outputs
            with open(os.path.join(trees_dir, f'episode_details_{env_name}.txt'), 'w') as f:
                f.write(f"Analysis for Environment: {env_name}\n")
                f.write("=" * 60 + "\n\n")
                
                for ep_idx, episode in enumerate(episode_details):
                    f.write(f"Episode {ep_idx}:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Actions: {episode['actions']}\n")
                    f.write(f"Boundaries: {episode['boundaries']}\n")
                    f.write(f"Predicted Subtasks: {episode['decoded_subtask']}\n")
                    f.write(f"GT Subtasks: {episode['gt_subtask']}\n")
                    f.write("\n")

            #Create a folder called 'tree_visualization' to save the visualizations
            tree_visualization_dir =  os.path.join(training_folder, 'tree_visualization')
            os.makedirs(tree_visualization_dir, exist_ok=True)
            for ep_idx, episode in enumerate(episode_details):
                # Visualize predicted tree
                pred_tree = episode['predicted_tree']
                gt_tree = episode['gt_tree']

                save_tree_to_json(pred_tree, os.path.join(tree_visualization_dir, f'predicted_tree_{ep_idx}.json'))
                visualize_tree(pred_tree, os.path.join(tree_visualization_dir, f'predicted_tree_{ep_idx}'))

                # Save ground truth tree to JSON
                save_tree_to_json(gt_tree, os.path.join(tree_visualization_dir, f'gt_tree_{ep_idx}.json'))
                visualize_tree(gt_tree, os.path.join(tree_visualization_dir, f'gt_tree_{ep_idx}'))

            

            print("Finished saving and visualizing trees")




def run(training_folder):
    print('Start IL...')
    # first_env = gym.make(FLAGS.envs[0])
    # n_feature, action_size = first_env.n_features, first_env.n_actions
    n_feature, action_size = 1087, 18
    # n_feature, action_size = 1075, 7
    bot = bots.make(vec_size=n_feature,
                    action_size=action_size,
                    arch=FLAGS.arch,
                    hidden_size=FLAGS.hidden_size,
                    nb_slots=FLAGS.nb_slots,
                    env_arch=FLAGS.env_arch)
    if FLAGS.cuda:
        bot = bot.cuda()

    params = [p for p in bot.parameters()]
    opt = torch.optim.Adam(params, lr=FLAGS.il_lr)
    print('Model: {}'.format(bot))
    dataloader = Dataloader(FLAGS.envs, FLAGS.il_val_ratio)
    print('Dataloader: {}'.format(dataloader))

    # Set random seeds for reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    try:
        print('Start training')
        main_loop(bot, dataloader, opt, training_folder, None)
    except KeyboardInterrupt:
        pass

    # Save the checkpoint
    with open(os.path.join(training_folder, 'bot.pkl'), 'wb') as f:
        torch.save(bot, f)
