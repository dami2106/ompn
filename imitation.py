"""
imitation
"""
from absl import flags, logging
from gym_psketch import *
from gym_psketch import bots
import gym
import torch
import time
from typing import Dict
from tensorboardX import SummaryWriter
import numpy as np

from gym_psketch.evaluate import evaluate_loop, parsing_loop
from gym_psketch.visualize import distance2ctree, tree_to_str

FLAGS = flags.FLAGS

# Misc
flags.DEFINE_bool('il_demo_from_model', default=False, help='Whether to use demo from bot')
flags.DEFINE_integer('il_train_steps', default=300, help='Trainig steps')
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
        else:
            #We are here
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
    for env_name in dataloader.env_names:
        print(f"Environment: {env_name}")
        val_iter = dataloader.val_iter(batch_size=1, env_names=[env_name])
        for ep_idx, (batch, lengths) in enumerate(val_iter):
            if FLAGS.cuda:
                batch.apply(lambda _t: _t.cuda())
                lengths = lengths.cuda()
            seq_len = lengths.item()
            env_ids = batch.env_id[:, 0]
            mems = bot.init_memory(env_ids)
            actions = batch.action
            skill_trace = []
            for t in range(seq_len):
                transition = batch[:, t]
                with torch.no_grad():
                    out = bot.forward(transition, transition.env_id, mems)
                p_hat = out.p  # shape [1, nb_slots+1]
                p_slots = p_hat[:, 1:]  # drop end logit
                p_dist = torch.nn.functional.normalize(p_slots, dim=1)
                skill_id = p_dist.argmax(dim=1).item()
                skill_trace.append(skill_id)
                mems = out.mems
            print(f"  Episode {ep_idx}: skills : {skill_trace}")
            print(f"  Episode {ep_idx}: actions: {actions.squeeze().tolist()}")

            # Extract predicted segments from skill trace
            def extract_segments(skill_trace):
                """Extract contiguous segments from skill trace.
                Returns list of (start, end, skill_id) tuples."""
                if not skill_trace:
                    return []

                segments = []
                current_skill = skill_trace[0]
                start_idx = 0

                for i in range(1, len(skill_trace)):
                    if skill_trace[i] != current_skill:
                        # End of current segment
                        segments.append((start_idx, i-1, current_skill))
                        # Start new segment
                        current_skill = skill_trace[i]
                        start_idx = i

                # Add final segment
                segments.append((start_idx, len(skill_trace)-1, current_skill))
                return segments

            segments = extract_segments(skill_trace)
            print(f"  Episode {ep_idx}: segments: {segments}")

            # Build and print parse tree based on skill assignments
            depths = np.array(skill_trace[:-1])  # drop last step to match actions length - 1
            action_seq = [str(a) for a in actions.squeeze().tolist()]
            parse_tree = distance2ctree(depths, action_seq, False)
            tree_str = tree_to_str(parse_tree)
            print(f"  Episode {ep_idx}: tree: {tree_str[1:-1]}")
            print("--------------------------")





def run(training_folder):
    print('Start IL...')
    first_env = gym.make(FLAGS.envs[0])
    n_feature, action_size = first_env.n_features, first_env.n_actions
    # n_feature, action_size = 1087, 17
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


    try:
        print('Start training')
        main_loop(bot, dataloader, opt, training_folder, None)
    except KeyboardInterrupt:
        pass

    # Save the checkpoint
    with open(os.path.join(training_folder, 'bot.pkl'), 'wb') as f:
        torch.save(bot, f)
