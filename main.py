"""
Main entry
"""
import sys
from absl import flags, logging
from absl import app
import time
import datetime
import os
import traceback
import string
import torch
import random
import uuid
import gym_psketch.bots.model_bot
import imitation as IL
import generate_demo as demo
from taco import train_taco
from compile import train_compile_asot

random.seed(0)
# np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

FLAGS = flags.FLAGS
flags.DEFINE_list('envs', default=['makebedfull-v0'], help='List of env to train. Use comma spearated')
flags.DEFINE_list('test_envs', default=[], help='Extra test envs')
flags.DEFINE_integer('max_steps', default=64, help='maximum environment steps')
flags.DEFINE_integer('width', default=10, help='width of env')
flags.DEFINE_integer('height', default=10, help='height of env')
#MLP, LSTM are from the paper; OMSTACK is the OMPN arch
flags.DEFINE_enum('arch', enum_values=['mlp', 'lstm', 'omstack'],
                  default='omstack', help='Architecture')
flags.DEFINE_integer('eval_episodes', default=30, help='Evaluation episode number')

# Model
flags.DEFINE_integer('hidden_size', default=128, help='model hidden size')
#Nb_slots is the number of levels in hierarchy 
flags.DEFINE_integer('nb_slots', default=3, help='model hidden size')
#env_arch is the amount of info we give the model about the env 
flags.DEFINE_enum('env_arch', enum_values=['emb', 'sketch', 'noenv', 'grusketch'],
                  default='noenv', help='env encoder Architecture')

# Misc
flags.DEFINE_bool('debug', default=False, help='Flag for debug mode')
#IL is OMPN
flags.DEFINE_enum('mode', default='IL', enum_values=['IL', 'demo', 'compile',
                                                     'taco'],
                  help='choosing between IL and baselines')
flags.DEFINE_string('experiment', default=None, help='Name of experiment')
flags.DEFINE_bool('cuda', default=True, help='Use cuda')
flags.DEFINE_integer('procs', default=4, help='Number of process')


flags.DEFINE_bool('minecraft', default=False, help='Whether or not to use minecraft loading')


def handler(type, value, tb):
    logging.exception("Uncaught exception: %s", str(value))
    logging.exception("\n".join(traceback.format_exception(type, value, tb)))


def random_string():
    return uuid.uuid4().hex[:10]


def setup_logging_and_exp_folder():
    # Random string if debug
    if FLAGS.debug:
        FLAGS.experiment = "{}_{}".format(FLAGS.mode, random_string())

    # Use time stamp or user specified if not debug
    else:
        ts = time.time()
        FLAGS.experiment = FLAGS.experiment if FLAGS.experiment is not None else \
            "{}_{}".format(FLAGS.mode,
                           datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M'))
    training_folder = os.path.join(gym_psketch.EXP_DIR, FLAGS.experiment)

    # Create train folder
    if os.path.exists(training_folder):
        print('{} exists!'.format(training_folder))
        exit(-1)
    else:
        os.makedirs(training_folder, exist_ok=False)

    # set up logging
    if FLAGS.debug:
        print("Debug mode")
        logging.get_absl_handler().python_handler.stream = sys.stdout
    else:
        logging.get_absl_handler().use_absl_log_file('absl_logging', training_folder)
    return training_folder


def main(_):
    trainig_folder = setup_logging_and_exp_folder()
    # FLAGS.cuda = False
    # logging.info('Use Cuda: {}'.format(FLAGS.cuda))
    # logging.info('Current git SHA: ' + gym_psketch.CURR_VERSION)

    flags.FLAGS.cuda = torch.cuda.is_available()
    if FLAGS.cuda:
        print("Using CUDA")
    else:
        print("Not using CUDA")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    # Number of GPUs
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")

    # Print details for each GPU
    for i in range(gpu_count):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Capability: {torch.cuda.get_device_capability(i)}")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        print(f"  Memory Reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")


    # save options
    fpath = os.path.join(trainig_folder, 'flagfile')
    with open(fpath, 'w') as f:
        f.write(FLAGS.flags_into_string())

    if FLAGS.mode == 'IL':
        IL.run(training_folder=trainig_folder)
    elif FLAGS.mode == 'demo':
        demo.main()
    elif FLAGS.mode == 'compile':
        train_compile_asot.main(training_folder=trainig_folder)
    elif FLAGS.mode == 'taco':
        train_taco.main(training_folder=trainig_folder)
    else:
        logging.fatal('Improper Mode {}'.format(FLAGS.mode))
    logging.info('Done')


if __name__ == '__main__':
    FLAGS(sys.argv)

    print("\nFlags:")
    for flag_name in FLAGS:
        print(f"{flag_name}: {getattr(FLAGS, flag_name)}")
    print()

    app.run(main)
