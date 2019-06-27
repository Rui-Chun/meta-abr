import numpy as np
import torch

from maml_rl.metalearner import MetaLearner
from maml_rl.policies import ActorNet, CriticNet
from maml_rl.sampler import MrcBatchSampler
from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       weighted_normalize)
from metaTester import noMetaTest

from tensorboardX import SummaryWriter

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001  # learning rate
CRITIC_LR_RATE = 0.001
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 500
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
HD_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.  # 49?
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
TEST_LOG_FOLDER = './test_results/'
TRAIN_TRACES = "./train_sim_traces/"


def total_rewards(episodes_rewards, aggregation=torch.mean):
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
        for rewards in episodes_rewards], dim=0))
    return rewards.item()


def main(args):
    np.random.seed(RANDOM_SEED)

    save_folder = './saves/{0}'.format(args.output_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    sampler = MrcBatchSampler(args.env_name, batch_size=args.fast_batch_size, train_folder=TRAIN_TRACES)

    policy = ActorNet(input_size=[S_INFO, S_LEN], output_size=A_DIM, learning_rate=ACTOR_LR_RATE)
    baseline = CriticNet(input_size=[S_INFO, S_LEN], output_size=A_DIM, learning_rate=CRITIC_LR_RATE)

    # metalearner = MetaLearner(sampler, policy, baseline, gamma=args.gamma,
    #     fast_lr=args.fast_lr, tau=args.tau, device=args.device)

    for batch in range(args.num_batches):
        print("===================================================================")
        print("=====================Now epoch: ", batch, "========================")

        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
        sampler.reset_task(0.5)
        episodes = sampler.sample(policy, gamma=args.gamma, device=args.device)

        rewards = np.array(episodes.rewards)
        rewards = rewards.sum(0)
        mean_reward = rewards.mean()

        entropys = np.array(episodes.entropys).sum(0)
        mean_entropy = entropys.mean()

        values = baseline(episodes.observations)
        advantages = episodes.gae(values, tau=1)
        advantages = weighted_normalize(advantages, weights=episodes.mask)
        advantages = np.array(advantages).sum(0)
        mean_ad = advantages.mean()
        print("  mean Ad: ", mean_ad,
            "  mean reward:", mean_reward,
              "  mean_entropy: ", mean_entropy)

        # episodes = metalearner.sample(tasks, first_order=args.first_order)
        # metalearner.step(episodes, max_kl=args.max_kl, cg_iters=args.cg_iters,
        #     cg_damping=args.cg_damping, ls_max_steps=args.ls_max_steps,
        #     ls_backtrack_ratio=args.ls_backtrack_ratio)

        baseline.fit(episodes, CRITIC_LR_RATE)
        policy.fit(episodes, baseline, ACTOR_LR_RATE)

        if not batch % 4:
            noMetaTest(policy, baseline, batch)
            # print("total_rewards", total_rewards([ep.rewards for ep in episodes]))
            # print('total_rewards/after_update', total_rewards([ep.rewards for _, ep in episodes]))
            # for taskid in range(args.meta_bath_size):
            #     before = episodes[0][0].rewards
            #     print()

            # metaTest(policy, baseline, batch)

            # Save policy network
            with open(os.path.join(save_folder,
                    'policy-{0}.pt'.format(batch)), 'wb') as f:
                torch.save(policy.state_dict(), f)

            with open(os.path.join(save_folder,
                    'baseline-{0}.pt'.format(batch)), 'wb') as f:
                torch.save(baseline.state_dict(), f)



if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
        'Model-Agnostic Meta-Learning (MAML)')

    # General
    parser.add_argument('--env-name', type=str, default='pensieve',
        help='name of the environment')
    parser.add_argument('--gamma', type=float, default=0.95,
        help='value of the discount factor gamma')
    parser.add_argument('--tau', type=float, default=1.0,
        help='value of the discount factor for GAE')
    parser.add_argument('--first-order', action='store_true',
        help='use the first-order approximation of MAML')

    # Task-specific
    parser.add_argument('--fast-batch-size', type=int, default=20,
        help='batch size for each individual task')
    parser.add_argument('--fast-lr', type=float, default=0.5,
        help='learning rate for the 1-step gradient update of MAML')

    # Optimization
    parser.add_argument('--num-batches', type=int, default=20000,
        help='number of batches')
    parser.add_argument('--meta-batch-size', type=int, default=20,
        help='number of tasks per batch')
    parser.add_argument('--max-kl', type=float, default=1e-2,
        help='maximum value for the KL constraint in TRPO')
    parser.add_argument('--cg-iters', type=int, default=10,
        help='number of iterations of conjugate gradient')
    parser.add_argument('--cg-damping', type=float, default=1e-5,
        help='damping in conjugate gradient')
    parser.add_argument('--ls-max-steps', type=int, default=15,
        help='maximum number of iterations for line search')
    parser.add_argument('--ls-backtrack-ratio', type=float, default=0.8,
        help='maximum number of iterations for line search')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='maml',
        help='name of the output folder')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling')
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda)')

    args = parser.parse_args()

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./saves'):
        os.makedirs('./saves')
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    main(args)