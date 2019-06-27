import torch
import multiprocessing as mp
import numpy as np
from collections import OrderedDict

from maml_rl.envs.subproc_vec_env import SubprocVecEnv
from maml_rl.episode import BatchEpisodes
from maml_rl.policies import ActorNet, CriticNet

import sim.load_trace as load_trace
import sim.env as env


A_DIM = 6
NUM_AGENTS = 4
S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
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
LOG_FILE = './maml_rl/agents_log/log'


def agent(agent_id, all_cooked_time, all_cooked_bw, net_params_queue, exp_queue, tasks_queue):
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              random_seed=agent_id)

    with open(LOG_FILE + '_agent_' + str(agent_id), 'w') as log_file:
        actor = ActorNet(input_size=[S_INFO, S_LEN], output_size=A_DIM)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params = net_params_queue.get()
        ################################
        #  task use
        ####################################
        task = tasks_queue.get()

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        time_stamp = 0
        while True:  # experience video streaming forever
            with torch.no_grad():
                # the action is from the last decision
                # this is to make the framework similar to the real
                delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, \
                end_of_video, video_chunk_remain = \
                    net_env.get_video_chunk(bit_rate)

                time_stamp += delay  # in ms
                time_stamp += sleep_time  # in ms

                ##########################
                # Different Rewards----------
                #############################
                if task<0.33:
                    # -- linear reward --
                    # reward is video quality - rebuffer penalty - smoothness
                    reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                             - REBUF_PENALTY * rebuf \
                             - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                                       VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
                elif task<0.66:
                    # -- log scale reward --
                        log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[-1]))
                        log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[-1]))
                        reward = log_bit_rate \
                                 - REBUF_PENALTY * rebuf \
                                 - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)
                else:
                    # -- HD reward --
                    reward = HD_REWARD[bit_rate] \
                             - REBUF_PENALTY * rebuf \
                             - SMOOTH_PENALTY * np.abs(HD_REWARD[bit_rate] - HD_REWARD[last_bit_rate])

                #
                # # -- linear reward --
                # # reward is video quality - rebuffer penalty - smoothness
                # reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                #          - (0.5+task)*REBUF_PENALTY * rebuf \
                #          - (1.5-task)*SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                #                                    VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K


                r_batch.append(reward)

                last_bit_rate = bit_rate

                # retrieve previous state
                if len(s_batch) == 0:
                    state = [np.zeros((S_INFO, S_LEN))]
                else:
                    state = np.array(s_batch[-1], copy=True)

                # dequeue history record
                state = np.roll(state, -1, axis=1)

                # this should be S_INFO number of terms
                state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
                state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
                state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
                state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
                state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
                state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

                # compute action probability vector
                action_categorical = actor(np.reshape(state, (1, 1, S_INFO, S_LEN)), params=actor_net_params)
                bit_rate = int(action_categorical.sample())
                # Note: we need to discretize the probability into 1/RAND_RANGE steps,
                # because there is an intrinsic discrepancy in passing single state and batch states


                # log time_stamp, bit_rate, buffer_size, reward
                log_file.write(str(time_stamp) + '\t' +
                               str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                               str(buffer_size) + '\t' +
                               str(rebuf) + '\t' +
                               str(video_chunk_size) + '\t' +
                               str(delay) + '\t' +
                               str(reward) + '\n')
                log_file.flush()

                # report experience to the coordinator
                if len(r_batch) >= TRAIN_SEQ_LEN or end_of_video:
                    exp_queue.put([s_batch[1:],  # ignore the first chuck
                                   a_batch[1:],  # since we don't have the
                                   r_batch[1:],  # control over it
                                   end_of_video,
                                   {'entropy': entropy_record}])

                    # synchronize the network parameters from the coordinator
                    # block if necessary until an item is available
                    actor_net_params = net_params_queue.get()
                    ################################
                    #  task use
                    ####################################
                    task = tasks_queue.get()

                    del s_batch[:]
                    del a_batch[:]
                    del r_batch[:]
                    del entropy_record[:]

                    log_file.write('\n')  # so that in the log we know where video ends

                # store the state and action into batches
                if end_of_video:
                    last_bit_rate = DEFAULT_QUALITY
                    bit_rate = DEFAULT_QUALITY  # use the default action here

                    # action_vec = np.zeros(A_DIM)
                    # action_vec[bit_rate] = 1

                    s_batch.append(np.zeros((S_INFO, S_LEN)))
                    a_batch.append(np.array(bit_rate))

                else:
                    s_batch.append(state)
                    entropy_record.append(action_categorical.entropy().squeeze())
                    # action_vec = np.zeros(A_DIM)
                    # action_vec[bit_rate] = 1
                    a_batch.append(np.array(bit_rate))


class MrcBatchSampler(object):
    def __init__(self, env_name, batch_size, train_folder):
        assert(env_name == 'pensieve')
        self.env_name = env_name
        self.batch_size = batch_size
        self.num_workers = NUM_AGENTS
        self.tasks = []

        self.net_params_queues = []
        self.exp_queues = []
        self.tasks_queues = []
        for i in range(NUM_AGENTS):
            self.net_params_queues.append(mp.Queue(1))
            self.exp_queues.append(mp.Queue(1))
            self.tasks_queues.append(mp.Queue(1))

        self.agents = []
        all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(train_folder)
        for i in range(NUM_AGENTS):
            self.agents.append(mp.Process(target=agent,
                                     args=(i, all_cooked_time, all_cooked_bw,
                                           self.net_params_queues[i],
                                           self.exp_queues[i],
                                           self.tasks_queues[i])))

        for i in range(NUM_AGENTS):
            self.agents[i].start()

        np.random.seed(RANDOM_SEED)

    def sample(self, policy, params=None, gamma=0.95, device='cpu'):
        episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
        if params is None:
            actor_params = OrderedDict(policy.named_parameters())
        else:
            actor_params = dict()
            for name in params:
                actor_params[name] = params[name].detach()

        bid = 0
        while True:
            for i in range(NUM_AGENTS):
                # block if necessary until a free slot is available
                self.net_params_queues[i].put(actor_params)
                self.tasks_queues[i].put(self.tasks[i])
            for i in range(NUM_AGENTS):
                s_batch, a_batch, r_batch, terminal, info = self.exp_queues[i].get()
                episodes.append(s_batch, a_batch, r_batch, info, bid)
                bid += 1
                if bid == self.batch_size:
                    break
            if bid == self.batch_size:
                break

        return episodes

    def reset_task(self, task):
        for i in range(NUM_AGENTS):
            self.tasks.append(task)
        return True

    def sample_tasks(self, num_tasks):
        # tasks = self._env.unwrapped.sample_tasks(num_tasks)

        ###########################################################
        # need change~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ##########################################################
        tasks = []
        for i in range(num_tasks):
            tasks.append(np.random.rand())
        return tasks

