import torch
import numpy as np
import os
import time
from Buffers import Buffer
from strategy_networks import StrategyNetworks, PolicyNet, loss_fn
from traversal import full_game_traversal, evaluate_traversal
import ray

from hanabi_learning_environment import rl_env
from hanabi_learning_environment import pyhanabi


env = rl_env.HanabiEnv(
        config={
            "colors":
                1,
            "ranks":
                2,
            "players":
                2,
            "hand_size":
                2,
            "max_information_tokens":
                3,
            "max_life_tokens":
                1,
            "observation_type":
                pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
        })
d_input = env.vectorized_observation_shape()[0]
d_output = env.num_moves()


def generate_local_buffers(buffer_size=int(1e6)):
    advantage_buffer_p0 = Buffer(int(buffer_size), obs_size=d_input, action_size=d_output)
    advantage_buffer_p1 = Buffer(int(buffer_size), obs_size=d_input, action_size=d_output)

    advantage_buff_arr = [advantage_buffer_p0, advantage_buffer_p1]

    # policy_buffer = Buffer(2 * int(buffer_size))

    return advantage_buff_arr, []


def initialize_networks():
    strategy_networks = StrategyNetworks(obs_size=d_input, action_size=d_output)
    policy_net = PolicyNet

    return strategy_networks, policy_net

@ray.remote(num_gpus=0.2)
def local_game_traversal(k, p, net_arr, t, t_id):
    advantage_buffer_arr, policy_buffer = generate_local_buffers()

    start_time = time.time()

    for i in range(1, k + 1):
        full_game_traversal(p, net_arr, advantage_buffer_arr, policy_buffer, t)
        if i % 1e2 == 0:
            print('Iteration {}, traversal {}, traversal id{}, time {:.2f}s, player {}, '
                  'with buffer_size M0:{} M1:{} Ms:{}'.format(t, i, t_id, time.time() - start_time, p,
                                                              len(advantage_buffer_arr[0]),
                                                              len(advantage_buffer_arr[1]), len(policy_buffer)))
            start_time = time.time()

    return [advantage_buffer_arr, policy_buffer]


@ray.remote(num_gpus=0.2,)
def evaluate(net_arr, K=int(1e2)):
    reward = 0
    for i in range(K):
        env = rl_env.HanabiEnv(
            config={
                "colors":
                    1,
                "ranks":
                    2,
                "players":
                    2,
                "hand_size":
                    2,
                "max_information_tokens":
                    3,
                "max_life_tokens":
                    1,
                "observation_type":
                    pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
            })
        _ = env.reset()
        obs_h = [[]] * env.players
        reward += evaluate_traversal(obs_h.copy(), env, net_arr)
    return reward / K


def train(K=int(1e3), max_t=int(1e5)):
    ray.init(num_gpus=1)
    Advantage_buffer_arr, Policy_buffer = generate_local_buffers()
    strategy_networks, policy_net = initialize_networks()
    print('Start Deep CFR...')
    last_buffer_length = [0, 0]
    opt = [torch.optim.Adam(strategy_networks.net_arr[0].parameters(), lr=1e-3), torch.optim.Adam(strategy_networks.net_arr[1].parameters(), lr=1e-3)]
    values = []
    for t in range(1, max_t + 1):
        for p in range(2):
            traversal_time = time.time()
            t_ids = [local_game_traversal.remote(int(K/10), p, strategy_networks, t, t_id) for t_id in range(10)]
            for t_id in t_ids:
                Advantage_buffer_arr[p].add_buffer(ray.get(t_id)[0][p])

            print('FINISHED traversal for player id:{}, and Iteration {}, Number of nodes sampled: {}, takes {}s'.format(p, t, len(Advantage_buffer_arr[p])-last_buffer_length[p], time.time() - traversal_time))
            last_buffer_length[p] = len(Advantage_buffer_arr[p])

            training_time = time.time()
            strategy_networks.net_arr[p] = train_advantage_net(strategy_networks.net_arr[p], loss_fn, Advantage_buffer_arr, p, opt[p])

            print('FINISHED training for player id:{}, takes {}s'.format(p, time.time() - training_time))

            if t % 10 == 0:
                torch.save(strategy_networks.net_arr[p].state_dict(), './checkpoints_ray/strategy' + str(p) + "_" + str(t) + '.model')

        print('FINISHED ITERATION ', t)
        evaluate_time = time.time()
        e_ids = [evaluate.remote(strategy_networks) for _ in range(10)]
        test_value = sum(ray.get(e_ids))/10
        values.append(test_value)
        print('@' * 100)
        print('        Evaluate for strategy_networks: {}, takes {}s'.format(test_value, time.time() - evaluate_time))
        print('@' * 100)
        if t%10==0 and t:
            np.save('./test_value/{}.npy'.format(t), np.array(values))
    # train_policy_net(policy_net, loss_fn, Policy_buffer)

'''@ray.remote(num_gpus=1,)'''
def train_advantage_net(net, loss_fn, advantage_buffer_arr, p, opt):
    batch_size = 2000
    n_minibatch = min(max(int(len(advantage_buffer_arr[p]) / batch_size), 1), 100)
    n_epoch = 5
    # net.to('cuda:0')
    net.train()

    print('#' * 100)
    print("        training for net {} with Buffer{} size: {}".format(p, p, len(advantage_buffer_arr[p])))
    print('#' * 100)

    for epoch in range(n_epoch):
        running_loss = 0.0

        for minibatch in range(n_minibatch):
            obs_h, t, arr, length = advantage_buffer_arr[p].recast(batch_size)

            # x_train.to('cuda:0')
            # t.to('cuda:0')
            # advantage.to('cuda:0')

            opt.zero_grad()

            pred_advantage = net(obs_h, length)
            loss = loss_fn(pred_advantage, t, arr)
            loss.backward()
            opt.step()

            running_loss += loss.item()

            # del x_train
            # del t
            # del advantage

            # print('Running loss after minibatch %d: %.3f' % (minibatch, running_loss / ((minibatch + 1) * batch_size)))

            if minibatch % 50 == 49:
                print('    Epoch: %d, minibatch_id: %d. Loss: %.3f' % (epoch + 1, minibatch, running_loss / 50))
                running_loss = 0.0
        if n_minibatch % 50:
            print('    Epoch: %d, minibatch_id: %d. Loss: %.3f' % (
                epoch + 1, n_minibatch, running_loss / (n_minibatch % 50)))

    return net


def train_policy_net(policy_net, loss_fn, policy_buffer):
    batch_size = 2000
    n_minibatch = max(int(len(policy_buffer) / batch_size), 1)
    n_epoch = 12
    policy_net.train()

    opt = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
    print('#' * 25)
    print("Training for policy_net with policy_buffer size: {}".format(len(policy_buffer)))
    print('#' * 25)

    for epoch in range(n_epoch):
        running_loss = 0.0

        for minibatch in range(n_minibatch):
            obs_h, t, arr, length = policy_buffer.recast(batch_size)

            opt.zero_grad()

            pred_strategy = policy_net(obs_h, length)
            loss = loss_fn(pred_strategy, t, arr, 1)
            loss.backward()
            opt.step()

            running_loss += loss.item()

            # print('Running loss after minibatch %d: %.3f' % (minibatch, running_loss / ((minibatch + 1) * batch_size)))

            if minibatch % 50 == 49:
                print('    Epoch: %d, minibatch_id: %d. Loss: %.3f' % (epoch + 1, minibatch, running_loss / 50))
                running_loss = 0.0
        if n_minibatch < 50:
            print('    Epoch: %d, minibatch_id: %d. Loss: %.3f' % (
                epoch + 1, n_minibatch - 1, running_loss / n_minibatch))

    torch.save(policy_net.state_dict(), './checkpoints/policy_net')


if __name__ == "__main__":
    train()
