import numpy as np
import os
import time
from hanabi_learning_environment import rl_env
from hanabi_learning_environment import pyhanabi

start_time = time.time()

out_come_sampling = True
epsilon = 0.6


def full_game_traversal(pid, net_arr, advantage_buffer_arr, strategy_buffer, t):
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
    traverse(obs_h.copy(), env, pid, net_arr, advantage_buffer_arr, strategy_buffer, t)


def traverse(obs_h, env, pid, net_arr, advantage_buffer_arr, strategy_buffer, t):
    if env.state.is_terminal():
        return env.state.score()

    if env.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
        env.state.deal_random_card()
        return traverse(obs_h, env, pid, net_arr, advantage_buffer_arr, strategy_buffer, t)

    if env.state.cur_player() == pid:

        obs_h1 = obs_h.copy()
        for i in range(env.players):
            obs_h1[i] = obs_h[i] + [env.observation_encoder.encode(env.state.observation(i))]

        softmax_values, clip_values = net_arr.get_values(obs_h1[pid], pid)

        if out_come_sampling:
            regret_arr = np.zeros(env.game.max_moves())

            actions = [i for i in range(env.game.max_moves())]

            avail_actions = np.zeros(env.game.max_moves())
            for a in range(env.game.max_moves()):
                if env.state.move_is_legal(env.game.get_move(a)):
                    avail_actions[a] = 1

            if max(clip_values) == 0:
                avail_softmax_strategy = np.multiply(softmax_values, avail_actions)
                action = np.argmax(avail_softmax_strategy)
                strategy = np.zeros(env.game.max_moves())
                strategy[action] = 1
            else:
                strategy = np.multiply(clip_values, avail_actions)
                if sum(strategy) < 1e-5:
                    strategy = avail_actions / sum(avail_actions)
                else:
                    strategy /= sum(strategy)

            if np.random.uniform() < epsilon:
                pro = avail_actions / sum(avail_actions)
                action = np.random.choice(actions, p=pro)
            else:
                pro = strategy
                action = np.random.choice(actions, p=pro)

            env.state.apply_move(env.game.get_move(action))
            regret_arr[action] = traverse(obs_h1, env, pid, net_arr, advantage_buffer_arr, strategy_buffer, t) / pro[action]
            value = regret_arr[action] * strategy[action]
            regret_arr -= value
            advantage_buffer_arr[pid].add(obs_h1[pid], t, regret_arr)

            return value

        else:
            regret_arr = np.zeros(env.game.max_moves())

            actions = [i for i in range(env.game.max_moves())]

            for a in actions:
                if env.state.move_is_legal(env.game.get_move(a)):
                    new_env = env.copy()
                    new_env.state.apply_move(new_env.game.get_move(a))
                    regret_arr[a] = traverse(obs_h1, new_env, pid, net_arr, advantage_buffer_arr, strategy_buffer, t)

            avail_actions = np.zeros(env.game.max_moves())
            for a in range(env.game.max_moves()):
                if env.state.move_is_legal(env.game.get_move(a)):
                    avail_actions[a] = 1

            if max(clip_values) == 0:
                strategy = avail_actions/sum(avail_actions)
            else:
                strategy = np.multiply(clip_values, avail_actions)
                if sum(strategy) < 1e-5:
                    strategy = avail_actions / sum(avail_actions)
                else:
                    strategy /= sum(strategy)

            mean_value = np.dot(strategy, regret_arr)
            regret_arr -= mean_value
            advantage_buffer_arr[pid].add(obs_h1[pid], t, regret_arr)

            return mean_value

    else:
        opp_idx = env.state.cur_player()

        obs_h1 = obs_h.copy()
        for i in range(env.players):
            obs_h1[i] = obs_h[i] + [env.observation_encoder.encode(env.state.observation(i))]

        softmax_values, clip_values = net_arr.get_values(obs_h1[opp_idx], opp_idx)

        # strategy_buffer.add(obs_h1[opp_idx], t, strategy)

        actions = [i for i in range(env.game.max_moves())]

        avail_actions = np.zeros(env.game.max_moves())
        for a in range(env.game.max_moves()):
            if env.state.move_is_legal(env.game.get_move(a)):
                avail_actions[a] = 1

        if max(clip_values) == 0:
            avail_softmax_strategy = np.multiply(softmax_values, avail_actions)
            action = np.argmax(avail_softmax_strategy)
            strategy = np.zeros(env.game.max_moves())
            strategy[action] = 1
        else:
            strategy = np.multiply(clip_values, avail_actions)
            if sum(strategy) < 1e-5:
                strategy = avail_actions / sum(avail_actions)
            else:
                strategy /= sum(strategy)
            action = np.random.choice(actions, p=strategy)
        env.state.apply_move(env.game.get_move(action))

        return traverse(obs_h1, env, pid, net_arr, advantage_buffer_arr, strategy_buffer, t)


def evaluate_traversal(obs_h, env, net_arr):
    if env.state.is_terminal():
        return env.state.score()
    if env.state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
        env.state.deal_random_card()
        return evaluate_traversal(obs_h, env, net_arr)
    else:
        pid = env.state.cur_player()

        obs_h1 = obs_h.copy()
        for i in range(env.players):
            obs_h1[i] = obs_h[i] + [env.observation_encoder.encode(env.state.observation(i))]

        softmax_values, clip_values = net_arr.get_values(obs_h1[pid], pid)

        actions = [i for i in range(env.game.max_moves())]

        avail_actions = np.zeros(env.game.max_moves())
        for a in range(env.game.max_moves()):
            if env.state.move_is_legal(env.game.get_move(a)):
                avail_actions[a] = 1

        avail_softmax_strategy = np.multiply(softmax_values, avail_actions)
        action = np.argmax(avail_softmax_strategy)

        env.state.apply_move(env.game.get_move(actions[action]))
        return evaluate_traversal(obs_h1, env, net_arr)
