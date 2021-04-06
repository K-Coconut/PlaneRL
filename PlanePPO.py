import os
from tqdm import tqdm
from datetime import datetime
import numpy as np
from collections import deque

from tensorboardX import SummaryWriter

from game import plane as game
from PPO.agent import Agent
from PPO.config import *
from utils import preprocess, save_dict, load_dict

ALGORITHM = 'PPO'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
TIMESTAMP = "{0:%Y-%m-%d %H-%M-%S/}".format(datetime.now())
log_tensorboard = './tensorboard/{}/{}'.format(ALGORITHM, TIMESTAMP)


def train(agent, n_episode, num_frame, n_update=4, update_frequency=1, max_t=1500, scale=1):
    rewards_log = []
    average_log = []
    state_history = []
    action_history = []
    done_history = []
    reward_history = []

    plane = game.GameState()
    summary_writer = SummaryWriter(log_tensorboard)

    agent, last_trained_epoch = load_dict(agent, ALGORITHM)
    pb = tqdm(range(last_trained_epoch + 1, last_trained_epoch + n_episode + 1), desc=ALGORITHM, unit='ep',
              total=n_episode)
    for i in pb:

        observation = preprocess(plane.reset())
        state_deque = deque(maxlen=num_frame)
        for _ in range(num_frame):
            state_deque.append(observation)

        if len(state_history) == 0:
            state_history.append(list(state_deque))
        else:
            state_history[-1] = list(state_deque)

        episode_reward = 0
        episode_score = 0
        terminal = False
        step = 0

        while not terminal and step < max_t:
            action = agent.act(state_deque)
            action = [1 if i == action else 0 for i in range(3)]
            nextObservation, reward, terminal = plane.frame_step(action)

            episode_reward += reward
            if not terminal:
                episode_score = plane.score
            action_history.append(action)
            done_history.append(terminal)
            reward_history.append(reward * scale)
            state_deque.append(preprocess(nextObservation))
            state_history.append(list(state_deque))
            step += 1

        if i % update_frequency == 0:
            states, actions, log_probs, rewards, dones = agent.process_data(state_history,
                                                                            action_history,
                                                                            reward_history,
                                                                            done_history, 64)
            for _ in range(n_update):
                agent.learn(states, actions, log_probs, rewards, dones)
            state_history = []
            action_history = []
            done_history = []
            reward_history = []

            for name, layer in agent.Actor.named_parameters():
                if 'bn' in name: continue
                summary_writer.add_histogram(name + '_data_weight', layer, i)

        rewards_log.append(episode_reward)
        average_log.append(np.mean(rewards_log[-100:]))

        summary_writer.add_scalar('reward', episode_reward, i)
        summary_writer.add_scalar('average reward', np.mean(rewards_log[-100:]), i)
        summary_writer.add_scalar('score', episode_score, i)

        postfix = dict(Epoch=i, score=episode_score, reward=episode_reward, avg_reward=np.mean(rewards_log[-100:]))
        pb.set_postfix(postfix)

        if i % 300 == 0:
            save_dict(agent, ALGORITHM, i)


def main():
    os.system('mkdir -p saved_networks/{}/'.format(ALGORITHM))

    n_action = 3
    n_observation_space = preprocess(game.GameState().reset()).shape

    agent = Agent(state_size=n_observation_space,
                  action_size=n_action,
                  lr=LEARNING_RATE,
                  beta=BETA,
                  eps=EPS,
                  tau=TAU,
                  gamma=GAMMA,
                  device=DEVICE,
                  num_frame=NUM_FRAME,
                  share=SHARE,
                  mode=MODE,
                  use_critic=CRITIC,
                  normalize=NORMALIZE)

    train(agent, n_episode=RAM_NUM_EPISODE, num_frame=NUM_FRAME, n_update=N_UPDATE, update_frequency=UPDATE_FREQUENCY,
          max_t=MAX_T,
          scale=SCALE)


if __name__ == '__main__':
    main()
