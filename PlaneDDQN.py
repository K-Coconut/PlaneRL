import os
from tqdm import tqdm
from datetime import datetime
import numpy as np
from collections import deque

from tensorboardX import SummaryWriter

from game import plane as game
from DDQN.agent import Agent
from DDQN.config import *

from utils import preprocess, save_dict, load_dict

ALGORITHM = 'DDQN'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
TIMESTAMP = "{0:%Y-%m-%d %H-%M-%S/}".format(datetime.now())
log_tensorboard = './tensorboard/{}/{}'.format(ALGORITHM, TIMESTAMP)


def train(agent, n_episode, eps_init, eps_decay, eps_min, max_t, num_frame=2):
    rewards_log = []
    average_log = []
    eps = eps_init

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
        state = np.stack(state_deque, axis=0)
        state = np.expand_dims(state, axis=0)
        episode_reward = 0
        episode_score = 0
        terminal = False
        step = 0

        while not terminal and step < max_t:

            action = agent.act(state, eps)
            action = [1 if i == action else 0 for i in range(3)]
            next_observation, reward, done = plane.frame_step(action)
            state_deque.append(preprocess(next_observation))
            next_state = np.stack(state_deque, axis=0)
            next_state = np.expand_dims(next_state, axis=0)
            agent.memory.remember(state, action, reward, next_state, done)

            if step % 5 == 0 and len(agent.memory) >= agent.bs:
                agent.learn()
                agent.soft_update(agent.tau)

            state = next_state.copy()
            episode_reward += reward
            if not terminal:
                episode_score = plane.score
            step += 1

        eps = max(eps * eps_decay, eps_min)

        rewards_log.append(episode_reward)
        average_log.append(np.mean(rewards_log[-100:]))

        summary_writer.add_scalar('reward', episode_reward, i)
        summary_writer.add_scalar('average reward', np.mean(rewards_log[-100:]), i)
        summary_writer.add_scalar('score', episode_score, i)

        postfix = dict(Epoch=i, score=episode_score, reward=episode_reward, avg_reward=np.mean(rewards_log[-100:]))
        pb.set_postfix(postfix)

        if i % 300 == 0:
            save_dict(agent, ALGORITHM, i)

    return rewards_log, average_log

def main():
    os.system('mkdir -p saved_networks/{}/'.format(ALGORITHM))

    n_action = 3
    agent = Agent(state_size=NUM_FRAME,
                  action_size=n_action,
                  bs=BATCH_SIZE,
                  lr=LEARNING_RATE,
                  tau=TAU,
                  gamma=GAMMA,
                  device=DEVICE,
                  double=True)

    train(agent, RAM_NUM_EPISODE, EPS_INIT, EPS_DECAY, EPS_MIN, MAX_T, NUM_FRAME)


if __name__ == '__main__':
    main()