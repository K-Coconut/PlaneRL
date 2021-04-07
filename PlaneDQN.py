import os
from datetime import datetime
from tqdm import tqdm
import cv2
import numpy as np

import tensorflow.compat.v1 as tf

from game import plane as game
from DQN.BrainDQN import BrainDQN
from utils import preprocess

tf.disable_eager_execution()

ALGORITHM = 'DQN'
RAM_NUM_EPISODE = 100000
MAX_T = 5000

TIMESTAMP = "{0:%Y-%m-%d %H-%M-%S/}".format(datetime.now())
log_tensorboard = './tensorboard/{}/{}'.format(ALGORITHM, TIMESTAMP)


def playPlane(n_episode, max_t):
    # Step 1: init BrainDQN
    actions = 3
    brain = BrainDQN(actions)
    summary_writer = tf.summary.FileWriter(log_tensorboard, brain.session.graph)
    # Step 2: init Plane Game
    plane = game.GameState()
    # Step 3: play game
    rewards_log = []
    pb = tqdm(range(1, n_episode + 1), desc=ALGORITHM, unit='ep',
              total=n_episode)
    for i in pb:

        # Step 3.1: obtain init state
        action0 = np.array([1, 0, 0])  # [1,0,0]do nothing,[0,1,0]left,[0,0,1]right
        observation0, reward0, terminal = plane.frame_step(action0)

        observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, observation0 = cv2.threshold(observation0, 1, 255, cv2.THRESH_BINARY)
        brain.setInitState(observation0)

        # Step 3.2: run the game
        step = 0
        episode_reward = 0
        episode_score = 0
        terminal = False

        while not terminal and step < max_t:
            action = brain.getAction()
            nextObservation, reward, terminal = plane.frame_step(action)
            nextObservation = np.expand_dims(preprocess(nextObservation), -1)

            episode_reward += reward
            if not terminal:
                episode_score = plane.score
            brain.setPerception(nextObservation, action, reward, terminal)
            step += 1

        rewards_log.append(episode_reward)

        summary_reward = tf.Summary(value=[tf.Summary.Value(tag="reward", simple_value=episode_reward)])
        summary_avg_reward = tf.Summary(value=[tf.Summary.Value(tag="average_reward", simple_value=np.mean(rewards_log[-100:]))])
        summary_score = tf.Summary(value=[tf.Summary.Value(tag="score", simple_value=episode_score)])
        summary_writer.add_summary(summary_reward, i)
        summary_writer.add_summary(summary_avg_reward, i)
        summary_writer.add_summary(summary_score, i)

        postfix = dict(Epoch=i, score=episode_score, reward=episode_reward, avg_reward=np.mean(rewards_log[-100:]))
        pb.set_postfix(postfix)


def main():
    os.system('mkdir -p saved_networks/{}'.format(ALGORITHM))
    playPlane(RAM_NUM_EPISODE, MAX_T)


if __name__ == '__main__':
    main()
