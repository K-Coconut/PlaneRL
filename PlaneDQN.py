import cv2
import sys

sys.path.append("game/")
import plane as game
from BrainDQN_Nature import BrainDQN
import tensorflow as tf
import numpy as np


# preprocess raw image to 80*80 gray image
def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)  # 灰度转化
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (80, 80, 1))


def playPlane():
    # Step 1: init BrainDQN
    actions = 3
    brain = BrainDQN(actions)
    summary_writer = tf.summary.FileWriter('./tensorboard', brain.session.graph)
    # Step 2: init Plane Game
    plane = game.GameState()
    # Step 3: play game
    # Step 3.1: obtain init state
    action0 = np.array([1, 0, 0])  # [1,0,0]do nothing,[0,1,0]left,[0,0,1]right
    observation0, reward0, terminal = plane.frame_step(action0)

    observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation0 = cv2.threshold(observation0, 1, 255, cv2.THRESH_BINARY)
    brain.setInitState(observation0)

    # Step 3.2: run the game
    step = 0
    round = 0
    while 1 != 0:
        action = brain.getAction()
        nextObservation, reward, terminal = plane.frame_step(action)
        nextObservation = preprocess(nextObservation)
        summary_reward = tf.Summary(value=[tf.Summary.Value(tag="reward", simple_value=reward)])
        summary_writer.add_summary(summary_reward, step)
        if terminal:
            score = tf.Summary(value=[tf.Summary.Value(tag="score", simple_value=plane.score)])
            print('score: ', plane.score)
            summary_writer.add_summary(score, round)
            round += 1
        brain.setPerception(nextObservation, action, reward, terminal)
        step += 1


def main():
    playPlane()


if __name__ == '__main__':
    main()
