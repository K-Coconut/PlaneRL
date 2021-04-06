import os
import cv2
import numpy as np
import torch


# preprocess raw image to 80*80 gray image
def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)  # 灰度转化
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (80, 80))


def _get_path(algorithm):
    fs = os.listdir('saved_networks/{}/'.format(algorithm))
    fs = sorted(fs, key=lambda x: x.split('-')[-1], reverse=False)
    if not fs: return None, 0
    epoch = int(fs[-1].split('-')[-1])
    path = 'saved_networks/{}/network-{}'.format(algorithm, epoch)
    return path, epoch


def save_dict(model, algorithm, epoch):
    path = 'saved_networks/{}/network-{}'.format(algorithm, epoch)
    torch.save(model.state_dict(), path)


def load_dict(model, algorithm):
    path, epoch = _get_path(algorithm)
    if not path: return model, epoch
    print('Loading model: {}'.format(path))
    model.load_state_dict(torch.load(path))
    model.train()
    return model, epoch


def set_seeds(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
