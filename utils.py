import os
import cv2
import numpy as np
import torch


class SumTree:

    def __init__(self, capacity):

        self.capacity = capacity
        # the first capacity-1 positions are not leaves
        self.vals = [0 for _ in range(2 * capacity - 1)]  # think about why if you are not familiar with this

    def retrive(self, num):
        '''
        This function find the first index whose cumsum is no smaller than num
        '''
        ind = 0  # search from root
        while ind < self.capacity - 1:  # not a leaf
            left = 2 * ind + 1
            right = left + 1
            if num > self.vals[left]:  # the sum of the whole left tree is not large enouth
                num -= self.vals[left]  # think about why?
                ind = right
            else:  # search in the left tree
                ind = left
        return ind - self.capacity + 1

    def update(self, delta, ind):
        '''
        Change the value at ind by delta, and update the tree
        Notice that this ind should be the index in real memory part, instead of the ind in self.vals
        '''
        ind += self.capacity - 1
        while True:
            self.vals[ind] += delta
            if ind == 0:
                break
            ind -= 1
            ind //= 2


# preprocess raw image to 80*80 gray image
def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)  # 灰度转化
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (80, 80))


def _get_path(algorithm):
    fs = os.listdir('saved_networks/{}/'.format(algorithm))
    fs = sorted(fs, key=lambda x: int(x.split('-')[-1]), reverse=False)
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
