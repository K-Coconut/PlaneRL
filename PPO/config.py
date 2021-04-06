import torch

GPU = 0
DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

# Agent parameters
LEARNING_RATE = 0.005
GAMMA = 0.99
BETA = 0
EPS = 0.2
TAU = 0.99
MODE = 'TD'
SHARE = False
CRITIC = True
NORMALIZE = False
HIDDEN_DISCRETE = [128]

# Training parameters
RAM_NUM_EPISODE = 2000
VISUAL_NUM_EPISODE = 5000
SCALE = 1
MAX_T = 2000
NUM_FRAME = 2
N_UPDATE = 4
UPDATE_FREQUENCY = 4
