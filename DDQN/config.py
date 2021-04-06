import torch

GPU = 0
DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

# Agent parameters
BATCH_SIZE = 128
LEARNING_RATE = 0.001
TAU = 0.001
GAMMA = 0.99
DUEL = False
DOUBLE = True
PRIORITIZED = False

# Replay buffer parameters
alpha = 0.5
epsilon = 0.05
td_init = 1

# Training parameters
RAM_NUM_EPISODE = 100000
EPS_INIT = 1
EPS_DECAY = 0.995
EPS_MIN = 0.05
MAX_T = 5000
NUM_FRAME = 2
