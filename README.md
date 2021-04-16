# Reinforcement Learning for Aircraft War Game

## Environment Setup
Download requirements   
```sh
pip install -r requirements.txt
```

## Train

```sh
# DQN algorithm
python PlaneDQN.py

# DDQN algorithm
python PlaneDDQN.py

# PPO algorithm
python PlanePPO.py
```

## Evaluation
```sh
tensorboard --logdir ./tensorboard/
```

