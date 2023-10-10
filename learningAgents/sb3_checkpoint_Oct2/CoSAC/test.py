import classes as cl
import globals as gl
from environments import ConPricingGame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import SAC

names = ["CoSAC-1694966441", "CoSAC-1695131273", "CoSAC-1695131273", "CoSAC-1694966441", "CoSAC-1694966441", "CoSAC-1694655185", "CoSAC-1694655185",
         "CoSAC-1694655185", "CoSAC-1694461629", "CoSAC-1694461629", "CoSAC-1694461629", "CoSAC-1694164525", "CoSAC-1694164525", "CoSAC-1694164525", "CoSAC-1695131273"]
for name in names:
    policy= (SAC.load("models/"+name,env=ConPricingGame)).predict

    