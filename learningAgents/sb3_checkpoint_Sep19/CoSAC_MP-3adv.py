from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import gymnasium as gym
import numpy as np
import globals as gl
import classes as cl
import time
import os
# from gymnasium_env import ContinousPricingGame
from Environment import ConPricingGame
import torch


def run(lr=None):    
    seed=int(time.time())

    iter_name = f"{model_name}-{str(seed)}"
    models_dir = os.path.join("models", iter_name)
    log_dir = os.path.join("logs", iter_name)
             
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
             
    train_env = make_vec_env(ConPricingGame, n_envs=num_procs, seed=seed, vec_env_cls=SubprocVecEnv,env_kwargs=\
                           dict(tuple_costs = costs,adversary_mixed_strategy = adv_mixed_strategy))
#     train_env = make_vec_env(ContinousPricingGame, n_envs=num_procs, seed=0, vec_env_cls=SubprocVecEnv,\
#                              env_kwargs=dict(tuple_cost=costs))

    lr_=(gl.LR if (lr is None) else lr)
#     train_env = ContinousPricingGame()
#     train_env.reset()
    
#     train_env.reset()
    model_ = model('MlpPolicy', train_env,learning_rate=lr_,verbose=0, tensorboard_log=log_dir, gamma=gl.GAMMA)
    
    start=time.time()
    for i in range(num_timesteps):
        model_.learn(total_timesteps=timesteps,
                     reset_num_timesteps=False,tb_log_name=iter_name)
    model_.save(os.path.join(models_dir, str(timesteps*i)))
    running_time=time.time()- start

    # test and write results
    env = ConPricingGame(tuple_costs=costs, adversary_mixed_strategy=adv_mixed_strategy)
    for iter in range(gl.NUM_STOCHASTIC_ITER):
             
        obs,_ = env.reset()
        done = False

        actions = []
        while not done:
            action, _states = model_.predict(obs)
            obs, reward, done,trunc, info = env.step(action)

            actions.append(int(action))
        # name	ep	costs	adversary	agent_returnC:\Users\sjaha\Desktop\RL\Gym\stableBaseline3\environment.py	adv_return	agent_rewards	actions	agent_prices	\
        # adv_prices	agent_demands	adv_demands	lr	hist	total_stages	action_step	num_actions	gamma	\
        # stae_onehot	seed	num_procs	running_time
        data=[iter_name, timesteps*num_timesteps,("L" if (costs[0]<costs[1]) else "H"), env.adversary_strategy.name,\
              sum(env.profit[0]), sum(env.profit[1]),  str(env.profit[0]), str(actions), str(env.prices[0]),\
              str(env.prices[1]), str(env.demand_potential[0]),str(env.demand_potential[1]), lr_, gl.NUM_ADV_HISTORY,\
              gl.TOTAL_STAGES, gl.ACTION_STEP, gl.NUM_ACTIONS, gl.GAMMA, False, seed, num_procs, running_time]
        cl.write_to_excel(data)
    print(lr, "=lr completed")


gl.initialize()

num_procs=3
model = SAC
model_name="CoSAC_MP_3"
timesteps= 1_000_000
num_timesteps=10
state_onehot=False
costs=[gl.LOW_COST, gl.HIGH_COST]




strt1=cl.Strategy(
        cl.StrategyType.static, model_or_func=cl.myopic, name="myopic")
strt2=cl.Strategy(
        cl.StrategyType.static, model_or_func=cl.guess, name="guess",first_price=132)
strt3=cl.Strategy(
        cl.StrategyType.static, model_or_func=cl.imit, name="imit",first_price=132)

adv_mixed_strategy = cl.MixedStrategy([strt1,strt2,strt3], [1/3,1/3,1/3])

lrs=[0.0003]

if __name__ == "__main__":
    for lr in lrs:

        for _ in range(1):
            run(lr=lr)
