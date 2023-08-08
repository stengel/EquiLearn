from stable_baselines3 import PPO
import model
import globals as gl
import classes as cl

gl.initialize()

lrs=[0.00008, 0.0003, 0.0009]

costs=[gl.LOW_COST, gl.HIGH_COST]
adv_mixed_strategy = cl.MixedStrategy(strategiesList=[cl.Strategy(
        cl.StrategyType.static, NNorFunc=cl.myopic, name="myopic")], probablitiesArray=[1])
for lr in lrs:
    for _ in range(2):
        model.train_agent(costs=costs, adv_mixed_strategy=adv_mixed_strategy, model= PPO, model_name="PPO", timesteps= 500_000, num_timesteps=20, lr=lr)

