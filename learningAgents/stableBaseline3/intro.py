import gym
from stable_baselines3 import A2C

env=gym.make("CarRacing-v2", render_mode="human")
env.reset()

model= A2C("MlpPolicy",env,verbose=1)
model.learn(total_timesteps=10_000)

# episodes=10

# for ep in range(episodes):
#     obs=env.reset()
#     done=False
#     while not done:
#         env.render()
#         obs, reward, done,trunc, info= env.step(env.action_space.sample())
env.close()