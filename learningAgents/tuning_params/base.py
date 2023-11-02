
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import time
import sys
from src.environments import ConPricingGame
import src.globals as gl
import src.classes as cl

if __name__ == "__main__": 

    global job_name
    job_name = "lr"
    gl.initialize()
    cl.set_job_name(job_name)

    alg = SAC
    env_class = ConPricingGame

    lrs = np.linspace(0.00001, 0.0005, 10)
    num_procs = gl.NUM_PROCESS if (len(sys.argv) < 2) else int(sys.argv[1])

    costs=[gl.LOW_COST,gl.HIGH_COST]

    adv_mixed_strategy = cl.MixedStrategy(strategies_lst=[cl.Strategy(cl.StrategyType.static, cl.myopic, name="myopic"), cl.Strategy(
        cl.StrategyType.static, cl.myopic, name="guess", first_price=132)], probablities_lst=[0.5, 0.5])

    for lr in lrs:
        for iter in range(2):
            cl.prt(f"\nlr = {lr:.5f}, iter= {iter+1} strated. {time.ctime(time.time())} --------------------\n")
            alg_name = f"{job_name}_lr"
            pricing_game = env_class(
                tuple_costs=costs, adversary_mixed_strategy=adv_mixed_strategy)

            seed = int(time.time())

            model_name = f"{alg_name}-{str(seed)}"
            models_dir = f"{gl.MODELS_DIR}/{model_name}"
            log_dir = f"{gl.LOG_DIR}/{model_name}"

            number_episodes = 4_000_000
            train_env = make_vec_env(env_class, n_envs=num_procs, seed=seed, vec_env_cls=SubprocVecEnv, env_kwargs=dict(
                tuple_costs=costs, adversary_mixed_strategy=adv_mixed_strategy))
            model = alg('MlpPolicy', train_env, learning_rate=gl.LR,
                        verbose=0, tensorboard_log=log_dir, gamma=gl.GAMMA, target_entropy=0)

            start = time.time()
            # for i in range(gl.NUM_MODEL_SAVE):
            # tmp = (number_episodes/gl.NUM_MODEL_SAVE)
            # model.learn(total_timesteps=tmp, reset_num_timesteps=False,
            #             tb_log_name=model_name)
            # model.save(os.path.join(models_dir, str(tmp*(i+1))))
            model.learn(total_timesteps=number_episodes, tb_log_name=model_name)
            model.save(models_dir)
            cl.prt(f"model {models_dir} is saved!\n")

            running_time = time.time() - start

        

            model_strategy = cl.Strategy(strategy_type=cl.StrategyType.sb3_model,
                                        model_or_func=alg, name=model_name, action_step=pricing_game.action_step)
            mean_payoffs=[0]*len(adv_mixed_strategy.strategies)
            total_mean=0
            for strategy_index in range(len(adv_mixed_strategy.strategies)):
                if adv_mixed_strategy.strategy_probs[strategy_index] > 0:
                    payoffs = []
                    for i in range(gl.NUM_STOCHASTIC_ITER):
                        # returns = algorithm.play_trained_agent(adversary=(
                        #     (adv_mixed_strategy._strategies[strategy_index]).to_mixed_strategy()), iterNum=gl.num_stochastic_iter)
                        payoffs.append(model_strategy.play_against(
                            env=pricing_game, adversary=adv_mixed_strategy.strategies[strategy_index]))
                        data = [model_name, number_episodes, ("L" if (costs[0] < costs[1]) else "H"), pricing_game.adversary_strategy.name,
                                sum(pricing_game.profit[0]), sum(pricing_game.profit[1]),  str(
                                    pricing_game.profit[0]), str(pricing_game.actions), str(pricing_game.prices[0]),
                                str(pricing_game.prices[1]), str(pricing_game.demand_potential[0]), str(
                                    pricing_game.demand_potential[1]), lr, gl.NUM_ADV_HISTORY,
                                gl.TOTAL_STAGES, pricing_game.action_step, gl.NUM_ACTIONS, gl.GAMMA, False, seed, num_procs, running_time]
                        cl.write_results(data)

                    mean_payoffs[strategy_index] = (np.array(payoffs).mean(axis=0))[0]
                    total_mean+= mean_payoffs[strategy_index]*adv_mixed_strategy.strategy_probs[strategy_index]

                    

            data = [model_name, number_episodes, ("L" if (costs[0] < costs[1]) else "H"), str(adv_mixed_strategy)+str(mean_payoffs), total_mean, 0,
                    lr, gl.NUM_ADV_HISTORY, gl.TOTAL_STAGES, pricing_game.action_step, gl.NUM_ACTIONS, gl.GAMMA, seed, num_procs, running_time,
                    time.ctime(time.time())]

            cl.write_agents(data)
            cl.prt(f"results of model {model_name} are written!\n")
    # alg.write_nn_data(("low" if costs[0] < costs[1] else "high"))



# if __name__ == "__main__":
#     gl.initialize()


#     equilibria = []

#     cl.create_directories()

#     strt1 = cl.Strategy(
#         cl.StrategyType.static, model_or_func=cl.myopic, name="myopic")
#     strt2 = cl.Strategy(
#         cl.StrategyType.static, model_or_func=cl.const, name="const", first_price=132)
#     strt3 = cl.Strategy(
#         cl.StrategyType.static, model_or_func=cl.guess, name="guess", first_price=132)

#     cl.prt("\n" + time.ctime(time.time())+"\n"+("-"*50)+"\n")

#     bimatrix_game = cl.BimatrixGame(
#         low_cost_strategies=[strt1, strt2, strt3], high_cost_strategies=[strt1, strt2, strt3], env_class=env_class)

#     bimatrix_game.reset_matrix()
#     bimatrix_game.fill_matrix()

#     num_procs = gl.NUM_PROCESS if (len(sys.argv) < 2) else int(sys.argv[1])

#     dictionaries = bimatrix_game.compute_equilibria()

#     # low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff = bimatrix_game.compute_equilibria()
#     for round in range(num_rounds):
#         cl.prt(f"Round {round} of {num_rounds}")
#         new_low = False
#         new_high = False
#         # inputs = []
#         # update = [False] * len(dictionaries)
#         # for equilibrium in dictionaries:
#         equi = dictionaries[0]
#         low_prob_str = ", ".join(
#             map("{0:.2f}".format, equi["low_cost_probs"]))
#         high_prob_str = ", ".join(
#             map("{0:.2f}".format, equi["high_cost_probs"]))
#         cl.prt(
#             f'equi: [{low_prob_str}], [{high_prob_str}], {equi["low_cost_payoff"]:.2f}, {equi["high_cost_payoff"]:.2f}')

#         # train a low-cost agent
#         high_mixed_strat = cl.MixedStrategy(
#             strategies_lst=bimatrix_game.high_strategies, probablities_lst=equi["high_cost_probs"])

#         [acceptable, agent_payoffs, adv_payoffs, agent_strategy, expected_payoff] = training(env_class=env_class, costs=[
#                                                                                              gl.LOW_COST, gl.HIGH_COST], adv_mixed_strategy=high_mixed_strat, target_payoff=equi["low_cost_payoff"], num_procs=num_procs)
#         if acceptable:
#             new_low = True
#             # update[int(i/2)] = True
#             bimatrix_game.low_strategies.append(agent_strategy)
#             bimatrix_game.add_low_cost_row(agent_payoffs, adv_payoffs)

#             # cl.prt(f"low cost player {agent_strategy.name} added, trained with ", [
#             #     equi["low_cost_probabilities"], equi["high_cost_probabilities"], equi["low_cost_payoff"], equi["high_cost_payoff"]])
#             cl.prt(
#                 f'low-cost player {agent_strategy.name} , payoff= {expected_payoff:.2f} added, trained against {equi["high_cost_probs"]}, payoff={equi["low_cost_payoff"]:.2f}')

#         # train a high-cost agent
#         low_mixed_strat = cl.MixedStrategy(
#             strategies_lst=bimatrix_game.low_strategies, probablities_lst=((equi["low_cost_probs"]+[0]) if new_low else equi["low_cost_probs"]))

#         [acceptable, agent_payoffs, adv_payoffs, agent_strategy, expected_payoff] = training(env_class=env_class, costs=[
#             gl.HIGH_COST, gl.LOW_COST], adv_mixed_strategy=low_mixed_strat, target_payoff=equi["high_cost_payoff"], num_procs=num_procs)
#         if acceptable:
#             new_high = True

#             bimatrix_game.high_strategies.append(agent_strategy)
#             bimatrix_game.add_high_cost_col(adv_payoffs, agent_payoffs)

#             cl.prt(
#                 f'high-cost player {agent_strategy.name} , payoff= {expected_payoff:.2f} added, trained against {equi["low_cost_probs"]}, payoff={equi["high_cost_payoff"]:.2f}')

#         if new_low or new_high:
#             equilibria.append(
#                 [equi["low_cost_probs"], equi["high_cost_probs"], equi["low_cost_payoff"], equi["high_cost_payoff"]])
#             dictionaries = bimatrix_game.compute_equilibria()
#             gl.NUM_EPISODES = gl.NUM_EPISODES_RESET
#         else:
#             gl.NUM_EPISODES += gl.EPISODE_ADV_INCREASE
