
import numpy as np
from stable_baselines3 import SAC, PPO
import time
from src.environments import ConPricingGame
import src.globals as gl
import src.classes as cl
import os


def initial_matrix(env_class, random=False):
    if not random:

        strt1 = cl.Strategy(
            cl.StrategyType.static, model_or_func=cl.myopic, name="myopic")
        strt2 = cl.Strategy(
            cl.StrategyType.static, model_or_func=cl.const, name="const", first_price=132)
        strt3 = cl.Strategy(
            cl.StrategyType.static, model_or_func=cl.guess, name="guess", first_price=132)
        # strt4 = cl.Strategy(
        #     cl.StrategyType.static, model_or_func=cl.spe, name="spe")
        init_low = [strt1, strt2, strt3]
        init_high = [strt1, strt2, strt3]
    else:
        model_name = f"rndstart_{job_name}"
        log_dir = f"{gl.LOG_DIR}/{model_name}"
        model_dir = f"{gl.MODELS_DIR}/{model_name}"
        if not os.path.exists(f"{model_dir}.zip"):
            train_env = env_class(tuple_costs=None, adversary_mixed_strategy=None, memory=12)
            model = SAC('MlpPolicy', train_env,
                        verbose=0, tensorboard_log=log_dir, gamma=gl.GAMMA, target_entropy=0)
            model.save(model_dir)

        strt_rnd = cl.Strategy(strategy_type=cl.StrategyType.sb3_model,
                               model_or_func=SAC, name=model_name, action_step=None, memory=12)

        init_low = [strt_rnd]
        init_high = [strt_rnd]

    low_strts, high_strts = db.get_list_of_added_strategies()
    return cl.BimatrixGame(
        low_cost_strategies=init_low+low_strts, high_cost_strategies=init_high+high_strts, env_class=env_class)


if __name__ == "__main__":

    gl.initialize()

    env_class = ConPricingGame

    num_rounds = 3
    num_procs = 1
    start_random = True
    job_name = "test"

    db_name = job_name+".db"
    db = cl.DataBase(db_name)
    cl.set_job_name(job_name)
    cl.create_directories()
    equilibria = []

    # params
    lrs = [0.0003, 0.00016]
    memories = [12, 18]
    algs = [SAC]

    start_game = initial_matrix(env_class=env_class, random=start_random)

    bimatrix_game = cl.load_latest_game(game_data_name=f"game_{job_name}", new_game=start_game)

    cl.prt("\n" + time.ctime(time.time())+"\n"+("-"*50)+"\n")

    all_equilibria = bimatrix_game.compute_equilibria()
    equilibria = all_equilibria[:min(len(all_equilibria), gl.NUM_TRACE_EQUILIBRIA)]
    game_size = bimatrix_game.size()

    # low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff = bimatrix_game.compute_equilibria()
    for round in range(num_rounds):
        cl.prt(f"Round {round} of {num_rounds}")

        added_low = 0
        added_high = 0
        # for equilibrium in dictionaries:
        for equi in equilibria:
            new_equi_low = 0
            new_equi_high = 0

            # low_prob_str = ", ".join(
            #     map("{0:.2f}".format, equi["low_cost_probs"]))
            # high_prob_str = ", ".join(
            #     map("{0:.2f}".format, equi["high_cost_probs"]))
            cl.prt(
                f'equi: {str(equi.row_support)}, {str(equi.col_support)}\n payoffs= {equi.row_payoff:.2f}, {equi.col_payoff:.2f}')

            # train a low-cost agent
            high_mixed_strat = cl.MixedStrategy(
                strategies_lst=bimatrix_game.high_strategies, probablities_lst=((equi.col_probs+([0]*added_high)) if
                                                                                added_high > 0 else equi.col_probs))

            for alg in algs:
                for lr in lrs:
                    for mem_i, memory in enumerate(memories):

                        print(f'training low-cost agents with alg={str(alg)}, lr={lr:.4f}, memory={memory}')

                        results = cl.train_processes(db=db, env_class=env_class, costs=[gl.LOW_COST, gl.HIGH_COST],
                                                     adv_mixed_strategy=high_mixed_strat, target_payoff=equi.row_payoff,
                                                     num_procs=num_procs, alg=alg, lr=lr, memory=memory)
                        for result in results:
                            acceptable, agent_payoffs, adv_payoffs, agent_strategy, expected_payoff, base_agent_name = result
                            if acceptable:
                                new_equi_low += 1
                                added_low += 1
                                bimatrix_game.low_strategies.append(agent_strategy)
                                bimatrix_game.add_low_cost_row(agent_payoffs, adv_payoffs)
                                cl.prt(
                                    f'low-cost player {agent_strategy.name} , payoff= {expected_payoff:.2f} added, base={base_agent_name} ,alg={str(alg)}, lr={lr:.4f}, memory={memory}')

            # train a high-cost agent
            low_mixed_strat = cl.MixedStrategy(
                strategies_lst=bimatrix_game.low_strategies, probablities_lst=((equi.row_probs+([0]*added_low)) if added_low > 0 else equi.row_probs))

            for alg in algs:
                for lr in lrs:
                    for memory in memories:
                        print(f'training high-cost player with alg={str(alg)}, lr={lr:.4f}, memory={memory}')

                        results = cl.train_processes(db=db, env_class=env_class, costs=[gl.HIGH_COST, gl.LOW_COST],
                                                     adv_mixed_strategy=low_mixed_strat, target_payoff=equi.col_payoff,
                                                     num_procs=num_procs, alg=alg, lr=lr, memory=memory)
                        for result in results:
                            acceptable, agent_payoffs, adv_payoffs, agent_strategy, expected_payoff, base_agent_name = result
                            if acceptable:
                                new_equi_high += 1
                                added_high += 1
                                bimatrix_game.high_strategies.append(agent_strategy)
                                bimatrix_game.add_high_cost_col(adv_payoffs, agent_payoffs)

                                cl.prt(
                                    f'high-cost player {agent_strategy.name} , payoff= {expected_payoff:.2f} added, base={base_agent_name}, alg={str(alg)}, lr={lr:.4f}, memory={memory}')

            # because high_mixed_strt is defined before the changes to bimatrix.high_strategies. (error in str(high_mixed))
            if new_equi_high > 0:
                high_mixed_strat.strategy_probs += [0]*new_equi_high

            # if new_equi_low>0 or new_equi_high>0:
                # equilibria.append(
                #     [equi["low_cost_probs"], equi["high_cost_probs"], equi["low_cost_payoff"], equi["high_cost_payoff"]])
                # to do: add the equilibria to the db
            db.insert_new_equi(game_size=game_size, low_strategy_txt=str(low_mixed_strat), high_strategy_txt=str(
                high_mixed_strat), low_payoff=equi.row_payoff, high_payoff=equi.col_payoff, low_new_num=new_equi_low, high_new_num=new_equi_high)

        if added_low == 0 and added_high == 0:
            gl.N_EPISODES_BASE *= 1.1
            gl.N_EPISODES_LOAD *= 1.1
        else:
            all_equilibria = bimatrix_game.compute_equilibria()
            equilibria = all_equilibria[:min(len(all_equilibria), gl.NUM_TRACE_EQUILIBRIA)]
            game_size = bimatrix_game.size()

    all_equilibria = bimatrix_game.compute_equilibria()
    equilibria = all_equilibria[:min(len(all_equilibria), gl.NUM_TRACE_EQUILIBRIA)]
