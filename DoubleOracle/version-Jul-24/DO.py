
import numpy as np
from stable_baselines3 import SAC, PPO
import time
from src.environments import ConPricingGame
import src.globals as gl
import src.classes as cl
import os
import multiprocessing as mp
from typing import List, Dict
from enum import Enum
import json
import logging
# Configure logging
logging.basicConfig(filename='error.log', level=logging.ERROR)

alg_params = {
    'SAC': {
        'learning_rate': 0.0003,
        'target_entropy': 'auto',
        'ent_coef': 'auto',
        'tau': 0.010,
        'train_freq': 1,
        'gradient_steps': 1,
        'verbose': 0,
        'buffer_size': 200_000
    },
    'PPO': {
        'learning_rate': 0.00016,
        'n_epochs': 10,
        'clip_range': 0.3,
        'clip_range_vf': None,
        'ent_coef': 0.010,
        'vf_coef': 0.5,
        'verbose': 0
    }
}



class ProcessInd(Enum):
    SAClow = 0
    PPOlow = 1
    SAChigh = 2
    PPOhigh = 3


class StartMode(Enum):
    """ double oracle game strting point, start from myopic-const-guess or from a random model or different strategies similar to guess"""
    myopicConstGuess = 0
    random = 1
    multiGuess = 2
    allVsSpe=3
    
def len_initial_game(start_mode:StartMode)->int:
    if start_mode==StartMode.myopicConstGuess or start_mode==StartMode.multiGuess:
        return 3
    elif start_mode==StartMode.random:
        return 1
    elif start_mode==StartMode.allVsSpe:
        return 7
    else:
        raise ValueError("len of start_mode not implemented!")
        
    
def remove_ineffective_agents(bimatrix_game:cl.BimatrixGame, db:cl.DataBase, start_mode:StartMode):
    """ according to average_probs table, removes that have never been part of an equilibrium, and they have been in the game for a while, also updates the added column of agents to -1, and the pickle file and game should be saved again"""
    query=f"select * from {db.PROBS_TABLE} order by id desc limit 1;"
    df= db.dataframe_select(query=query)
    if df.empty:
        return bimatrix_game
    row = df.iloc[0]
    
    
    probs = json.loads(row['strategy_probs'])
    start_ind=len_initial_game(start_mode=start_mode)
    #########to be done


def initial_matrix(env_class, start_mode):
    """ returns double oracle game with strategies from last stopping point but the matrix and strategies are not loaded , creates the base matrix and adds the trained strategies"""
    if start_mode == StartMode.myopicConstGuess:

        strt1 = cl.Strategy(
            cl.StrategyType.static, model_or_func=cl.myopic, name="myopic")
        strt2 = cl.Strategy(
            cl.StrategyType.static, model_or_func=cl.const, name="const", first_price=132)
        strt3 = cl.Strategy(
            cl.StrategyType.static, model_or_func=cl.guess, name="guess", first_price=132)

        init_low = [strt1, strt2, strt3]
        init_high = [strt1, strt2, strt3]
    elif start_mode == StartMode.allVsSpe:
        strt0 = cl.Strategy(
            cl.StrategyType.static, model_or_func=cl.spe, name="spe", first_price=132)
        strt1 = cl.Strategy(
            cl.StrategyType.static, model_or_func=cl.myopic, name="myopic")
        strt2 = cl.Strategy(
            cl.StrategyType.static, model_or_func=cl.const, name="const", first_price=132)
        strt3 = cl.Strategy(
            cl.StrategyType.static, model_or_func=cl.imit, name="imit", first_price=132)
        
        strt4 = cl.Strategy(
            cl.StrategyType.static, model_or_func=cl.guess, name="normal_guess",first_price=132)
        strt5 = cl.Strategy(
            cl.StrategyType.static, model_or_func=cl.guess2, name="coop_guess", first_price=132)
        strt6 = cl.Strategy(
            cl.StrategyType.static, model_or_func=cl.guess3, name="compete_guess", first_price=132)

        init_low = [strt0,strt1, strt2, strt3,strt4,strt5,strt6]
        init_high = [strt0,strt1, strt2, strt3,strt4,strt5,strt6]
    elif start_mode == StartMode.random:
        model_name = f"rndstart_{job_name}"
        log_dir = f"{gl.LOG_DIR}/{model_name}"
        model_dir = f"{gl.MODELS_DIR}/{model_name}"
        if not os.path.exists(f"{model_dir}.zip"):
            # tuple_costs and others are none just to make sure no play is happening here
            train_env = env_class(tuple_costs=None, adversary_mixed_strategy=None, memory=gl.MEMORY)
            model = SAC('MlpPolicy', train_env,
                        verbose=0, tensorboard_log=log_dir, gamma=gl.GAMMA, target_entropy=0)
            model.save(model_dir)

        strt_rnd = cl.Strategy(strategy_type=cl.StrategyType.sb3_model,
                               model_or_func=SAC, name=model_name, action_step=None, memory=gl.MEMORY)

        init_low = [strt_rnd]
        init_high = [strt_rnd]
    elif start_mode==StartMode.multiGuess:
        strt1 = cl.Strategy(
            cl.StrategyType.static, model_or_func=cl.guess, name="normal_guess",first_price=132)
        strt2 = cl.Strategy(
            cl.StrategyType.static, model_or_func=cl.guess2, name="coop_guess", first_price=132)
        strt3 = cl.Strategy(
            cl.StrategyType.static, model_or_func=cl.guess3, name="compete_guess", first_price=132)

        init_low = [strt1, strt2, strt3]
        init_high = [strt1, strt2, strt3]
    else:
        raise("Error: initial_matrix mode not implemented!")
    

    low_strts, high_strts = db.get_list_of_added_strategies(action_step=None, memory=gl.MEMORY)
    return cl.BimatrixGame(
        low_cost_strategies=init_low+low_strts, high_cost_strategies=init_high+high_strts, env_class=env_class)


def get_proc_input(seed, proc_ind: ProcessInd, low_mixed_strt, high_mixed_strt, payoffs_low_high, job_name, env_class,num_ep_coef,equi_id,db) -> cl.TrainInputRow:
    """
    creates the input tuple for new_train method, to use in multiprocessing
    """
    # input=(id, seed, job_name, env, base_agent, alg, alg_params, adv_mixed_strategy,target_payoff, db)
    if proc_ind == ProcessInd.PPOlow or proc_ind == ProcessInd.SAClow:
        costs = [gl.LOW_COST, gl.HIGH_COST]
        own_strt = low_mixed_strt.copy_unload()
        adv_strt = high_mixed_strt.copy_unload()
        payoff = payoffs_low_high[0]
    elif proc_ind == ProcessInd.PPOhigh or proc_ind == ProcessInd.SAChigh:
        costs = [gl.HIGH_COST, gl.LOW_COST]
        own_strt = high_mixed_strt.copy_unload()
        adv_strt = low_mixed_strt.copy_unload()
        payoff = payoffs_low_high[1]

    if proc_ind == ProcessInd.SAChigh or proc_ind == ProcessInd.SAClow:
        alg = SAC
    elif proc_ind == ProcessInd.PPOhigh or proc_ind == ProcessInd.PPOlow:
        alg = PPO

    iid = proc_ind.value
    env = env_class(tuple_costs=costs, adversary_mixed_strategy=adv_strt, memory=gl.MEMORY)
    base_agent = cl.find_base_agent(db, alg, costs[0], own_strt)
    return cl.TrainInputRow(iid, seed+iid, job_name, env, base_agent, alg, alg_params[cl.name_of(alg)], adv_strt, payoff, db,num_ep_coef,equi_id)


if __name__ == "__main__":
    try:
        gl.initialize()

        env_class = ConPricingGame

        num_rounds = 10
        num_procs = 1
        # works best with num_process=1 or >=4
        start_mode = StartMode.random
        job_name = "test"

        db_name = job_name+".db"
        db = cl.DataBase(db_name)
        cl.set_job_name(job_name)
        cl.create_directories()
        equis_id_dict = []

        start_game = initial_matrix(env_class=env_class, start_mode=start_mode)

        bimatrix_game = cl.load_latest_game(game_data_name=f"game_{job_name}", new_game=start_game)

        cl.prt("\n" + time.ctime(time.time())+"\n"+("-"*50)+"\n")

        equis_id_dict = cl.get_coop_equilibria(bimatrix_game=bimatrix_game, num_trace=100, db=db)
        game_size = bimatrix_game.size()
        
        #coeficient of how much num_episodes should be increased
        num_ep_coef=1
        
        # db.delete_extra_rows(db.ITERS_TABLE, 'agent_id', gl.DB_ITER_LIMIT)

        # low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff = bimatrix_game.compute_equilibria()
        for round in range(num_rounds):
            cl.prt(f"\n\tRound {round+1} of {num_rounds}")

            added_low = 0
            added_high = 0
            # for equilibrium in dictionaries:
            for equi in equis_id_dict:

                db.updates_equi_average_probs(equis_id_dict[equi], equi)
                new_equi_low = 0
                new_equi_high = 0
                print(
                    f'round {round+1} equi: {str(equi.row_support)}, {str(equi.col_support)}\n payoffs= {equi.row_payoff:.2f}, {equi.col_payoff:.2f}')
                cl.prt(
                    f'equi: {str(equi.row_support)}, {str(equi.col_support)}\n payoffs= {equi.row_payoff:.2f}, {equi.col_payoff:.2f}')

                # train a low-cost agent
                high_mixed_strat = cl.MixedStrategy(
                    strategies_lst=bimatrix_game.high_strategies, probablities_lst=((equi.col_probs+([0]*added_high)) if
                                                                                    added_high > 0 else equi.col_probs))
                low_mixed_strat = cl.MixedStrategy(
                    strategies_lst=bimatrix_game.low_strategies, probablities_lst=((equi.row_probs+([0]*added_low)) if added_low > 0 else equi.row_probs))

                # prepare processes
                proc_inputs = []
                input_id_dict: Dict[int, cl.TrainInputRow] = {}
                seed = int(time.time())
                for proc_ind in ProcessInd:
                    inp = get_proc_input(seed, proc_ind, low_mixed_strat, high_mixed_strat, [
                                        equi.row_payoff, equi.col_payoff], job_name, env_class,num_ep_coef,equis_id_dict[equi],db)
                    proc_inputs.append(inp)
                    input_id_dict[inp.id] = inp

                if num_procs > 1:
                    if (extra_prcs := num_procs-len(ProcessInd)) > 0:
                        for ext_ind in range(extra_prcs):
                            # choose a random processInd to be done with the extra core
                            new_proc_ind = np.random.choice(list(ProcessInd))
                            new_seed = seed+1 + (ext_ind+1) * len(ProcessInd)
                            inp = get_proc_input(new_seed, new_proc_ind, low_mixed_strat, high_mixed_strat, [
                                                equi.row_payoff, equi.col_payoff], job_name, env_class,num_ep_coef,equis_id_dict[equi],db)
                            proc_inputs.append(inp)
                            input_id_dict[inp.id] = inp
                    pool = mp.Pool(processes=num_procs)
                    outputs = pool.imap_unordered(cl.new_train, proc_inputs)
                    pool.close()
                    pool.join()
                else:
                    outputs = []
                    for inp in proc_inputs:
                        outputs.append(cl.new_train(inp))
                        input_id_dict[inp.id] = inp

                for output in outputs:
                    id, acceptable, strategy_name, agent_payoffs, adv_payoffs, expected_payoff = output
                    inp = input_id_dict[id]

                # id,acceptable,strategy_name,agent_payoffs, adv_payoffs, expected_payoff= train(inputs[0])
                    # pricing_game = env_class(tuple_costs=costs, adversary_mixed_strategy=adv_strt, memory=memory)
                    pricing_game = inp.env
                    model_strategy = cl.Strategy(strategy_type=cl.StrategyType.sb3_model,
                                                model_or_func=inp.alg, name=strategy_name, action_step=pricing_game.action_step, memory=pricing_game.memory)
                # compute the payoff against all adv strategies, to be added to the matrix
                    if acceptable:
                        updated_adv_strategy, agent_payoffs, adv_payoffs = cl.match_updated_size(
                            bimatrix_game, inp.adv_mixed_strategy, pricing_game.costs[0], agent_payoffs, adv_payoffs)
                        for strategy_index in range(len(updated_adv_strategy.strategies)):
                            if updated_adv_strategy.strategy_probs[strategy_index] == 0:
                                payoffs = []
                                for _ in range(gl.NUM_STOCHASTIC_ITER):
                                    payoffs.append(model_strategy.play_against(
                                        env=pricing_game, adversary=updated_adv_strategy.strategies[strategy_index]))
                                mean_payoffs = np.array(payoffs).mean(axis=0)

                                agent_payoffs[strategy_index] = mean_payoffs[0]
                                adv_payoffs[strategy_index] = mean_payoffs[1]
                        # results.append((acceptable, agent_payoffs, adv_payoffs, model_strategy, expected_payoff, inp.base_agent))
                        if pricing_game.costs[0] == gl.LOW_COST:
                            new_equi_low += 1
                            added_low += 1
                            bimatrix_game.low_strategies.append(model_strategy)
                            bimatrix_game.add_low_cost_row(agent_payoffs, adv_payoffs)
                            cl.prt(
                                f'low-cost player {model_strategy.name} , payoff= {expected_payoff:.2f} added, base={inp.base_agent} ,alg={cl.name_of(inp.alg)}')
                        elif pricing_game.costs[0] == gl.HIGH_COST:
                            new_equi_high += 1
                            added_high += 1
                            bimatrix_game.high_strategies.append(model_strategy)
                            bimatrix_game.add_high_cost_col(adv_payoffs, agent_payoffs)

                            cl.prt(
                                f'high-cost player {model_strategy.name} , payoff= {expected_payoff:.2f} added, base={inp.base_agent} ,alg={cl.name_of(inp.alg)}')

                db.update_equi(id=equis_id_dict[equi], used=(new_equi_low > 0 or new_equi_high > 0),
                            num_new_low=new_equi_low, num_new_high=new_equi_high)

                # because high_mixed_strt is defined before the changes to bimatrix.high_strategies. (error in str(high_mixed))
                if new_equi_high > 0:
                    high_mixed_strat.strategy_probs += [0]*new_equi_high

                # if new_equi_low>0 or new_equi_high>0:
                    # equilibria.append(
                    #     [equi["low_cost_probs"], equi["high_cost_probs"], equi["low_cost_payoff"], equi["high_cost_payoff"]])
                    # to do: add the equilibria to the db
                # db.insert_new_equi(game_size=game_size, low_strategy_txt=str(low_mixed_strat), high_strategy_txt=str(
                #     high_mixed_strat), low_payoff=equi.row_payoff, high_payoff=equi.col_payoff, low_new_num=new_equi_low, high_new_num=new_equi_high)

            if added_low == 0 and added_high == 0:
                num_ep_coef *= gl.EPISODE_INCREASE_COEF
            else:
                equis_id_dict = cl.get_coop_equilibria(bimatrix_game=bimatrix_game, num_trace=100, db=db)
                game_size = bimatrix_game.size()
                num_ep_coef=1

        all_equilibria = bimatrix_game.compute_equilibria()
        equis_id_dict = all_equilibria[:min(len(all_equilibria), gl.NUM_TRACE_EQUILIBRIA)]
    except Exception as e:
        # Log the error
        logging.error("An error occurred: %s", str(e))
        # You can also print the error if needed
        print("An error occurred:", e)