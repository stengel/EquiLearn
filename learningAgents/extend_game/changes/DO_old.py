
import numpy as np
from stable_baselines3 import SAC, PPO
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import time
import sys
from src.environments import ConPricingGame
import src.globals as gl
import src.classes as cl
import multiprocessing as mp

def train(inputs):
    """ traines one agent against the adversary, if the expected payoff of new agent is greater than expected payoff of NE, returns acceptable=true and the new strategy and payoff to be added to the strategies and matrix."""
    id,seed, job_name,env, base_agent, alg, adv_mixed_strategy,lr,target_payoff,db=inputs
    
    gl.initialize()
    model_name = f"{job_name}-{str(seed)}"
    models_dir = f"{gl.MODELS_DIR}/{model_name}"
    log_dir = f"{gl.LOG_DIR}/{model_name}"
    
    acceptable = False
    if base_agent is None:
        number_episodes = gl.N_EPISODES_BASE * (1 + gl.EPISODE_INCREASE_PORTION * (adv_mixed_strategy.support_size-1))
        if alg is SAC:
            model = alg('MlpPolicy', env, learning_rate=lr,
                        verbose=0, tensorboard_log=log_dir, gamma=gl.GAMMA, target_entropy=0, seed=seed)
        else:
            model = alg('MlpPolicy', env, learning_rate=lr,
                        verbose=0, tensorboard_log=log_dir, gamma=gl.GAMMA,seed=seed)
    else:
        number_episodes = gl.N_EPISODES_LOAD * (1 + gl.EPISODE_INCREASE_PORTION * (adv_mixed_strategy.support_size-1))
        base_agent_dir = f"{gl.MODELS_DIR}/{base_agent}"
        if alg is SAC:
            model = alg.load(base_agent_dir, env, learning_rate=lr,
                             verbose=0, tensorboard_log=log_dir, gamma=gl.GAMMA, target_entropy=0)
        else:
            model = alg.load(base_agent_dir, env, learning_rate=lr,
                             verbose=0, tensorboard_log=log_dir, gamma=gl.GAMMA)
    start = time.time()
    # for i in range(gl.NUM_MODEL_SAVE):
    # tmp = (number_episodes/gl.NUM_MODEL_SAVE)
    # model.learn(total_timesteps=tmp, reset_num_timesteps=False,
    #             tb_log_name=model_name)
    # model.save(os.path.join(models_dir, str(tmp*(i+1))))
    
    # https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#id3
    #check to save and load replay buffer as well
    model.learn(total_timesteps=number_episodes, tb_log_name=model_name)
    model.save(models_dir)
    running_time = time.time() - start
    
    model_strategy = cl.Strategy(strategy_type=cl.StrategyType.sb3_model,
                                 model_or_func=alg, name=model_name, action_step=env.action_step,memory=env.memory)
    
    iter_rows = []
    agent_payoffs = np.zeros(len(adv_mixed_strategy.strategies))
    adv_payoffs = np.zeros(len(adv_mixed_strategy.strategies))
    expected_payoff = 0
    for strategy_index in range(len(adv_mixed_strategy.strategies)):
        if adv_mixed_strategy.strategy_probs[strategy_index] > 0:
            payoffs = []
            for _ in range(gl.NUM_STOCHASTIC_ITER):
                # returns = algorithm.play_trained_agent(adversary=(
                #     (adv_mixed_strategy._strategies[strategy_index]).to_mixed_strategy()), iterNum=gl.num_stochastic_iter)
                payoffs.append(model_strategy.play_against(
                    env=env, adversary=adv_mixed_strategy.strategies[strategy_index]))
                
                #adv, agent_return, adv_return, rewards, adv_rewards, actions, prices, adv_prices, demands, adv_demands
                iter_row = cl.Iter_row(adv=env.adversary_strategy.name, agent_return=sum(env.profit[0]), adv_return=sum(env.profit[1]), rewards=str(
                    env.profit[0]), adv_rewards=str(env.profit[1]), actions=str(env.actions),prices=str(env.prices[0]), adv_prices=str(env.prices[1]) ,demands=str(env.demand_potential[0]), adv_demands=str(env.demand_potential[1]))

                iter_rows.append(iter_row)

            mean_payoffs = np.array(payoffs).mean(axis=0)

            agent_payoffs[strategy_index] = mean_payoffs[0]
            adv_payoffs[strategy_index] = mean_payoffs[1]
            expected_payoff += (agent_payoffs[strategy_index]) * \
                (adv_mixed_strategy.strategy_probs[strategy_index])

    acceptable=(expected_payoff > target_payoff)
    # agent_id=db.insert_new_agent(model_name,number_episodes,costs[0], str(adv_mixed_strategy), expected_payoff,target_payoff, lr,memory, acceptable, pricing_game.action_step, seed,num_procs,running_time)
    agent_id = db.insert_new_agent(db.AgentRow(model_name, base_agent, number_episodes, env.costs[0], str(
        adv_mixed_strategy), expected_payoff, target_payoff,  str(alg),lr, env.memory, acceptable, env.action_step, seed, 1, running_time))
    #num_processes=1 because it just uses one process in training this agent

    if acceptable:
        for row in iter_rows:
            db.insert_new_iteration(agent_id, row.adv, row.agent_return, row.adv_return, row.rewards,
                                    row.adv_rewards, row.actions, row.prices, row.adv_prices, row.demands, row.adv_demands)
    
    return (id,acceptable,model_strategy.name,agent_payoffs, adv_payoffs, expected_payoff)

def train_processes(db, env_class, costs, adv_mixed_strategy, target_payoff, num_procs, alg, lr, memory):
    """
    trains multiple agents with multiprocessing against mixed_adversary. 
    """
    inputs=[]
    seed = int(time.time())
    adv_strt= adv_mixed_strategy.copy_unload()
    
    base_agents= cl.find_base_agents(db=db,alg=alg,memory=memory,cost=costs[0],mix_strt=adv_strt,size=num_procs)
    
    for p in range(num_procs):
        env = env_class(tuple_costs=costs, adversary_mixed_strategy=adv_strt, memory=memory)
        input_proc=(p,seed+p, job_name,env, base_agents[p],alg, adv_strt,lr,target_payoff,db)
        inputs.append(input_proc)
    results=[]
    # with cf.ProcessPoolExecutor() as executor:
    # # Submit all the tasks to the executor and get the future objects
    #     futures = [executor.submit(train, input_proc) for input_proc in inputs]
    #     for future in cf.as_completed(futures):
    #         res=future.result()
    pool = mp.Pool(processes=num_procs)
    
    outputs=pool.imap_unordered(train,inputs)
    for output in outputs:
        id,acceptable,strategy_name,agent_payoffs, adv_payoffs, expected_payoff=output
        # id,acceptable,model_strategy,agent_payoffs, adv_payoffs, expected_payoff= train(inputs[0])
        pricing_game = env_class(tuple_costs=costs, adversary_mixed_strategy=adv_strt, memory=memory)
        model_strategy = cl.Strategy(strategy_type=cl.StrategyType.sb3_model,
                                 model_or_func=alg, name=strategy_name, action_step=pricing_game.action_step,memory=memory)
    # compute the payoff against all adv strategies, to be added to the matrix
        if acceptable:
            for strategy_index in range(len(adv_mixed_strategy.strategies)):
                if adv_mixed_strategy.strategy_probs[strategy_index] == 0:
                    payoffs = []
                    for _ in range(gl.NUM_STOCHASTIC_ITER):
                        payoffs.append(model_strategy.play_against(
                            env=pricing_game, adversary=adv_mixed_strategy.strategies[strategy_index]))
                    mean_payoffs = np.array(payoffs).mean(axis=0)

                    agent_payoffs[strategy_index] = mean_payoffs[0]
                    adv_payoffs[strategy_index] = mean_payoffs[1]
            results.append((acceptable, agent_payoffs, adv_payoffs, model_strategy, expected_payoff, base_agents[id]))
    pool.close()
    pool.join()
    return results


if __name__=="__main__":
    env_class = ConPricingGame
    gl.initialize()

    num_rounds = 3

    job_name = "rnd_Mar11"
    db_name = job_name+".db"
    db = cl.DataBase(db_name)
    low_strts, high_strts=db.get_list_of_added_strategies()
    cl.set_job_name(job_name)
    # num_procs = gl.NUM_PROCESS if (len(sys.argv) < 2) else int(sys.argv[1])
    num_procs = 6



    # changing params
    lrs = [0.0003, 0.00016]
    memories = [12,18]
    # memories_agents=[[None]*len(memories)]*2
    algs = [SAC]

    equilibria = []

    cl.create_directories()

    # strt1 = cl.Strategy(
    #     cl.StrategyType.static, model_or_func=cl.myopic, name="myopic")
    # strt2 = cl.Strategy(
    #     cl.StrategyType.static, model_or_func=cl.const, name="const", first_price=132)
    # strt3 = cl.Strategy(
    #     cl.StrategyType.static, model_or_func=cl.guess, name="guess", first_price=132)
    # strt4 = cl.Strategy(
    #     cl.StrategyType.static, model_or_func=cl.spe, name="spe")

    train_env = env_class(tuple_costs=None, adversary_mixed_strategy=None, memory=12)
    model_name="rnd_start"
    log_dir = f"{gl.LOG_DIR}/{model_name}"
    model = SAC('MlpPolicy', train_env,
                            verbose=0, tensorboard_log=log_dir, gamma=gl.GAMMA, target_entropy=0)
    # model.learn(total_timesteps=1, tb_log_name=model_name)
    model.save(f"{gl.MODELS_DIR}/{model_name}")

    strt_rnd= cl.Strategy(strategy_type=cl.StrategyType.sb3_model,
                                    model_or_func=SAC, name=model_name, action_step=None,memory=12)

    bimatrix_game = cl.BimatrixGame(
        low_cost_strategies=[strt_rnd]+low_strts, high_cost_strategies=[strt_rnd]+high_strts, env_class=env_class)

    bimatrix_game.reset_matrix()
    bimatrix_game.fill_matrix()



    cl.prt("\n" + time.ctime(time.time())+"\n"+("-"*50)+"\n")

    dictionaries = bimatrix_game.compute_equilibria()
    game_size=bimatrix_game.size()

    # low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff = bimatrix_game.compute_equilibria()
    for round in range(num_rounds):
        cl.prt(f"Round {round} of {num_rounds}")
        
        added_low=0
        added_high=0
        # for equilibrium in dictionaries:
        for equi_i in range(len(dictionaries)):
            new_equi_low = 0
            new_equi_high = 0
            equi = dictionaries[equi_i]
            # low_prob_str = ", ".join(
            #     map("{0:.2f}".format, equi["low_cost_probs"]))
            # high_prob_str = ", ".join(
            #     map("{0:.2f}".format, equi["high_cost_probs"]))
            cl.prt(
                f'equi: {str(equi["low_cost_support"])}, {str(equi["high_cost_support"])}\n payoffs= {equi["low_cost_payoff"]:.2f}, {equi["high_cost_payoff"]:.2f}')
        
            # train a low-cost agent
            high_mixed_strat = cl.MixedStrategy(
                strategies_lst=bimatrix_game.high_strategies, probablities_lst=((equi["high_cost_probs"]+([0]*added_high)) if 
                                                                                added_high> 0 else equi["high_cost_probs"]))
        
            
            for alg in algs:
                for lr in lrs:
                    for mem_i,memory in enumerate(memories):
                        
                        print(f'training low-cost agents with alg={str(alg)}, lr={lr:.4f}, memory={memory}')
        
                        results= train_processes(db=db, env_class=env_class, costs=[gl.LOW_COST, gl.HIGH_COST], 
                                                adv_mixed_strategy=high_mixed_strat, target_payoff=equi["low_cost_payoff"], 
                                                num_procs=num_procs, alg=alg, lr=lr, memory=memory)
                        for result in results:
                            acceptable, agent_payoffs, adv_payoffs, agent_strategy, expected_payoff,base_agent_name = result
                            if acceptable:
                                new_equi_low += 1
                                added_low+=1
                                bimatrix_game.low_strategies.append(agent_strategy)
                                bimatrix_game.add_low_cost_row(agent_payoffs, adv_payoffs)
                                cl.prt(
                                    f'low-cost player {agent_strategy.name} , payoff= {expected_payoff:.2f} added, base={base_agent_name} ,alg={str(alg)}, lr={lr:.4f}, memory={memory}')
        
            # train a high-cost agent
            low_mixed_strat = cl.MixedStrategy(
                strategies_lst=bimatrix_game.low_strategies, probablities_lst=
                ((equi["low_cost_probs"]+([0]*added_low)) if added_low > 0 else equi["low_cost_probs"]))
            
            
            for alg in algs:
                for lr in lrs:
                    for memory in memories:
                        print(f'training high-cost player with alg={str(alg)}, lr={lr:.4f}, memory={memory}')
                        results= train_processes(db=db, env_class=env_class, costs=[ gl.HIGH_COST,gl.LOW_COST],
                                                adv_mixed_strategy=low_mixed_strat, target_payoff=equi["high_cost_payoff"],
                                                num_procs=num_procs, alg=alg, lr=lr, memory=memory)
                        for result in results:
                            acceptable, agent_payoffs, adv_payoffs, agent_strategy, expected_payoff,base_agent_name = result 
                            if acceptable:
                                new_equi_high += 1
                                added_high+=1
                                bimatrix_game.high_strategies.append(agent_strategy)
                                bimatrix_game.add_high_cost_col(adv_payoffs, agent_payoffs)
            
                                cl.prt(
                                    f'high-cost player {agent_strategy.name} , payoff= {expected_payoff:.2f} added, base={base_agent_name}, alg={str(alg)}, lr={lr:.4f}, memory={memory}')
            if new_equi_high>0:
                high_mixed_strat.strategy_probs+=[0]*new_equi_high
        
            # if new_equi_low>0 or new_equi_high>0:
                # equilibria.append(
                #     [equi["low_cost_probs"], equi["high_cost_probs"], equi["low_cost_payoff"], equi["high_cost_payoff"]])
                #to do: add the equilibria to the db
            db.insert_new_equi(game_size=game_size, low_strategy_txt=str(low_mixed_strat),high_strategy_txt=str(high_mixed_strat), low_payoff=equi["low_cost_payoff"], high_payoff=equi["high_cost_payoff"], low_new_num=new_equi_low, high_new_num=new_equi_high)
                
                
        if added_low==0 and added_high==0:
            gl.N_EPISODES_BASE *= 1.1
            gl.N_EPISODES_LOAD *= 1.1
        else:
            dictionaries = bimatrix_game.compute_equilibria()
            game_size=bimatrix_game.size()