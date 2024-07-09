from enum import Enum
import numpy as np
import src.globals as gl
# import torch
# from torch.distributions import Categorical
# from openpyxl import load_workbook
from fractions import Fraction
import src.bimatrix as bimatrix
import time
import os
import json
import sqlite3 as sql
from collections import namedtuple
from stable_baselines3 import SAC, PPO
import copy
import pickle
import src.environments as envs
import multiprocessing as mp
import pandas as pd
from typing import List, Dict


class Iter_row:
    def __init__(self, adv, agent_return, adv_return, rewards, adv_rewards, actions, prices, adv_prices, demands, adv_demands):
        self.adv = adv
        self.agent_return = agent_return
        self.adv_return = adv_return
        self.rewards = rewards
        self.adv_rewards = adv_rewards
        self.actions = actions
        self.prices = prices
        self.adv_prices = adv_prices
        self.demands = demands
        self.adv_demands = adv_demands


class BaseDataBase():
    def __init__(self, name="data.db") -> None:
        self.db_name = name
        self.reset()

    def reset(self) -> None:
        pass
    
    
    def execute_insert_query(self, query):
        connection = sql.connect(self.db_name, timeout=100)
        cursor = connection.cursor()
        cursor.execute(query)
        connection.commit()
        last_id = cursor.lastrowid
        connection.close()
        return last_id
    
    def batch_insert_to_table(self, table, values_list):
        """
        for each of the values string in values_list, inserts to the table. values should be of the form: "(... , ...)"
        """
        conn = sql.connect(self.db_name, timeout=200)
        res=None
        try:
            conn.execute('BEGIN TRANSACTION')  # Start a transaction
            
            cursor = conn.cursor()
            for values in values_list:
                cursor.execute(f'INSERT INTO {table} VALUES {values}')
            conn.commit()  # Commit the transaction
            res=cursor.lastrowid
        except Exception as e:
            # Log the error
            conn.rollback()  # Rollback the transaction in case of error
            prt(f"Transaction failed, rolled back. Error: {e}")
        finally:
            conn.close()
        return res
        
    def execute_select_query(self, query, fetch_one=False):
        connection = sql.connect(self.db_name, timeout=100)
        cursor = connection.cursor()
        cursor.execute(query)
        if fetch_one:
            result = cursor.fetchone()
        else:
            result = cursor.fetchall()
        return result

    def dataframe_select(self, query)->pd.DataFrame:
        connection = sql.connect(self.db_name, timeout=100)
        df = pd.read_sql_query(query, connection)
        connection.close()
        return df
    def delete_extra_rows(self, table_name, group_col_name, limit, id_col_name='id'):
        """
        groups the rows by group_col_name, keeps the first limit rows and delete the rest of them. This function is used for making the db lighter. The table should have a column called id to be used for deleting.
        """
        df=self.dataframe_select(f"select * from {table_name};")
        if not df.empty:
            connection = sql.connect(self.db_name, timeout=100)
            cursor = connection.cursor()
        
            gr_df=df.groupby(group_col_name)
            for name, data in gr_df:
                if len(data)>limit:
                    del_ids=[]
                    for ind in range(limit, len(data)):
                        del_ids.append(data.at[data.index[ind],id_col_name])
                    cursor.execute(f"DELETE FROM {table_name} WHERE {id_col_name} IN {tuple(del_ids)}")
            connection.commit()
            cursor.execute("VACUUM")
            connection.close()            
            


class DataBase(BaseDataBase):

    AGENTS_TABLE = "trained_agents"
    ITERS_TABLE = "agents_iters"
    EQUI_TABLE = "game_equilibria"
    PROBS_TABLE = "strategy_average_probs"

    AGENTS_COLS = "id integer PRIMARY  key AUTOINCREMENT,time text,name text NOT NULL,base_agent text DEFAULT NULL,n_ep integer NOT NULL,cost integer NOT NULL,mixed_adv text NOT NULL,alg text NOT NULL,seed integer,num_procs integer DEFAULT 1,running_time  integer,return_std real, expected_payoff real,payoff_treshhold real,added integer, equi_id integer"
    # id and time at the start should be added separately
    AgentRow = namedtuple(
        "AgentRow", "name, base_agent,  n_ep, cost, mixed_adv_txt, alg, seed, num_process, running_time, return_std,  expected_payoff, payoff_treshhold, added, equi_id")
    ITERS_COLS = "id integer PRIMARY key AUTOINCREMENT,agent_id integer NOT NULL,adv text  NOT NULL,agent_return text,adv_return text,agent_rewards text,adv_rewards text,actions text,agent_prices text,adv_prices text, agent_demands text,adv_demands text"
    # id should be added as NULL
    IterRow = namedtuple(
        "IterRow", "agent_id,adv,agent_return,adv_return,agent_rewards,adv_rewards,actions,agent_prices,adv_prices,agent_demands,adv_demands")

    EQUI_COLS = "id integer PRIMARY key AUTOINCREMENT,time text,game_size text NOT NULL,freq real,low_strategy text NOT NULL,high_strategy text NOT NULL,low_payoff real NOT NULL,high_payoff real NOT NULL, used INTEGER DEFAULT 0 ,num_new_low integer DEFAULT 0, num_new_high integer DEFAULT 0"
    # id and time at the start should be added separately
    EquiRow = namedtuple(
        "EquiRow", "game_size, freq,low_strategy,high_strategy,low_payoff,high_payoff,used,num_new_low,num_new_high")

    PROBS_COLS = "id integer PRIMARY key AUTOINCREMENT, time text,game_size text,equi_count integer,last_equi_id integer, cost integer, strategy_probs text"
    # id and time at the start should be added separately
    ProbsRow = namedtuple("ProbsRow", "game_size,equi_count, last_equi_id,cost, strategy_probs")

    def reset(self):
        connection = sql.connect(self.db_name, timeout=100)
        cursor = connection.cursor()
        cursor.execute(
            f'CREATE TABLE IF NOT EXISTS {self.AGENTS_TABLE}({self.AGENTS_COLS});')
        cursor.execute(
            f'CREATE TABLE IF NOT EXISTS {self.ITERS_TABLE}({self.ITERS_COLS});')
        cursor.execute(
            f'CREATE TABLE IF NOT EXISTS {self.EQUI_TABLE}({self.EQUI_COLS}, UNIQUE(game_size,low_strategy,high_strategy));')
        cursor.execute(
            f'CREATE TABLE IF NOT EXISTS {self.PROBS_TABLE}({self.PROBS_COLS}, UNIQUE(game_size,last_equi_id,cost));')
        connection.close()

    # def insert_new_agent(self, name, base_agent,  num_ep, cost, mixed_adv_txt, expected_payoff, payoff_treshhold, lr, memory, added, action_step=None, seed=0, num_process=1, running_time=0):

    def insert_new_agent(self, row: AgentRow):
        """
         adds a new agent to db and returns the id
         row: AgentRow named tuple
        """
        # "name, base_agent,  n_ep, cost, mixed_adv_txt, alg, seed, num_process, running_time, return_std,  expected_payoff, payoff_treshhold, added"
        query = f'INSERT INTO {self.AGENTS_TABLE} VALUES (NULL,\'{ time.ctime(time.time())}\',\'{row.name}\',' + ('NULL' if (row.base_agent is None) else f'\'{row.base_agent}\'') + \
            f',{row.n_ep},{row.cost},\'{row.mixed_adv_txt}\',\"{row.alg}\",{row.seed},{row.num_process},{row.running_time},{row.return_std},{row.expected_payoff},{row.payoff_treshhold},{int(row.added)},{row.equi_id})'
        # print(query)
        return self.execute_insert_query(query=query)

    def insert_new_iteration(self, agent_id, adv_txt, agent_return, adv_return, agent_rewards_txt, adv_rewards_txt, actions_txt, agent_prices_txt, adv_prices_txt, agent_demands_txt, adv_demands_txt):
        """
        adds a new iteration to db and returns the id
        """
        # "agent_id,adv,agent_return,adv_return,agent_rewards,adv_rewards,actions,agent_prices,adv_prices,agent_demands,adv_demands"
        query = f'INSERT INTO {self.ITERS_TABLE} VALUES (NULL,{agent_id},\'{adv_txt}\',{agent_return},{adv_return},\'{agent_rewards_txt}\',\'{adv_rewards_txt}\',\
            \'{actions_txt}\',\'{agent_prices_txt}\',\'{adv_prices_txt}\',\'{agent_demands_txt}\',\'{adv_demands_txt}\')'
        # print(query)
        return self.execute_insert_query(query=query)
    
    def insert_many_new_iters(self, iterRow_list:List[IterRow]):
        values_list=[]
        for iterRow in iterRow_list:
            values_list.append( f'(NULL,{iterRow.agent_id},\'{iterRow.adv}\',{iterRow.agent_return},{iterRow.adv_return},\'{iterRow.agent_rewards}\',\'{iterRow.adv_rewards}\',\
            \'{iterRow.actions}\',\'{iterRow.agent_prices}\',\'{iterRow.adv_prices}\',\'{iterRow.agent_demands}\',\'{iterRow.adv_demands}\')')
        self.batch_insert_to_table(self.ITERS_TABLE,values_list)
        

    def insert_new_equi(self, game_size, freq, low_strategy_txt, high_strategy_txt, low_payoff, high_payoff, used, num_new_low, num_new_high):
        """
        adds a new equilibrium to db and returns the id
        """
        # check if this equi is already added
        row=self.execute_select_query(f"SELECT id FROM {self.EQUI_TABLE} WHERE game_size=\'{str(game_size)}\' AND low_strategy = \'{low_strategy_txt}\' AND high_strategy = \'{high_strategy_txt}\';",fetch_one=True)
        if row:
            return row[0]
        else:
        # "game_size, freq,low_strategy,high_strategy,low_payoff,high_payoff,used,num_new_low,num_new_high"
            query = f'INSERT INTO {self.EQUI_TABLE} VALUES (NULL, \'{ time.ctime(time.time())}\',\'{str(game_size)}\',{freq},\'{low_strategy_txt}\',\'{high_strategy_txt}\',{low_payoff},{high_payoff},{int(used)},{num_new_low},{num_new_high})'
            # print(query)
            return self.execute_insert_query(query=query)

    def update_equi(self, id, used, num_new_low, num_new_high):
        """
        upadtes details of the used equi after training
        """
        query = f"UPDATE {self.EQUI_TABLE} SET used={used}, num_new_low={num_new_low}, num_new_high={num_new_high} WHERE id={id};"
        return self.execute_insert_query(query=query)

    def insert_new_average_probs(self, game_size, equi_count, last_equi_id, cost, strategy_probs_str):
        """
        adds a new row of average probablity of strategies in equilibria to db and returns the id. For each equi two rows should be added, one for each cost
        """
        # check if this equi is already added to the average 
        # game_size,last_equi_id,cost
        row=self.execute_select_query(f"SELECT id FROM {self.PROBS_TABLE} WHERE game_size=\'{str(game_size)}\' AND last_equi_id = {last_equi_id} AND cost = {cost};",fetch_one=True)
        if row:
            return row[0]
        else:
        # "game_size, freq,low_strategy,high_strategy,low_payoff,high_payoff,used,num_new_low,num_new_high"
            query = f'INSERT INTO {self.PROBS_TABLE} VALUES (NULL, \'{ time.ctime(time.time())}\',\'{str(game_size)}\',{equi_count},{last_equi_id},{cost},\"{strategy_probs_str}\")'
            # print(query)
            return self.execute_insert_query(query=query)

    def updates_equi_average_probs(self, equi_id: int, equi: bimatrix.Equi):
        """updates average probs based on the previous average and the new equi"""
        game_size = (len(equi.row_probs), len(equi.col_probs))
        for cost in [gl.LOW_COST, gl.HIGH_COST]:
            new_prbs = equi.row_probs if cost == gl.LOW_COST else equi.col_probs
            query = f"select * from {self.PROBS_TABLE} where cost={cost} order by id desc limit 1;"
            df = self.dataframe_select(query)
            if df.empty:
                equi_count = 1
                avg_prbs = new_prbs
            else:
                row = df.iloc[0]
                equi_count = row['equi_count'] + 1
                
                prev_prbs = json.loads(row['strategy_probs'])
                if (extra:=len(new_prbs)-len(prev_prbs)) >0 :
                    prev_prbs.extend([0]*extra)
                avg_prbs = ((np.array(new_prbs) + (np.array(prev_prbs) * (equi_count-1)))/equi_count).tolist()
            self.insert_new_average_probs(game_size, equi_count, equi_id, cost, str(avg_prbs))

    def get_list_of_added_strategies(self,memory, action_step):
        """ returns two lists of low_cost and high_cost strategies """
        low_q = f"SELECT name, alg FROM {self.AGENTS_TABLE} WHERE (added=1 and cost={gl.LOW_COST})"
        high_q = f"SELECT name, alg FROM {self.AGENTS_TABLE} WHERE (added=1 and cost={gl.HIGH_COST})"
        low_lst = []
        high_lst = []

        connection = sql.connect(self.db_name, timeout=100)
        cursor = connection.cursor()

        cursor.execute(low_q)
        low_all = cursor.fetchall()
        for tup in low_all:
            model=alg_classes[tup[1]]
            low_lst.append(Strategy(strategy_type=StrategyType.sb3_model, model_or_func=model,
                           name=tup[0], memory=memory, action_step=action_step))

        cursor.execute(high_q)
        high_all = cursor.fetchall()
        for tup in high_all:
            model=alg_classes[tup[1]] 
            high_lst.append(Strategy(strategy_type=StrategyType.sb3_model, model_or_func=model,
                            name=tup[0], memory=memory, action_step=action_step))
        connection.close()
        return low_lst, high_lst


class BimatrixGame():
    """
    strategies play against each other and fill the matrix of payoff, then the equilibria would be computed using Lemke algorithm
    """

    def __init__(self, low_cost_strategies, high_cost_strategies, env_class) -> None:
        # globals.initialize()
        self.low_strategies = low_cost_strategies
        self.high_strategies = high_cost_strategies
        self.env_class = env_class

    def size(self):
        return (len(self.low_strategies), len(self.high_strategies))

    def get_subgame(self, num_row, num_col):
        sub_game = BimatrixGame(low_cost_strategies=self.low_strategies[:num_row],
                                high_cost_strategies=self.high_strategies[:num_col],
                                env_class=self.env_class)
        sub_game.matrix_A = self.matrix_A[:num_row, :num_col]
        sub_game.matrix_B = self.matrix_B[:num_row, :num_col]
        return sub_game

    def to_dict(self):
        return {
            'low_strategies': [strt.to_dict() for strt in self.low_strategies],
            'high_strategies': [strt.to_dict() for strt in self.high_strategies],
            'env_class': self.env_class.__name__,
            'matrix_A': self.matrix_A,
            'matrix_B': self.matrix_B
        }

    @classmethod
    def from_dict(cls, data_dict):

        env_class = str_to_envclass(data_dict['env_class'])
        low_strategies = [Strategy.from_dict(strt_data) for strt_data in data_dict['low_strategies']]
        high_strategies = [Strategy.from_dict(strt_data) for strt_data in data_dict['high_strategies']]

        obj = cls(low_cost_strategies=low_strategies, high_cost_strategies=high_strategies, env_class=env_class)
        obj.matrix_A = data_dict['matrix_A']
        obj.matrix_B = data_dict['matrix_B']
        return obj

    def save_game(self, file_name):
        with open(f"{file_name}.pickle", "wb") as file:
            pickle.dump(self.to_dict(), file)

    def load_game(file_name):
        name = f"{file_name}.pickle"
        if os.path.exists(name):
            with open(name, "rb") as file:
                instance_data = pickle.load(file)
                return BimatrixGame.from_dict(instance_data)
        else:
            return None

    def reset_matrix(self):
        self.matrix_A = np.zeros(
            (len(self.low_strategies), len(self.high_strategies)))
        self.matrix_B = np.zeros(
            (len(self.low_strategies), len(self.high_strategies)))

    def fill_matrix(self):
        self.reset_matrix()
        for low in range(len(self.low_strategies)):
            for high in range(len(self.high_strategies)):
                self.update_matrix_entry(low, high)

    def update_matrix_entry(self, low_index, high_index):
        strt_L = self.low_strategies[low_index]
        strt_H = self.high_strategies[high_index]
        strt_L.reset()
        strt_H.reset()

        env = self.env_class(tuple_costs=(
            gl.LOW_COST, gl.HIGH_COST), adversary_mixed_strategy=strt_H.to_mixed_strategy(), memory=strt_L.memory)
        payoffs = [strt_L.play_against(env, strt_H)
                   for _ in range(gl.NUM_MATRIX_ITER)]

        mean_payoffs = (np.mean(np.array(payoffs), axis=0))

        self.matrix_A[low_index][high_index], self.matrix_B[low_index][high_index] = mean_payoffs[0], mean_payoffs[1]

    def write_all_matrix(self):
        # print("A: \n", self._matrix_A)
        # print("B: \n", self._matrix_B)

        output= f"{len(self.matrix_A)} {len(self.matrix_A[0])}\n\n"
        int_output= f"{len(self.matrix_A)} {len(self.matrix_A[0])}\n\n"
        game_name = f"game_{job_name}"

        self.save_game(game_name)

        for matrix in [self.matrix_A, self.matrix_B]:
            for i in range(len(self.matrix_A)):
                for j in range(len(self.matrix_A[0])):
                    output += f"{matrix[i][j]:8.3f} "
                    int_output += f"{matrix[i][j]:5.0f} "
                output += "\n"
                int_output += "\n"
            output += "\n"
            int_output += "\n"

        with open(f"{game_name}.txt", "w") as out:
            out.write(output)
        int_output += "\nlow-cost strategies: \n"
        for strt in self.low_strategies:
            int_output += f" {strt.name} "

        int_output += "\nhigh-cost strategies: \n"
        for strt in self.high_strategies:
            int_output += f" {strt.name} "

        with open(f"games/game_{job_name}_{int(time.time())}.txt", "w") as out:
            out.write(int_output)

    def add_low_cost_row(self, row_A, row_B):
        self.matrix_A = np.append(self.matrix_A, [row_A], axis=0)
        self.matrix_B = np.append(self.matrix_B, [row_B], axis=0)

    def add_high_cost_col(self, colA, colB):

        self.matrix_A = np.hstack((self.matrix_A, np.atleast_2d(colA).T))
        self.matrix_B = np.hstack((self.matrix_B, np.atleast_2d(colB).T))
        # for j in range(len(self._matrix_A)):
        #     self._matrix_A[j].append(colA[j])
        #     self._matrix_B[j].append(colB[j])

    def compute_equilibria(self, num_trace=100, write_matrix=True, prt_progress=True) -> List[bimatrix.Equi]:
        """ returns a list of Equi, all the equilibria found for the bimatrix game"""
        if write_matrix:
            self.write_all_matrix()
        game = bimatrix.bimatrix(f"game_{job_name}.txt")
        equilibria_all = game.tracing(num_trace)
        if prt_progress:
            prt(f"\nall equilibria: {len(equilibria_all)}")
            for i in range(len(equilibria_all)):
                prt(f"{i} - {equilibria_all[i]}")

        return equilibria_all

        # return equilibria_all[0:(min(gl.NUM_TRACE_EQUILIBRIA, len(equilibria_all)))]
        # for equilibrium in equilibria_traces:
        #     low_cost_probs, high_cost_probs, low_cost_support, high_cost_support = recover_probs(
        #         equilibrium)
        #     low_cost_probabilities = return_distribution(
        #         len(self.low_strategies), low_cost_probs, low_cost_support)
        #     high_cost_probabilities = return_distribution(
        #         len(self.high_strategies), high_cost_probs, high_cost_support)
        #     low_cost_payoff = np.matmul(low_cost_probabilities, np.matmul(
        #         self.matrix_A, np.transpose(high_cost_probabilities)))
        #     high_cost_payoff = np.matmul(low_cost_probabilities, np.matmul(
        #         self.matrix_B, np.transpose(high_cost_probabilities)))

        #     result = {"low_cost_probs": low_cost_probabilities,
        #               "high_cost_probs": high_cost_probabilities,
        #               "low_cost_payoff": low_cost_payoff,
        #               "high_cost_payoff": high_cost_payoff,
        #               "low_cost_support": low_cost_support,
        #               "high_cost_support": high_cost_support
        #               }
        #     equilibria.append(result)
        # return equilibria


class Strategy():
    """
    strategies can be static or they can be models trained with sb3.
    """
    type = None
    env = None
    name = None
    memory = None
    policy = None
    model = None
    first_price = None

    def __init__(self, strategy_type, model_or_func, name, first_price=132, memory=0, action_step=None) -> None:
        """
        model_or_func: for static strategy is the function, for sb3 is the optimizer class
        """
        self.type = strategy_type
        self.name = name
        # self._env = environment
        self.memory = memory

        self.action_step = action_step

        if strategy_type == StrategyType.sb3_model:
            self.dir = f"{gl.MODELS_DIR}/{name}"
            self.model = model_or_func
            # self.policy = self.model.predict

        else:
            self.policy = model_or_func
            self.first_price = first_price

    def __str__(self) -> str:
        return f"{self.name}:{self.memory},{self.action_step}"

    def reset(self):
        pass

    def to_dict(self):
        return {
            'type': self.type,
            'name': self.name,
            'model_or_func': (self.model if self.type == StrategyType.sb3_model else self.policy),
            'first_price': self.first_price,
            'memory': self.memory,
            'action_step': self.action_step
        }

    @classmethod
    def from_dict(cls, data_dict):
        return cls(strategy_type=data_dict['type'], model_or_func=data_dict['model_or_func'], name=data_dict['name'], first_price=data_dict['first_price'], memory=data_dict['memory'], action_step=data_dict['action_step'])

    def play(self, env, player=1):
        """
            Computes the price to be played in the environment, nn.step_action is the step size for pricing less than myopic
        """

        if self.type == StrategyType.sb3_model:
            if self.policy is None:
                if env.memory != self.memory:
                    env_new = (env.__class__)(tuple_costs=env.costs,
                                              adversary_mixed_strategy=env.adversary_mixed_strategy, memory=self.memory)
                    # env_new shouldn't it be multi level?
                    self.policy = (self.model.load(self.dir, env=env_new)).predict
                else:
                    self.policy = (self.model.load(self.dir, env=env)).predict
            state = env.get_state(
                stage=env.stage, player=player, memory=self.memory)
            action, _ = self.policy(state)
            # compute price for co model and disc model
            price = (env.myopic(player)-action[0]) if (self.action_step is None) else (
                env.myopic(player)-(self.action_step*action))

            if player == 0:
                env.actions[env.stage] = (action[0] if (
                    self.action_step is None) else (self.action_step*action))

            return price
        else:
            return self.policy(env, player, self.first_price)

    def play_against(self, env, adversary:'Strategy'):
        """ 
        self is player 0 and adversary is player 1. The environment should be specified. action_step for the neural netwroks should be set.
        output: tuple (payoff of low cost, payoff of high cost)
        """
        # self.env = env
        env.adversary_mixed_strategy = adversary.to_mixed_strategy()

        state, _ = env.reset()
        while env.stage < (env.T):
            prices = [0, 0]
            prices[0], prices[1] = self.play(env, 0), adversary.play(env, 1)
            env.update_game_variables(prices)
            env.stage += 1

        return [sum(env.profit[0]), sum(env.profit[1])]

    def to_mixed_strategy(self):
        """
        Returns a MixedStrategy, Pr(self)=1
        """
        mix = MixedStrategy(probablities_lst=[1],
                            strategies_lst=[self])

        return mix


class MixedStrategy():
    strategies = []
    strategy_probs = None

    def __init__(self, strategies_lst, probablities_lst) -> None:
        self.strategies = strategies_lst
        self.strategy_probs = probablities_lst
        self.support_size = support_count(probablities_lst)

    def choose_strategy(self):
        if len(self.strategies) > 0:
            # adversaryDist = Categorical(torch.tensor(self._strategyProbs))
            # if not torch.is_tensor(self._strategyProbs):
            #     self._strategyProbs = torch.tensor(self._strategyProbs)
            # adversaryDist = Categorical(self._strategyProbs)
            # strategyInd = (adversaryDist.sample()).item()
            strategy_ind = np.random.choice(
                len(self.strategies), size=1, p=self.strategy_probs)
            return self.strategies[strategy_ind[0]]
        else:
            print("adversary's strategy can not be set!")
            return None

    def play_against(self, env, adversary):
        pass

    def __str__(self) -> str:
        if len(self.strategies) != len(self.strategy_probs):
            print(len(self.strategies))
            print(self.strategy_probs)
        s = ""
        for i in range(len(self.strategies)):
            if self.strategy_probs[i] > 0:
                s += f"{self.strategies[i].name}-{self.strategy_probs[i]:.2f},"
        return s

    def reduce(self):
        """ only keeps the strategies with positive probablity, returns a new mixed strategy"""
        strts = []
        probs = []
        for i in range(len(self.strategies)):
            if self.strategy_probs[i] > 0:
                strts.append(self.strategies[i])
                probs.append(self.strategy_probs[i])
        return MixedStrategy(strategies_lst=strts, probablities_lst=probs)

    def copy_unload(self):
        """a copy of strategies with models not loaded, returns a mixed strategy """
        strts = []
        probs = []
        for i in range(len(self.strategies)):
            strt = copy.deepcopy(self.strategies[i])
            if self.strategies[i].type == StrategyType.sb3_model:
                strt.policy = None
            strts.append(strt)
            probs.append(self.strategy_probs[i])

        return MixedStrategy(strategies_lst=strts, probablities_lst=probs)


class StrategyType(Enum):
    static = 0
    neural_net = 1
    sb3_model = 2


def myopic(env, player, firstprice=0):
    """
        Adversary follows Myopic strategy
    """
    return env.myopic(player)


def const(env, player, firstprice):  # constant price strategy
    """
        Adversary follows Constant strategy
    """
    if env.stage == env.T-1:
        return env.myopic(player)
    return firstprice


def imit(env, player, firstprice):  # price imitator strategy
    if env.stage == 0:
        return firstprice
    if env.stage == env.T-1:
        return env.myopic(player)
    return env.prices[1-player][env.stage-1]


def fight(env, player, firstprice):  # simplified fighting strategy
    if env.stage == 0:
        return firstprice
    if env.stage == env.T-1:
        return env.myopic(player)
    # aspire = [ 207, 193 ] # aspiration level for demand potential
    aspire = [0, 0]
    for i in range(2):
        aspire[i] = (env.total_demand-env.costs[player] +
                     env.costs[1-player])/2

    D = env.demand_potential[player][env.stage]
    Asp = aspire[player]
    if D >= Asp:  # keep price; DANGER: price will never rise
        return env.prices[player][env.stage-1]
    # adjust to get to aspiration level using previous
    # opponent price; own price has to be reduced by twice
    # the negative amount D - Asp to getenv.demandPotential to Asp
    P = env.prices[1-player][env.stage-1] + 2*(D - Asp)
    # never price to high because even 125 gives good profits
    # P = min(P, 125)
    aspire_price = (env.total_demand+env.costs[0]+env.costs[1])/4
    P = min(P, int(0.95*aspire_price))

    return P


def fight_lb(env, player, firstprice):
    P = env.fight(player, firstprice)
    # never price less than production cost
    P = max(P, env.costs[player])
    return P

# sophisticated fighting strategy, compare fight()
# estimate *sales* of opponent as their target


def guess(env, player, firstprice):  # predictive fighting strategy
    if env.stage == 0:
        # only first round these should be set in the env
        env.aspireDemand = [(env.total_demand/2 + env.costs[1]-env.costs[0]),
                            (env.total_demand/2 + env.costs[0]-env.costs[1])]  # aspiration level
        env.aspirePrice = (env.total_demand+env.costs[0]+env.costs[1])/4
        # first guess opponent sales as in monopoly ( sale= demand-price)
        env.saleGuess = [env.aspireDemand[0]-env.aspirePrice,
                            env.aspireDemand[1]-env.aspirePrice]

        return firstprice
    

    if env.stage == env.T-1:
        return env.myopic(player)

    D = env.demand_potential[player][env.stage]
    Asp = env.aspireDemand[player]

    if D >= Asp:  # keep price, but go slightly towards monopoly if good
        pmono = env.myopic(player)
        pcurrent = env.prices[player][env.stage-1]
        if pcurrent > pmono:  # shouldn't happen
            return pmono
        elif pcurrent > pmono-7:  # no change
            return pcurrent
        # current low price at 60%, be accommodating towards "collusion"
        return .6 * pcurrent + .4 * (pmono-7)

    # guess current *opponent price* from previous sales
    prevsales = env.demand_potential[1 -
                                     player][env.stage-1] - env.prices[1-player][env.stage-1]
    # adjust with weight alpha from previous guess
    alpha = .5
    newsalesguess = alpha * env.saleGuess[player] + (1-alpha)*prevsales
    # update
    env.saleGuess[player] = newsalesguess
    guessoppPrice = env.total_demand - D - newsalesguess
    # D < Asp 
    P = guessoppPrice + 2*(D - Asp)

    if player == 0:
        P = min(P, 125)
    if player == 1:
        P = min(P, 130)
    return P

def guess2(env, player, firstprice):  
    """ more cooperative guess"""
    if env.stage == 0:
        env.aspireDemand = [(env.total_demand/2 + env.costs[1]-env.costs[0]),
                            (env.total_demand/2 + env.costs[0]-env.costs[1])]  # aspiration level
        env.aspirePrice = (env.total_demand+env.costs[0]+env.costs[1])/4
        # first guess opponent sales as in monopoly ( sale= demand-price)
        env.saleGuess = [env.aspireDemand[0]-env.aspirePrice,
                         env.aspireDemand[1]-env.aspirePrice]

        return firstprice

    if env.stage == env.T-1:
        return env.myopic(player)

    D = env.demand_potential[player][env.stage]
    Asp = env.aspireDemand[player]
    allowed_range=15
    if D >= Asp:  # keep price, but go slightly towards monopoly if good
        pmono = env.myopic(player)
        pcurrent = env.prices[player][env.stage-1]
        if pcurrent > pmono:  # shouldn't happen
            return pmono
        elif pcurrent > pmono-allowed_range:  # no change
            return pcurrent
        # current low price at 60%, be accommodating towards "collusion"
        return .6 * pcurrent + .4 * (pmono-allowed_range)

    # guess current *opponent price* from previous sales
    prevsales = env.demand_potential[1 -
                                     player][env.stage-1] - env.prices[1-player][env.stage-1]
    # adjust with weight alpha from previous guess
    alpha = .5
    newsalesguess = alpha * env.saleGuess[player] + (1-alpha)*prevsales
    # update
    env.saleGuess[player] = newsalesguess
    guessoppPrice = env.total_demand - D - newsalesguess
    P = guessoppPrice + 2*(D - Asp)

    if player == 0:
        P = min(P, 135)
    if player == 1:
        P = min(P, 140)
    return P

def guess3(env, player, firstprice):  
    """ less cooperative guess"""
    if env.stage == 0:
        env.aspireDemand = [(env.total_demand/2 + env.costs[1]-env.costs[0]),
                            (env.total_demand/2 + env.costs[0]-env.costs[1])]  # aspiration level
        env.aspirePrice = (env.total_demand+env.costs[0]+env.costs[1])/4
        # first guess opponent sales as in monopoly ( sale= demand-price)
        env.saleGuess = [env.aspireDemand[0]-env.aspirePrice,
                         env.aspireDemand[1]-env.aspirePrice]

        return firstprice

    if env.stage == env.T-1:
        return env.myopic(player)

    D = env.demand_potential[player][env.stage]
    Asp = env.aspireDemand[player]
    allowed_range=3
    if D >= Asp:  # keep price, but go slightly towards monopoly if good
        pmono = env.myopic(player)
        pcurrent = env.prices[player][env.stage-1]
        if pcurrent > pmono:  # shouldn't happen
            return pmono
        elif pcurrent > pmono-allowed_range:  # no change
            return pcurrent
        # current low price at 60%, be accommodating towards "collusion"
        return .6 * pcurrent + .4 * (pmono-allowed_range)

    # guess current *opponent price* from previous sales
    prevsales = env.demand_potential[1 -
                                     player][env.stage-1] - env.prices[1-player][env.stage-1]
    # adjust with weight alpha from previous guess
    alpha = .5
    newsalesguess = alpha * env.saleGuess[player] + (1-alpha)*prevsales
    # update
    env.saleGuess[player] = newsalesguess
    guessoppPrice = env.total_demand - D - newsalesguess
    P = guessoppPrice + 2*(D - Asp)

    if player == 0:
        P = min(P, 120)
    if player == 1:
        P = min(P, 125)
    return P


def spe(env, player, firstprice=0):
    """
    returns the subgame perfect equilibrium price
    """
    t = env.stage
    P = gl.SPE_a[t]*(env.demand_potential[player][t]-200) + gl.SPE_b[t] + gl.SPE_k[t]*(env.costs[player]-64)
    return P


def monopolyPrice(demand, cost):  # myopic monopoly price
    """
        Computes Monopoly prices.
    """
    return (demand + cost) / 2
    # return (self.demandPotential[player][self.stage] + self.costs[player])/2


def prt(string):
    """
    writing the progres into a file instead of print
    """
    global job_name
    with open(f'progress_{job_name}.txt', 'a') as file:
        file.write("\n"+string)


# def write_to_excel(file_name, new_row):
#     """
#     row includes:  name	ep	costs	adversary	agent_return	adv_return	agent_rewards	actions	agent_prices	adv_prices	agent_demands	adv_demands	lr	hist	total_stages	action_step	num_actions	gamma	stae_onehot	seed	num_procs	running_time
#     """

#     path = f'results_{job_name}.xlsx' if (file_name is None) else file_name

#     wb = load_workbook(path)
#     sheet = wb.active
#     row = 2
#     col = 1
#     sheet.insert_rows(idx=row)

#     for i in range(len(new_row)):
#         sheet.cell(row=row, column=col+i).value = new_row[i]
#     wb.save(path)


# def write_results(new_row):
#     write_to_excel(f'results_{job_name}.xlsx', new_row)


# def write_agents(new_row):
#     # name	ep	costs	adversary	expected_payoff	payoff_treshhold	lr	hist	total_stages	action_step	num_actions\
#     # gamma	seed	num_procs	running_time	date

#     write_to_excel(f'trained_agents_{job_name}.xlsx', new_row)


def support_count(list):
    """
    gets a list and returns the number of elements that are greater than zero
    """
    counter = 0
    for item in list:
        if item > 0:
            counter += 1
    return counter


# def recover_probs(test):
#     low_cost_probs, high_cost_probs, rest = test.split(")")
#     low_cost_probs = low_cost_probs.split("(")[1]
#     _, high_cost_probs = high_cost_probs.split("(")
#     high_cost_probs = [float(Fraction(s)) for s in high_cost_probs.split(',')]
#     low_cost_probs = [float(Fraction(s)) for s in low_cost_probs.split(',')]
#     _, low_cost_support, high_cost_support = rest.split('[')
#     high_cost_support, _ = high_cost_support.split(']')
#     high_cost_support = [int(s) for s in high_cost_support.split(',')]
#     low_cost_support, _ = low_cost_support.split(']')
#     low_cost_support = [int(s) for s in low_cost_support.split(',')]
#     return low_cost_probs, high_cost_probs, low_cost_support, high_cost_support


# def return_distribution(number_players, cost_probs, cost_support):
#     player_probabilities = [0] * number_players
#     for index, support in enumerate(cost_support):
#         player_probabilities[support] = cost_probs[support]
#     return player_probabilities


def create_directories():
    if not os.path.exists(gl.MODELS_DIR):
        os.makedirs(gl.MODELS_DIR)
    if not os.path.exists(gl.LOG_DIR):
        os.makedirs(gl.LOG_DIR)
    if not os.path.exists(gl.GAMES_DIR):
        os.makedirs(gl.GAMES_DIR)


def set_job_name(name):
    global job_name
    job_name = name


def find_base_agents(db, alg, memory, cost, mix_strt, size):
    """ the startegies should be the same class of agents as we are training. if low cost then low-cost strategies should be given to find similar ones. The trained agents that are not even added will be considered """
    # is mix_strt adversary's strategy? then why?! the costs are different
    strats = copy.deepcopy(mix_strt.strategies)
    probs = copy.deepcopy(mix_strt.strategy_probs)
    cands = [None]
    cand_w = [1.00]
    for i in range(len(probs)-1):
        for j in range(i+1, len(probs)):
            if probs[i] < probs[j]:
                strats[i], strats[j] = strats[j], strats[i]
                probs[i], probs[j] = probs[j], probs[i]
    for st in strats:
        if st.type == StrategyType.sb3_model and memory == st.memory and (alg is st.model):
            cands.append(st.name)
            cand_w.append(2.00)

    query = f'SELECT name FROM {DataBase.AGENTS_TABLE} WHERE cost={cost} and memory={memory} and alg=\"{str(alg)}\" and added=1 ORDER BY id DESC LIMIT {(size-len(cands))}'

    tmps = db.execute_select_query(query=query)
    if tmps is not None:
        for tmp in tmps:
            cands.append(tmp[0])
            cand_w.append(1.00)
    cand_w = np.array(cand_w)
    cand_w /= sum(cand_w)
    agents = np.random.choice(np.array(cands), size, replace=True, p=cand_w)
    return agents


def find_base_agent(db, alg, cost, own_mix_strt):
    """ the startegies should be the same class of agents as we are training. if low cost then low-cost strategies should be given to find similar ones. also there is some chance of training from zero """
    strats = copy.deepcopy(own_mix_strt.strategies)
    probs = copy.deepcopy(own_mix_strt.strategy_probs)
    cands = [None]
    cand_w = [1.00]
    for i in range(len(probs)-1):
        for j in range(i+1, len(probs)):
            if probs[i] < probs[j]:
                strats[i], strats[j] = strats[j], strats[i]
                probs[i], probs[j] = probs[j], probs[i]
    for st in strats:
        if st.type == StrategyType.sb3_model and (alg is st.model):
            cands.append(st.name)
            cand_w.append(1.00)

    query = f'SELECT name FROM {DataBase.AGENTS_TABLE} WHERE cost={cost} and alg=\"{name_of(alg)}\" and added=1'

    tmps = db.execute_select_query(query=query)
    if tmps is not None:
        for tmp in tmps:
            cands.append(tmp[0])
            cand_w.append(1.00)
    cand_w = np.array(cand_w)
    cand_w /= sum(cand_w)
    agent = np.random.choice(np.array(cands), p=cand_w)
    return agent


def read_matrices_from_file(file_name):
    lines = [*open(file=file_name)]
    size = tuple(int(num) for num in lines[0].split())
    matrix_A = np.zeros(size)
    matrix_B = np.zeros(size)

    for i in range(2, 2+size[0]):
        matrix_A[i-2] = [float(num) for num in lines[i].split()]
    for i in range(3+size[0], 3+2*size[0]):
        matrix_B[i-3-size[0]] = [float(num) for num in lines[i].split()]

    return matrix_A, matrix_B


def load_latest_game(game_data_name, new_game):
    """ loads the game from game_data file and adds the extra strategies from new_game that were not saved in the data file and should be in the game"""

    old_game = BimatrixGame.load_game(game_data_name)
    if old_game is None:  # no data to load
        new_game.reset_matrix()
        new_game.fill_matrix()
        return new_game
    else:
        new_lows = new_game.low_strategies
        new_highs = new_game.high_strategies

        low_trained_i = 0
        while low_trained_i < len(old_game.low_strategies) and old_game.low_strategies[low_trained_i].type != StrategyType.sb3_model:
            low_trained_i += 1

        low_new_trained_i = 0
        while low_new_trained_i < len(new_lows) and new_lows[low_new_trained_i].type != StrategyType.sb3_model:
            low_new_trained_i += 1

        low_extra_start = len(old_game.low_strategies) - low_trained_i-1 + low_new_trained_i
        # low_extra_start shows index of 1st low strategy that is not in the previous saved game, all strategies after this index should be added to the game
        while low_extra_start >= 0 and (new_lows[low_extra_start].name != old_game.low_strategies[-1].name):
            low_extra_start -= 1

        for i in range(low_extra_start+1, len(new_lows)):
            old_game.low_strategies.append(new_lows[i])
            n = len(old_game.high_strategies)
            old_game.add_low_cost_row(np.zeros(n), np.zeros(n))
            for j in range(len(old_game.high_strategies)):
                old_game.update_matrix_entry((len(old_game.low_strategies)-1), j)

        high_trained_i = 0
        while high_trained_i < len(old_game.high_strategies) and old_game.high_strategies[high_trained_i].type != StrategyType.sb3_model:
            high_trained_i += 1

        high_new_trained_i = 0
        while high_new_trained_i < len(new_highs) and new_highs[high_new_trained_i].type != StrategyType.sb3_model:
            high_new_trained_i += 1

        high_extra_start = len(old_game.high_strategies) - high_trained_i-1 + high_new_trained_i
        # low_extra_start shows index of 1st low strategy that is not in the previous saved game, all strategies after this index should be added to the game
        while high_extra_start >= 0 and (new_highs[high_extra_start].name != old_game.high_strategies[-1].name):
            high_extra_start -= 1

        for i in range(high_extra_start+1, len(new_highs)):
            old_game.high_strategies.append(new_highs[i])
            n = len(old_game.low_strategies)
            old_game.add_high_cost_col(np.zeros(n), np.zeros(n))
            for j in range(len(old_game.low_strategies)):
                old_game.update_matrix_entry(j, (len(old_game.high_strategies)-1))

        return old_game


def str_to_envclass(s):
    if s == envs.ConPricingGame.__name__:
        return envs.ConPricingGame
    elif s == envs.DisPricingGame.__name__:
        return envs.DisPricingGame
    else:
        return None


TrainInputRow = namedtuple(
    "TrainInputRow", "id, seed, job_name, env, base_agent, alg, alg_params, adv_mixed_strategy,target_payoff, db,num_ep_coef,equi_id")


def new_train(inputs):
    """ traines one agent against the adversary, if the expected payoff of new agent is greater than expected payoff of NE, returns acceptable=true and the new strategy and payoff to be added to the strategies and matrix. 
    inputs = (id, seed, job_name, env, base_agent, alg, alg_params, adv_mixed_strategy,target_payoff, db, num_ep_coef, equi_id) """
    id, seed, job_name, env, base_agent, alg, alg_params, adv_mixed_strategy, target_payoff, db, num_ep_coef, equi_id = inputs

    gl.initialize()

    model_name = f"{job_name}-{str(seed)}"
    models_dir = f"{gl.MODELS_DIR}/{model_name}"
    log_dir = f"{gl.LOG_DIR}/{model_name}"
    
    alg_ep_coef=3 if name_of(alg)==name_of(PPO) else 1

    acceptable = False
    if base_agent is None:
        number_episodes = int(gl.N_EPISODES_BASE*num_ep_coef)*alg_ep_coef
        model = alg('MlpPolicy', env, tensorboard_log=log_dir, seed=seed, gamma=gl.GAMMA, **alg_params)

    else:
        number_episodes = int(gl.N_EPISODES_LOAD*num_ep_coef)*alg_ep_coef
        base_agent_dir = f"{gl.MODELS_DIR}/{base_agent}"
        model = alg.load(base_agent_dir, env, tensorboard_log=log_dir, gamma=gl.GAMMA, seed=seed, **alg_params)

    start = time.time()
    # for i in range(gl.NUM_MODEL_SAVE):
    # tmp = (number_episodes/gl.NUM_MODEL_SAVE)
    # model.learn(total_timesteps=tmp, reset_num_timesteps=False,
    #             tb_log_name=model_name)
    # model.save(os.path.join(models_dir, str(tmp*(i+1))))

    # https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#id3
    # check to save and load replay buffer as well
    model.learn(total_timesteps=number_episodes, tb_log_name=model_name)
    model.save(models_dir)
    running_time = time.time() - start

    model_strategy = Strategy(strategy_type=StrategyType.sb3_model,
                              model_or_func=alg, name=model_name, action_step=env.action_step, memory=env.memory)

    iter_rows = []
    agent_payoffs = np.zeros(len(adv_mixed_strategy.strategies))
    adv_payoffs = np.zeros(len(adv_mixed_strategy.strategies))
    expected_payoff = 0
    expected_payoff_std = 0
    for strategy_index in range(len(adv_mixed_strategy.strategies)):
        if adv_mixed_strategy.strategy_probs[strategy_index] > 0:
            payoffs = []
            for _ in range(gl.NUM_STOCHASTIC_ITER):
                # returns = algorithm.play_trained_agent(adversary=(
                #     (adv_mixed_strategy._strategies[strategy_index]).to_mixed_strategy()), iterNum=gl.num_stochastic_iter)
                payoffs.append(model_strategy.play_against(
                    env=env, adversary=adv_mixed_strategy.strategies[strategy_index]))

                # adv, agent_return, adv_return, rewards, adv_rewards, actions, prices, adv_prices, demands, adv_demands
                iter_row = Iter_row(adv=env.adversary_strategy.name, agent_return=sum(env.profit[0]), adv_return=sum(env.profit[1]), rewards=str(
                    env.profit[0]), adv_rewards=str(env.profit[1]), actions=str(env.actions), prices=str(env.prices[0]), adv_prices=str(env.prices[1]), demands=str(env.demand_potential[0]), adv_demands=str(env.demand_potential[1]))

                iter_rows.append(iter_row)

            std_payoffs = np.array(payoffs).std(axis=0)
            mean_payoffs = np.array(payoffs).mean(axis=0)

            agent_payoffs[strategy_index] = mean_payoffs[0]
            adv_payoffs[strategy_index] = mean_payoffs[1]
            expected_payoff += (agent_payoffs[strategy_index]) * (adv_mixed_strategy.strategy_probs[strategy_index])
            expected_payoff_std += (std_payoffs[0]) * (adv_mixed_strategy.strategy_probs[strategy_index])

    acceptable = (expected_payoff > target_payoff)

    #  "name, base_agent,  n_ep, cost, mixed_adv_txt, alg, seed, num_process, running_time, return_std,  expected_payoff, payoff_treshhold, added"
    agent_id = db.insert_new_agent(db.AgentRow(model_name, base_agent, number_episodes, env.costs[0], str(
        adv_mixed_strategy),   name_of(alg), seed, 1, running_time, expected_payoff_std, expected_payoff, target_payoff, acceptable, equi_id))

    # num_processes=1 because it just uses one process in training this agent

    if acceptable:
        tuple_list=[]
        for row in iter_rows:
            # db.insert_new_iteration(agent_id, row.adv, row.agent_return, row.adv_return, row.rewards,
            #                         row.adv_rewards, row.actions, row.prices, row.adv_prices, row.demands, row.adv_demands)
            if len(tuple_list)<gl.DB_ITER_LIMIT:
            # just insert the first iter limit rows to db (db can get too big)
                tuple_list.append(db.IterRow(agent_id, row.adv, row.agent_return, row.adv_return, row.rewards,
                                        row.adv_rewards, row.actions, row.prices, row.adv_prices, row.demands, row.adv_demands))
        db.insert_many_new_iters(tuple_list)

    return (id, acceptable, model_strategy.name, agent_payoffs, adv_payoffs, expected_payoff)


# def train(inputs):
#     """ traines one agent against the adversary, if the expected payoff of new agent is greater than expected payoff of NE, returns acceptable=true and the new strategy and payoff to be added to the strategies and matrix. inputs = (id, seed, job_name, env, base_agent, alg, adv_mixed_strategy, lr, target_payoff, db) """
#     id, seed, job_name, env, base_agent, alg, adv_mixed_strategy, lr, target_payoff, db = inputs

#     gl.initialize()

#     model_name = f"{job_name}-{str(seed)}"
#     models_dir = f"{gl.MODELS_DIR}/{model_name}"
#     log_dir = f"{gl.LOG_DIR}/{model_name}"

#     acceptable = False
#     if base_agent is None:
#         number_episodes = gl.N_EPISODES_BASE * (1 + gl.EPISODE_INCREASE_PORTION * (adv_mixed_strategy.support_size-1))
#         if alg is SAC:
#             model = alg('MlpPolicy', env, learning_rate=lr,
#                         verbose=0, tensorboard_log=log_dir, gamma=gl.GAMMA, target_entropy=0, seed=seed)
#         else:
#             model = alg('MlpPolicy', env, learning_rate=lr,
#                         verbose=0, tensorboard_log=log_dir, gamma=gl.GAMMA, seed=seed)
#     else:
#         number_episodes = gl.N_EPISODES_LOAD * (1 + gl.EPISODE_INCREASE_PORTION * (adv_mixed_strategy.support_size-1))
#         base_agent_dir = f"{gl.MODELS_DIR}/{base_agent}"
#         if alg is SAC:
#             model = alg.load(base_agent_dir, env, learning_rate=lr,
#                              verbose=0, tensorboard_log=log_dir, gamma=gl.GAMMA, target_entropy=0)
#         else:
#             model = alg.load(base_agent_dir, env, learning_rate=lr,
#                              verbose=0, tensorboard_log=log_dir, gamma=gl.GAMMA)
#     start = time.time()
#     # for i in range(gl.NUM_MODEL_SAVE):
#     # tmp = (number_episodes/gl.NUM_MODEL_SAVE)
#     # model.learn(total_timesteps=tmp, reset_num_timesteps=False,
#     #             tb_log_name=model_name)
#     # model.save(os.path.join(models_dir, str(tmp*(i+1))))

#     # https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#id3
#     # check to save and load replay buffer as well
#     model.learn(total_timesteps=number_episodes, tb_log_name=model_name)
#     model.save(models_dir)
#     running_time = time.time() - start

#     model_strategy = Strategy(strategy_type=StrategyType.sb3_model,
#                               model_or_func=alg, name=model_name, action_step=env.action_step, memory=env.memory)

#     iter_rows = []
#     agent_payoffs = np.zeros(len(adv_mixed_strategy.strategies))
#     adv_payoffs = np.zeros(len(adv_mixed_strategy.strategies))
#     expected_payoff = 0
#     for strategy_index in range(len(adv_mixed_strategy.strategies)):
#         if adv_mixed_strategy.strategy_probs[strategy_index] > 0:
#             payoffs = []
#             for _ in range(gl.NUM_STOCHASTIC_ITER):
#                 # returns = algorithm.play_trained_agent(adversary=(
#                 #     (adv_mixed_strategy._strategies[strategy_index]).to_mixed_strategy()), iterNum=gl.num_stochastic_iter)
#                 payoffs.append(model_strategy.play_against(
#                     env=env, adversary=adv_mixed_strategy.strategies[strategy_index]))

#                 # adv, agent_return, adv_return, rewards, adv_rewards, actions, prices, adv_prices, demands, adv_demands
#                 iter_row = Iter_row(adv=env.adversary_strategy.name, agent_return=sum(env.profit[0]), adv_return=sum(env.profit[1]), rewards=str(
#                     env.profit[0]), adv_rewards=str(env.profit[1]), actions=str(env.actions), prices=str(env.prices[0]), adv_prices=str(env.prices[1]), demands=str(env.demand_potential[0]), adv_demands=str(env.demand_potential[1]))

#                 iter_rows.append(iter_row)

#             mean_payoffs = np.array(payoffs).mean(axis=0)

#             agent_payoffs[strategy_index] = mean_payoffs[0]
#             adv_payoffs[strategy_index] = mean_payoffs[1]
#             expected_payoff += (agent_payoffs[strategy_index]) * \
#                 (adv_mixed_strategy.strategy_probs[strategy_index])

#     acceptable = (expected_payoff > target_payoff)
#     # agent_id=db.insert_new_agent(model_name,number_episodes,costs[0], str(adv_mixed_strategy), expected_payoff,target_payoff, lr,memory, acceptable, pricing_game.action_step, seed,num_procs,running_time)
#     agent_id = db.insert_new_agent(db.AgentRow(model_name, base_agent, number_episodes, env.costs[0], str(
#         adv_mixed_strategy), expected_payoff, target_payoff,  str(alg), lr, env.memory, acceptable, env.action_step, seed, 1, running_time))
#     # num_processes=1 because it just uses one process in training this agent

#     if acceptable:
#         for row in iter_rows:
#             db.insert_new_iteration(agent_id, row.adv, row.agent_return, row.adv_return, row.rewards,
#                                     row.adv_rewards, row.actions, row.prices, row.adv_prices, row.demands, row.adv_demands)

#     return (id, acceptable, model_strategy.name, agent_payoffs, adv_payoffs, expected_payoff)


def train_processes(db, env_class, costs, adv_mixed_strategy, target_payoff, num_procs, alg, lr, memory):
    """
    trains multiple agents with multiprocessing against mixed_adversary. 
    """
    inputs = []
    seed = int(time.time())
    adv_strt = adv_mixed_strategy.copy_unload()

    base_agents = find_base_agents(db=db, alg=alg, memory=memory, cost=costs[0], mix_strt=adv_strt, size=num_procs)

    for p in range(num_procs):
        env = env_class(tuple_costs=costs, adversary_mixed_strategy=adv_strt, memory=memory)
        input_proc = (p, seed+p, job_name, env, base_agents[p], alg, adv_strt, lr, target_payoff, db)
        inputs.append(input_proc)
    results = []

    pool = mp.Pool(processes=num_procs)

    outputs = pool.imap_unordered(train, inputs)
    for output in outputs:
        id, acceptable, strategy_name, agent_payoffs, adv_payoffs, expected_payoff = output
    # id,acceptable,strategy_name,agent_payoffs, adv_payoffs, expected_payoff= train(inputs[0])
        pricing_game = env_class(tuple_costs=costs, adversary_mixed_strategy=adv_strt, memory=memory)
        model_strategy = Strategy(strategy_type=StrategyType.sb3_model,
                                  model_or_func=alg, name=strategy_name, action_step=pricing_game.action_step, memory=memory)
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


def equi_sort_social_welfare(equis: List[bimatrix.Equi]) -> List[bimatrix.Equi]:
    for i in range(len(equis)):
        for j in range(i+1, len(equis)):
            if (equis[i].row_payoff + equis[i].col_payoff) < (equis[j].row_payoff + equis[j].col_payoff):
                equis[i], equis[j] = equis[j], equis[i]
    return equis


def get_coop_equilibria(bimatrix_game: BimatrixGame, num_trace: int, db: DataBase) -> Dict[bimatrix.Equi, int]:
    """
    computes the equilibria of bimatrix game and selects most cooperative ones, also writes all of them in db. returns a dictionary of equis with key as equi and value as id in db.
    """
    all_equilibria = bimatrix_game.compute_equilibria(num_trace=num_trace)
    num_selected_equis = min(len(all_equilibria), gl.NUM_TRACE_EQUILIBRIA)
    # sort equis based on social welfare
    all_equilibria = equi_sort_social_welfare(all_equilibria)
    equi_ids = {}
    for i, equi in enumerate(all_equilibria):
        # self, game_size,freq, low_strategy_txt, high_strategy_txt, low_payoff, high_payoff, used,num_new_low, num_new_high)
        iid = db.insert_new_equi(game_size=bimatrix_game.size(), freq=(equi.found/float(num_trace)), low_strategy_txt=str(equi.row_probs), high_strategy_txt=str(
            equi.col_probs), low_payoff=equi.row_payoff, high_payoff=equi.col_payoff, used=0, num_new_low=0, num_new_high=0)
        if i < num_selected_equis:
            equi_ids[equi] = iid

    return equi_ids


def name_of(alg) -> str:
    """ returns name of training model"""
    if alg is SAC:
        return 'SAC'
    elif alg is PPO:
        return 'PPO'
    else:
        return str(alg)


def match_updated_size(main_game: BimatrixGame, old_adv_mixed_strt: MixedStrategy, own_cost: int, own_payoff: np.array, adv_payoff: np.array):
    """ when the mixed_adv strategies can be expanded by adding other strategies, this method adds zeros to payoff lists to make them same size as rows or cols in main_game and also, updates the adv_mixed_strategy by adding prob 0 for new ones.
    Attention: payoffs should be computed later
    returns new_adv_mixed_strt, new_own_payoff, new_adv_payoff that are ready to be added as rows or columns to the main_game"""

    if own_cost == gl.LOW_COST:  # the adv is high_cost
        # payoff_equal = (len(main_game.high_strategies) == len(own_payoff))
        if len(main_game.high_strategies) == len(old_adv_mixed_strt.strategy_probs):
            pass  # everything good
        elif (extra_strts := len(main_game.high_strategies)-len(old_adv_mixed_strt.strategy_probs)) > 0:
            old_adv_mixed_strt.strategies = main_game.high_strategies.copy()
            for _ in range(extra_strts):
                old_adv_mixed_strt.strategy_probs.append(0)
        else:
            prt("Error in match updated size, more strategies in adv_strategy than the game_high_strts")
            raise ValueError
    elif own_cost == gl.HIGH_COST:  # the adv is low_cost
        if len(main_game.low_strategies) == len(old_adv_mixed_strt.strategy_probs):
            pass
        elif (extra_strts := len(main_game.low_strategies)-len(old_adv_mixed_strt.strategy_probs)) > 0:
            old_adv_mixed_strt.strategies = main_game.low_strategies.copy()
            for _ in range(extra_strts):
                old_adv_mixed_strt.strategy_probs.append(0)
        else:
            prt("Error in match updated size, more strategies in adv_strategy than the game_low_strts")
            raise ValueError
    if (extra:=len(old_adv_mixed_strt.strategies)-len(own_payoff))>0:
        own_payoff=np.append(own_payoff,[0]*(extra))
        adv_payoff=np.append(adv_payoff,[0]*(extra))
    if len(own_payoff)!= len(old_adv_mixed_strt.strategies):
        print("Error in match updated size ")
    return old_adv_mixed_strt, own_payoff, adv_payoff
# alg_classes = {
#     'SAC': SAC,
#     'PPO': PPO
# }

# def remove_ineffective_agents(bimatrix_game:BimatrixGame, db:DataBase):
#     """ accrosing to average_probs table, removes that have never been part of an equilibrium, and they have been in the game for a while, also updates the added column of agents to -1, and the pickle file and game should be saved again"""
#     query=f"select * from {db.PROBS_TABLE} order by id desc limit 1;"
#     df= db.dataframe_select(query=query)
#     if df.empty:
#         return bimatrix_game
#     row = df.iloc[0]
    
    
#     probs = json.loads(row['strategy_probs'])
#     # ##########to be done 
# class ProcessInd(Enum):
#     SAClow = 0
#     PPOlow = 1
#     SAChigh = 2
#     PPOhigh = 3


# class StartMode(Enum):
#     """ double oracle game strting point, start from myopic-const-guess or from a random model or different strategies similar to guess"""
#     myopicConstGuess = 0
#     random = 1
#     multiGuess = 2
#     allVsSpe=3
    
# def len_initial_game(start_mode:StartMode)->int:
#     if start_mode==StartMode.myopicConstGuess or start_mode==StartMode.multiGuess:
#         return 3
#     elif start_mode==StartMode.random:
#         return 1
#     elif start_mode==StartMode.allVsSpe:
#         return 7
#     else:
#         raise ValueError("len of start_mode not implemented!")
        
    
# def remove_ineffective_agents(bimatrix_game:BimatrixGame, db:DataBase, start_mode:StartMode):
#     """ according to average_probs table, removes that have never been part of an equilibrium, and they have been in the game for a while, also updates the added column of agents to -1, and the pickle file and game should be saved again"""
#     query=f"select * from {db.PROBS_TABLE} order by id desc limit 1;"
#     df= db.dataframe_select(query=query)
#     if df.empty:
#         return bimatrix_game
#     row = df.iloc[0]
    
    
#     probs = json.loads(row['strategy_probs'])
#     start_ind=len_initial_game(start_mode=start_mode)
#     #########to be done


# def initial_matrix(env_class, start_mode):
#     """ returns double oracle game with strategies from last stopping point but the matrix and strategies are not loaded , creates the base matrix and adds the trained strategies"""
#     if start_mode == StartMode.myopicConstGuess:

#         strt1 = cl.Strategy(
#             cl.StrategyType.static, model_or_func=cl.myopic, name="myopic")
#         strt2 = cl.Strategy(
#             cl.StrategyType.static, model_or_func=cl.const, name="const", first_price=132)
#         strt3 = cl.Strategy(
#             cl.StrategyType.static, model_or_func=cl.guess, name="guess", first_price=132)

#         init_low = [strt1, strt2, strt3]
#         init_high = [strt1, strt2, strt3]
#     elif start_mode == StartMode.allVsSpe:
#         strt0 = cl.Strategy(
#             cl.StrategyType.static, model_or_func=cl.spe, name="spe", first_price=132)
#         strt1 = cl.Strategy(
#             cl.StrategyType.static, model_or_func=cl.myopic, name="myopic")
#         strt2 = cl.Strategy(
#             cl.StrategyType.static, model_or_func=cl.const, name="const", first_price=132)
#         strt3 = cl.Strategy(
#             cl.StrategyType.static, model_or_func=cl.imit, name="imit", first_price=132)
        
#         strt4 = cl.Strategy(
#             cl.StrategyType.static, model_or_func=cl.guess, name="normal_guess",first_price=132)
#         strt5 = cl.Strategy(
#             cl.StrategyType.static, model_or_func=cl.guess2, name="coop_guess", first_price=132)
#         strt6 = cl.Strategy(
#             cl.StrategyType.static, model_or_func=cl.guess3, name="compete_guess", first_price=132)

#         init_low = [strt0,strt1, strt2, strt3,strt4,strt5,strt6]
#         init_high = [strt0,strt1, strt2, strt3,strt4,strt5,strt6]
#     elif start_mode == StartMode.random:
#         model_name = f"rndstart_{job_name}"
#         log_dir = f"{gl.LOG_DIR}/{model_name}"
#         model_dir = f"{gl.MODELS_DIR}/{model_name}"
#         if not os.path.exists(f"{model_dir}.zip"):
#             # tuple_costs and others are none just to make sure no play is happening here
#             train_env = env_class(tuple_costs=None, adversary_mixed_strategy=None, memory=gl.MEMORY)
#             model = SAC('MlpPolicy', train_env,
#                         verbose=0, tensorboard_log=log_dir, gamma=gl.GAMMA, target_entropy=0)
#             model.save(model_dir)

#         strt_rnd = cl.Strategy(strategy_type=cl.StrategyType.sb3_model,
#                                model_or_func=SAC, name=model_name, action_step=None, memory=gl.MEMORY)

#         init_low = [strt_rnd]
#         init_high = [strt_rnd]
#     elif start_mode==StartMode.multiGuess:
#         strt1 = cl.Strategy(
#             cl.StrategyType.static, model_or_func=cl.guess, name="normal_guess",first_price=132)
#         strt2 = cl.Strategy(
#             cl.StrategyType.static, model_or_func=cl.guess2, name="coop_guess", first_price=132)
#         strt3 = cl.Strategy(
#             cl.StrategyType.static, model_or_func=cl.guess3, name="compete_guess", first_price=132)

#         init_low = [strt1, strt2, strt3]
#         init_high = [strt1, strt2, strt3]
#     else:
#         raise("Error: initial_matrix mode not implemented!")
    

#     low_strts, high_strts = db.get_list_of_added_strategies(action_step=None, memory=gl.MEMORY)
#     return cl.BimatrixGame(
#         low_cost_strategies=init_low+low_strts, high_cost_strategies=init_high+high_strts, env_class=env_class)


# def get_proc_input(seed, proc_ind: ProcessInd, low_mixed_strt, high_mixed_strt, payoffs_low_high, job_name, env_class,num_ep_coef,equi_id,db) -> cl.TrainInputRow:
#     """
#     creates the input tuple for new_train method, to use in multiprocessing
#     """
#     # input=(id, seed, job_name, env, base_agent, alg, alg_params, adv_mixed_strategy,target_payoff, db)
#     if proc_ind == ProcessInd.PPOlow or proc_ind == ProcessInd.SAClow:
#         costs = [gl.LOW_COST, gl.HIGH_COST]
#         own_strt = low_mixed_strt.copy_unload()
#         adv_strt = high_mixed_strt.copy_unload()
#         payoff = payoffs_low_high[0]
#     elif proc_ind == ProcessInd.PPOhigh or proc_ind == ProcessInd.SAChigh:
#         costs = [gl.HIGH_COST, gl.LOW_COST]
#         own_strt = high_mixed_strt.copy_unload()
#         adv_strt = low_mixed_strt.copy_unload()
#         payoff = payoffs_low_high[1]

#     if proc_ind == ProcessInd.SAChigh or proc_ind == ProcessInd.SAClow:
#         alg = SAC
#     elif proc_ind == ProcessInd.PPOhigh or proc_ind == ProcessInd.PPOlow:
#         alg = PPO

#     iid = proc_ind.value
#     env = env_class(tuple_costs=costs, adversary_mixed_strategy=adv_strt, memory=gl.MEMORY)
#     base_agent = cl.find_base_agent(db, alg, costs[0], own_strt)
#     return cl.TrainInputRow(iid, seed+iid, job_name, env, base_agent, alg, gl.ALG_PARAMS[cl.name_of(alg)], adv_strt, payoff, db,num_ep_coef,equi_id)

