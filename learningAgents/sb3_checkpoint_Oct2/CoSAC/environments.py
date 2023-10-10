
from enum import Enum
import numpy as np 
import gymnasium as gym
from gymnasium import spaces
import globals as gl
import classes as cl
class ConPricingGame(gym.Env):


    def __init__(self,tuple_costs, adversary_mixed_strategy):
        super().__init__()
        gl.initialize()
        
        # Actions that we can take: From 0 to 49 below the myopic price
        # self.action_space = spaces.Discrete(50)
        self.action_step=None

        self.total_demand = gl.TOTAL_DEMAND
        self.costs = tuple_costs
        self.T = gl.TOTAL_STAGES
        self.demand_potential = None # two lists for the two players
        self.prices = None # prices over rounds
        self.profit = None  # profit in each round
        self.stage = None
        self.done = False
        
        self.adversary_mixed_strategy = adversary_mixed_strategy
        self.state_adv_history = gl.NUM_ADV_HISTORY
        self.reward_division = gl.REWARDS_DIVISION_CONST

        self.action_space = spaces.Box(low=0, high=gl.CON_ACTIONS_RANGE, shape=(1,))
        
        # State space
        self.observation_space = spaces.Box(
            low=0, high=self.total_demand, shape=(3+gl.NUM_ADV_HISTORY,))


        

    def reset(self, seed = None, options = None):
        super().reset(seed=seed)

        self.resetGame()
        self.adversary_strategy = self.adversary_mixed_strategy.choose_strategy()
        # [stage, agent_ demand, agent_last_price, adversary_price_history]
        observation = self.get_state(stage=0)
        return observation, {}# reward, done, info can't be included

 

    def resetGame(self):
        """
        Method resets game memory: Demand Potential, prices, profits
        """
        self.episodesMemory = list()
        self.stage = 0
        self.done = False
        self.demand_potential = [
            [0]*(self.T+1), [0]*(self.T+1)]  # two lists for the two players
        self.prices = [[0]*self.T, [0]*self.T]  # prices over T rounds
        self.myopic_prices = [[0]*self.T, [0]*self.T]  # prices over T rounds
        self.profit = [[0]*self.T, [0]*self.T]  # profit in each of T rounds
        # initialize first round 0
        self.demand_potential[0][0] = self.demand_potential[1][0] = self.total_demand / 2
        self.actions=[0]*self.T
    
    def get_state(self, stage, player=0, adv_hist=None):

        num_adv_hist = adv_hist if (
            adv_hist is not None) else self.state_adv_history
        adv_history = []

        stage_part = [stage]
        # if self.state_onehot:
        #     stage_part=[0]*3
        #     if stage==0:
        #         stage_part[0]=1
        #     elif stage==self.T-1:
        #         stage_part[2]=1
        #     else:
        #         stage_part[1]=1


        if stage == 0:
            if (num_adv_hist > 0):
                adv_history = [0]*num_adv_hist
            observation = stage_part+[ self.demand_potential[player]
                           [self.stage], 0] + adv_history
        else:
            if (num_adv_hist > 0):
                adv_history = [0]*num_adv_hist
                j = num_adv_hist-1
                for i in range(stage-1, max(-1, stage-1-num_adv_hist), -1):
                    adv_history[j] = self.prices[1-player][i]
                    j -= 1

            observation = stage_part+ [self.demand_potential[player]
                           [self.stage], self.prices[player][stage-1]] + adv_history

        return np.array(observation)
        
        
    

    def step(self,action):
        self.actions[self.stage]=action[0]
        adversary_action  = self.adversary_strategy.play(
            env=self, player=1)
        self.update_game_variables( [self.myopic() - action[0], adversary_action] ) 

        done = (self.stage == self.T-1)

        reward = self.profit[0][self.stage]
        self.stage += 1

        info = {}

        return self.get_state(stage=self.stage), reward, done,False, info



    def update_game_variables(self, price_pair):
        """
        Updates Prices, Profit and Demand Potential Memory.
        Parameters. 
        price_pair: Pair of prices from the learning agent and adversary.
        """

        for player in [0,1]:
            price = price_pair[player]
            self.prices[player][self.stage] = price
            self.profit[player][self.stage] = (self.demand_potential[player][self.stage] - price) * (price - self.costs[player])/self.reward_division
            if self.stage < self.T - 1 :
                self.demand_potential[player][ self.stage + 1] = \
                    self.demand_potential[player][self.stage] + (price_pair[1-player] - price)/2
                


    def myopic(self, player = 0): 
        """
            Adversary follows Myopic strategy
        """
        return (self.demand_potential[player][self.stage]+self.costs[player])/2
        # return self.monopoly_price(player)    

       
    
    def render(self):
        pass

    def close(self):
        pass



class DisPricingGame(ConPricingGame):
    def __init__(self,tuple_costs, adversary_mixed_strategy):
        super().__init__(tuple_costs, adversary_mixed_strategy)

        self.action_step=gl.ACTION_STEP

        self.action_space = spaces.Discrete(gl.NUM_ACTIONS)
    
    def step(self, action):
        self.actions[self.stage]=action
        adversary_action  = self.adversary_strategy.play(
            env=self, player=1)
        self.update_game_variables( [self.myopic() - (action*self.action_step), adversary_action] ) 

        done = (self.stage == self.T-1)

        reward = self.profit[0][self.stage]
        self.stage += 1

        info = {}

        return self.get_state(stage=self.stage), reward, done,False, info
