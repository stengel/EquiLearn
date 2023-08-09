
# import torch
import numpy as np  # numerical python
import gym
from gym import spaces
import globals as gl
import classes
from collections import deque

# printoptions: output limited to 2 digits after decimal point
np.set_printoptions(precision=2, suppress=False)


class PricingGame(gym.Env):
    """
        Fully defines PricingGame. It contains game rules, memory and agents strategies.
    """

    def __init__(self, tuple_costs, adversary_mixed_strategy, state_onehot=False):
        super(PricingGame, self).__init__()

        self.ONEHOT_LENGTH = 3

        # first index is always the agent
        self.costs = tuple_costs
        self.adversary_mixed_strategy = adversary_mixed_strategy
        self.state_onehot = state_onehot

        self.total_demand = gl.TOTAL_DEMAND
        self.action_step = gl.ACTION_STEP

        self.T = gl.TOTAL_STAGES
        # number of previous adversary's prices we consider in the state
        self.state_adv_history = gl.NUM_ADV_HISTORY
        self.reward_division = gl.REWARDS_DIVISION_CONST

        self.action_space = spaces.Discrete(gl.NUM_ACTIONS)

        state_shape = (self.ONEHOT_LENGTH if self.state_onehot else 1) + \
            2 + self.state_adv_history
        self.observation_space = spaces.Box(
            low=0, high=self.total_demand, shape=(state_shape,))

    def step(self, action):
        adversaryPrice = self.adversary_strategy.play(
            environment=self, player=1)

        self.update_game_variables(
            [self.myopic()-(action * self.action_step), adversaryPrice])

        done = (self.stage == self.T-1)

        reward = self.profit[0][self.stage]
        self.stage += 1

        info = {}

        return self.get_state(stage=self.stage), reward, done, info

    def reset(self,seed=None):
        

        self.resetGame()
        self.resetAdversary()
        # [stage, agent_ demand, agent_last_price, adversary_price_history]
        observation = self.get_state(stage=0)
        return observation  # reward, done, info can't be included

    # def render(self, mode='human'):
    # 	...
    # def close (self):
    # 	...

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

        # self.our_target_demand = (
        #     (self.totalDemand + self.costs[1]-self.costs[0])/2)  # target demand
        # self.target_price = (self.our_target_demand+self.costs[0])/2

    def resetAdversary(self):
        self.adversary_strategy = self.adversary_mixed_strategy.set_adversary_strategy()

    # def get_adversary_price(self):
    #     """
    #         Strategy followed by the adversary.
    #     """

    #     return self.adversary_strategy.play(environment=self, player=1)

    # def reward_function(self, player=0):
    #     """
    #     Computes profits. Player 0 is the learning agent.
    #     """
    #     return self.profit[player][self.stage]

    def update_game_variables(self, price_pair):
        """
        Updates Prices, Profit and Demand Potential Memory.
        Parameters. 
        pricePair: Pair of prices from the Learning agent and adversary.
        reward is divided by reward_division constant
        """

        for player in [0, 1]:
            self.myopic_prices[player][self.stage] = self.myopic(player)
            price = price_pair[player]
            self.prices[player][self.stage] = price
            self.profit[player][self.stage] = (
                self.demand_potential[player][self.stage] - price)*(price - self.costs[player])/self.reward_division
            if self.stage < self.T-1:
                self.demand_potential[player][self.stage + 1] = \
                    self.demand_potential[player][self.stage] + \
                    (price_pair[1-player] - price)/2

    def myopic(self, player=0):
        """
            Adversary follows Myopic strategy
        """
        return (self.demand_potential[player][self.stage]+self.costs[player])/2

    # def step(self, price):
    #     """
    #     Transition Function.
    #     Parameters:
    #     - action: Price
    #     - state: list in the latest stage (stage ,Demand Potential, Agent's Price, Adversary's price hisotry)
    #     """

    #     adversaryPrice = self.get_adversary_price()
    #     p = self.myopic()
    #     # myopicPrice = self.myopic()
    #     self.update_game_variables(
    #         [price, adversaryPrice])

    #     done = (self.stage == self.T-1)

    #     reward = self.rewardFunction()
    #     self.stage += 1

    #     return self.get_state(self.stage), reward, done

    def get_state(self, stage, player=0, adv_hist=None):

        num_adv_hist = adv_hist if (
            adv_hist is not None) else self.state_adv_history
        adv_history = []

        stage_part = [stage]
        if self.state_onehot:
            stage_part=[0]*3
            if stage==0:
                stage_part[0]=1
            elif stage==self.T-1:
                stage_part[2]=1
            else:
                stage_part[1]=1


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

        # return define_state(stage, self.total_demand, self.T, self.costs[player], self.prices[player], self.prices[1-player], self.demandPotential[player], num_adv_hist)


# def define_state(stage, total_demand, total_stages, agent_cost, agent_prices, adv_prices, agent_demands, num_adv_hist):
#     # [one-hote encoding of stage, our demand, our price, adversary's price history]

#     stageEncode = [0]*total_stages
#     if stage < total_stages:
#         stageEncode[stage] = 1

#     if stage == 0:
#         state = stageEncode + \
#             [total_demand/2,
#                 ((total_demand/2) + agent_cost)/2] + ([0]*num_adv_hist)

#     else:
#         # check last stageeee demand
#         state = stageEncode+[agent_demands[stage], agent_prices[stage-1]]
#         if (num_adv_hist > 0):
#             adv_history = [0]*num_adv_hist
#             j = num_adv_hist-1
#             for i in range(stage-1, max(-1, stage-1-num_adv_hist), -1):
#                 adv_history[j] = adv_prices[i]
#                 j -= 1
#             state += adv_history

#     return torch.tensor(state, dtype=torch.float32)
