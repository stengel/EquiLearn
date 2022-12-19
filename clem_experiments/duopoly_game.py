import random #TODO use seed
import numpy as np
np.set_printoptions(precision=2)
from hard_coded_agents import *

#TODO: create base game class in a way that abstracts all info from specific games. 
#       > probably would invovle creating "environment", "observation", and "action" types

class DuopolyGame(): #TODO: inheret from base game class
    def __init__(self, T):
        self.T = T
        self.costs = [57, 71]
        self.reset()

    def reset(self):
        self.prices = [[],[]]
        self.profits = [[],[]]
        self.ds = [[200], [200]]

    def single_round(self, p1, p2, t):
        next_price = [0,0]
        next_price[0] = p1.choose_price(self.ds[0][-1], self.costs[0], t)
        next_price[1] = p2.choose_price(self.ds[1][-1], self.costs[1], t)
        for player in [0,1]:
            self.prices[player].append(next_price[player])
            self.profits[player].append((self.ds[player][-1] - self.prices[player][-1]) * (self.prices[player][-1] - self.costs[player]))
        
        self.ds[0].append((self.ds[0][-1] + 0.5*(self.prices[1][-1] - self.prices[0][-1])))
        self.ds[1].append((self.ds[1][-1] + 0.5*(self.prices[0][-1] - self.prices[1][-1])))

    def run(self, p1, p2):
        p1.reset()
        p2.reset()
        self.reset()
        for t in range(self.T-1):
            self.single_round(p1,p2,t)

        #HACK END EFFECT:
        for player in [0,1]:
            self.prices[player].append((self.costs[player] + self.ds[player][-1])*0.5)
            self.profits[player].append((self.ds[player][-1] - self.prices[player][-1]) * (self.prices[player][-1] - self.costs[player]))
        #END HACK

        return sum(self.profits[0])/self.T, sum(self.profits[1])/self.T


class DuopolyGame_infy(): #TODO: inheret from base game class
    def __init__(self, cont_prob):
        self.cont_prob = cont_prob #contiuation probability
        self.costs = [57, 71]
        self.reset()

    def reset(self):
        self.prices = [[],[]]
        self.profits = [[],[]]
        self.ds = [[200], [200]]

    def single_round(self, p1, p2, t):
        next_price = [0,0]
        next_price[0] = p1.choose_price(self.ds[0][-1], self.costs[0], t)
        next_price[1] = p2.choose_price(self.ds[1][-1], self.costs[1], t)
        for player in [0,1]:
            self.prices[player].append(next_price[player])
            self.profits[player].append((self.ds[player][-1] - self.prices[player][-1]) * (self.prices[player][-1] - self.costs[player]))
        
        self.ds[0].append((self.ds[0][-1] + 0.5*(self.prices[1][-1] - self.prices[0][-1])))
        self.ds[1].append((self.ds[1][-1] + 0.5*(self.prices[0][-1] - self.prices[1][-1])))

    def run(self, p1, p2): #w/ continuation probability
        p1.reset()
        p2.reset()
        self.reset()

        self.single_round(p1, p2, 0)
        t = 1
        while self.cont_prob > random.random(): #TODO: use seed
            self.single_round(p1, p2, t)
            t += 1
        return sum(self.profits[0])/t, sum(self.profits[1])/t

def tournament_fixed_length(p1_list, p2_list, T):
    game = DuopolyGame(T)
    A = np.zeros((len(p1_list), len(p2_list)))
    B = np.zeros((len(p1_list), len(p2_list)))
    for i, p1 in enumerate(p1_list):
        for j, p2 in enumerate(p2_list):
            p1_profit, p2_profit = game.run(p1, p2)
            A[i][j] = p1_profit
            B[i][j] = p2_profit
    return A, B

def tournament_stochastic(p1_list, p2_list, cont_prob = 0.96, tries = 5):
    game = DuopolyGame_infy(cont_prob)
    A = np.zeros((len(p1_list), len(p2_list)))
    B = np.zeros((len(p1_list), len(p2_list)))
    for i, p1 in enumerate(p1_list):
        for j, p2 in enumerate(p2_list):
            p1_profit_avgs = []
            p2_profit_avgs = []
            for _ in range(tries):
                p1_profit, p2_profit = game.run(p1, p2)
                p1_profit_avgs.append(p1_profit)
                p2_profit_avgs.append(p2_profit)
            A[i][j] = sum(p1_profit_avgs) / tries
            B[i][j] = sum(p2_profit_avgs) / tries
    return A, B


#TESTING
if __name__ == "__main__":
    p1_list = [
        Myopic({}),
        Guess({
            "start_price": 125, 
            "aspiration_level": 207, 
            "init_op_sales_guess": 61, 
            "max_price": 125, 
            "step_size": 7, 
            "alpha": 0.5
        }),
        Const({"c": 125}),
        Const({"c": 117}),
        Const({"c": 114.2}),
        Const({"c": 105}),
        Const({"c": 100}),
        Const({"c": 95}),
        Imit({"start_price": 120}),
        Imit({"start_price": 110})
    ]

    p2_list = [
        Myopic({}),
        Guess({
            "start_price": 130, 
            "aspiration_level": 193, 
            "init_op_sales_guess": 75, 
            "max_price": 130, 
            "step_size": 7, 
            "alpha": 0.5
        }),
        Imit({"start_price": 131}),
        Imit({"start_price": 114.2})
    ]

    A1, B1 = tournament_fixed_length(p1_list, p2_list, 25)
    A2, B2 = tournament_stochastic(p1_list, p2_list)

    print(A1)
    print(A2)
    print(A1-A2)