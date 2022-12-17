import random #TODO use seed
import numpy as np
np.set_printoptions(precision=2, suppress=False)
from agent import *

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
        return self.profits[0], self.profits[1], t

def tournament(p1_list, p2_list, cont_prob = 0.96, tries = 5):
    game = DuopolyGame_infy(cont_prob)
    A = np.zeros((len(p1_list), len(p2_list)))
    B = np.zeros((len(p1_list), len(p2_list)))
    for i, p1 in enumerate(p1_list):
        for j, p2 in enumerate(p2_list):
            p1_profit_avgs = []
            p2_profit_avgs = []
            for _ in range(tries):
                p1_profits, p2_profits, T = game.run(p1, p2)
                p1_profit_avgs.append(sum(p1_profits) / T)
                p2_profit_avgs.append(sum(p2_profits) / T)
            A[i][j] = sum(p1_profit_avgs) / tries
            B[i][j] = sum(p2_profit_avgs) / tries
    return A, B

#TESTING Equivalence with play.py
p1_list = [
    Myopic(),
    Guess(125, 207, 61, 125),
    Const(125),
    Const(117),
    Const(114.2),
    Const(105),
    Const(100),
    Const(95),
    Imit(120),
    Imit(110)
]

p2_list = [
    Myopic(),
    Guess(130, 193, 75, 130),
    Imit(131),
    Imit(114.2)
]

A,B = tournament(p1_list, p2_list)

print(A)
print(B)