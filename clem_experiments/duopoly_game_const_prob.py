import random #TODO use seed
import numpy as np
np.set_printoptions(precision=2, suppress=False)
from players import *

def single_round(p1, p2):
    p1_price_t = p1.price[-1]
    p2_price_t = p2.price[-1]

    p1_d_t = p1.d[-1]
    p2_d_t = p2.d[-1]

    p1_price_chosen = p1.choose_price(p2_price_t)
    p2_price_chosen = p2.choose_price(p1_price_t)

    p1.d.append(p1_d_t + 0.5*(p2_price_t - p1_price_t))
    p2.d.append(p2_d_t + 0.5*(p1_price_t - p2_price_t))

    p1.price.append(p1_price_chosen)
    p2.price.append(p2_price_chosen)

    p1_profit = (p1.d[-1] - p1.price[-1])*(p1.price[-1] - p1.cost)
    p2_profit = (p2.d[-1] - p2.price[-1])*(p2.price[-1] - p2.cost)
    return p1_profit, p2_profit

def match(p1, p2, cont_prob = 0.96): #w/ continuation probability
    T = 1
    p1.start()
    p2.start()
    p1_profits = [(p1.start_d - p1.start_price)*(p1.start_price - p1.cost)]
    p2_profits = [(p2.start_d - p2.start_price)*(p2.start_price - p2.cost)]
    while cont_prob > random.random(): #TODO: use seed
        p1_profit, p2_profit = single_round(p1, p2)
        p1_profits.append(p1_profit)
        p2_profits.append(p2_profit)
        T += 1
    return p1_profits, p2_profits, T

def tournament(p1_list, p2_list, cont_prob = 0.96, tries = 5):
    A = np.zeros((len(p1_list), len(p2_list)))
    B = np.zeros((len(p1_list), len(p2_list)))
    for i, p1 in enumerate(p1_list):
        for j, p2 in enumerate(p2_list):
            p1_profit_avgs = []
            p2_profit_avgs = []
            for _ in range(tries):
                p1_profits, p2_profits, T = match(p1, p2, cont_prob)
                p1_profit_avgs.append(sum(p1_profits) / T)
                p2_profit_avgs.append(sum(p2_profits) / T)
            A[i][j] = sum(p1_profit_avgs) / tries
            B[i][j] = sum(p2_profit_avgs) / tries
    return A, B

#TESTING Equivalence with play.py
p1_list = [
    Myopic(57, 200),
    Guess(57, 200, 125, 207, 61, 125),
    Const(57, 200, 125),
    Const(57, 200, 117),
    Const(57, 200, 114.2),
    Const(57, 200, 105),
    Const(57, 200, 100),
    Const(57, 200, 95),
    Imit(57, 200, 120),
    Imit(57, 200, 110),
    Fight(52, 200, 125, 207)
]

p2_list = [
    Myopic(71, 200),
    Guess(71, 200, 130, 193, 75, 130),
    Imit(71, 200, 131),
    Imit(71, 200, 114.2),
    Fight(71, 200, 130, 193)
]

A,B = tournament(p1_list, p2_list)

print(A)
print(B)