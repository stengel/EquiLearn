# PLOTS PROFIT LANDSCAPE FOR GUESS
# HELPS FOR VISUALISING OPTIMISATION
# MAKES CLEAR WHY FINITE DIFFERENCES FAILS TO CONVERGE

from duopoly_game import DuopolyGame
from hard_coded_agents import Guess

import numpy as np
import matplotlib.pyplot as plt

PARAMETER_1 = "step_size"
P1_MIN = 3
P1_MAX = 8

PARAMETER_2 = "alpha"
P2_MIN = 0
P2_MAX = 1

low_cost = Guess({
    "start_price": 125,
    "aspiration_level": 207,
    "init_op_sales_guess": 61,
    "max_price": 125,
    "step_size": 7,
    "alpha": 0.7
})

high_cost = Guess({
    "start_price": 130,
    "aspiration_level": 193,
    "init_op_sales_guess": 75,
    "max_price": 130,
    "step_size": 7,
    "alpha": 0.5
})

game = DuopolyGame(25)

def match(x, y):
    low_cost.parameters[PARAMETER_1] = x
    low_cost.parameters[PARAMETER_2] = y
    profit, _ = game.run(low_cost, high_cost)
    return profit

f = np.vectorize(match)

x = np.linspace(P1_MIN, P1_MAX, 50)
y = np.linspace(P2_MIN, P2_MAX, 50)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

plt.show()