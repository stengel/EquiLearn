from duopoly_game import DuopolyGame
from hard_coded_agents import Guess

import numpy as np
import matplotlib.pyplot as plt

low_cost = Guess({
    "start_price": 125,
    "aspiration_level": 207,
    "init_op_sales_guess": 61,
    "max_price": 125,
    "step_size": 7,
    "alpha": 0.5
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

l = 120
s = []
p = []
high_cost.agent.parameters["start_price"] = l
for i in range(2000):
    s.append(l + i/100)
    high_cost.agent.parameters["start_price"] += 0.01
    _, profit = game.run(low_cost, high_cost)
    p.append(profit)

plt.plot(s, p)
plt.show()