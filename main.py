
from learningAgent import ReinforceAlgorithm
from environment import Model, AdversaryModes
import numpy as np

agent_cost = 57
adv_cost = 71

game = Model(totalDemand = 400,
               tupleCosts = (agent_cost, adv_cost),
               totalStages = 25,
               initState = [400/2,0], adversaryMode=AdversaryModes.myopic)

num_Actions = 6
num_States = abs(adv_cost - agent_cost) + 2 * num_Actions + 2
print(num_States)

Qtable = np.zeros((num_States, num_Actions))

algorithm = ReinforceAlgorithm(game, Qtable, numberEpisodes = 1000000, discountFactor = 0.9)

algorithm.solver()

print(Qtable)
