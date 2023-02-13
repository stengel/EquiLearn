from learningAgent import LearningAlgorithm
from environment import Model, AdversaryModes
import numpy as np

np.random.seed(10)
agent_cost = 57
adv_cost = 71

game = Model(totalDemand = 400,
               tupleCosts = (agent_cost, adv_cost),
               totalStages = 25,
               initState = [400/2,0], adversaryMode=AdversaryModes.myopic)

num_Actions = 50
num_States = abs(adv_cost - agent_cost) + 2 * num_Actions + 2
gamma = 0.99


Qtable = np.zeros((num_States, num_Actions))
Qtable_error = np.zeros((num_States, num_Actions))

algorithm = LearningAlgorithm(game, Qtable, numberEpisodes = 10000, discountFactor = gamma)

algorithm.solver()

def bestAction(Qtable, state):
    row = Qtable[int(state -(200-self.numStates/2))]
    action = np.argmax(row) + (state + agent_cost)/2- num_Actions+1
    return action

    


# printing Qtable for excel
for s in range(num_States):
    for a in range(num_Actions):
        print("%.5f, " % Qtable[s,a], end="")
    print("")

# Checking the convergence of the Qtable

for s in range(num_States):
    for a in range(num_Actions):
        lowestState = int(200-(num_States)/2)
        highestState = int(200+(num_States)/2 - 1)
        state = s + lowestState

        monopoly_price = int((state + agent_cost)/2) + 1
        action = a + monopoly_price - num_Actions + 1

        reward = (state - action) * (action - agent_cost)
        adv_action = int((400 -state + adv_cost)/2) + 1
        next_state = int(state + (adv_action - action)/2)
        #print(state,monopoly_price,action,reward,adv_action, next_state)

        ns = next_state - lowestState
        opt_value_next = max(Qtable[ns])
        new_value = (1-gamma)*reward + gamma * opt_value_next
        Qtable_error[s,a] = (new_value - Qtable[s,a])/new_value

# Printing Qtable error
for s in range(num_States):
    for a in range(num_Actions):
        #print(Qtable[s,a],",", end="")
        print("%.5f, " % Qtable_error[s,a], end="")
    print("")

