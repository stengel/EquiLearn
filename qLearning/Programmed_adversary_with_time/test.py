# Ed, Galit and Katerina 
# Relates to the Q-Learning approach.
# Testing out the Q-table against a given opponent

import numpy as np
from environment import AdversaryModes


class Test():
    

    def __init__(self, Model, Qtable, discountFactor, adversaryProbs) -> None:

        
        self.env = Model
        self.Qtable = Qtable.Q_table 
        self.numStates=Qtable.num_States
        self.numActions=Qtable.num_Actions
        self.numStages=Qtable.num_Stages
        self.discountFactor = discountFactor  
        self.adversaryProbs = adversaryProbs
        self.adversary = None
        
    
    
    def setAdversary(self):
        options = list(range(len(self.adversaryProbs)))
        adversaryIndex = int(np.random.choice(options, 1, p= self.adversaryProbs))
        self.adversary = AdversaryModes(adversaryIndex)
        newProbs = [0]*len(self.adversaryProbs)
        newProbs[adversaryIndex] = 1
        self.env.adversaryProbs = newProbs
        
    
    def bestResponses(self):
        states = [0]* self.numStates
        bestResponses = [0]* self.numStates
        for i in range(self.numStates):
            demand = int((200-self.numStates/2)) + i
            states[i] = demand
            action = np.argmax(self.Qtable[i]) + int((demand + self.env.costs[0])/2) - self.numActions + 1
            bestResponses[i] = action
        return states, bestResponses
            

    def totalPayoff(self):
        
        self.setAdversary()
        
        delta = 1/self.discountFactor
        utility = 0
        advUtility = 0
        actions = [0]*self.numStages
        advActions = [0]*self.numStages
        demands = [0]*self.numStages
        stateVector, reward, done = self.env.reset()
        demand = stateVector[1]
        
        for stage in range(self.numStages):
            delta = delta * self.discountFactor
            demands[stage] = demand
            if (int(demand -(200-self.numStates/2)) > len(self.Qtable) -1):
                print("max action reached")
                demand = int((200-self.numStates/2) + len(self.Qtable) -1)
            if (int(demand -(200-self.numStates/2)) < 0):
                print("min action reached")
                demand = int((200-self.numStates/2))
            demand_index = int(demand -(200-self.numStates/2))
            action_index = np.argmax(self.Qtable[demand_index, :, stage])
            action = action_index + int((demand + self.env.costs[0])/2) - self.numActions + 1
            actions[stage] = action
            utility += (demand-action)*(action-self.env.costs[0]) * delta
            advDemand = 400 - demand
            stateVector, reward, done = self.env.step(stateVector, action)
            demand = stateVector[1]
            advActions[stage] = stateVector[2]
            advUtility += (advDemand-advActions[stage])*(advActions[stage]-self.env.costs[1]) * delta
        return utility, advUtility, np.transpose(actions), np.transpose(advActions), np.transpose(demands)

    
    def error(self):
        
        Qtable_error = np.zeros((self.numStates, self.numActions, self.numStages))
        lowestState = int(200-(self.numStates - 1)/2)
        
        for stateIndex in range(self.numStates):
            for actionIndex in range(self.numActions):
                for stage in range(self.numStages):
                    state = stateIndex + lowestState
                    monopoly_price = int((state + self.env.costs[0])/2) 
                    action = actionIndex + monopoly_price - self.numActions + 1
                    reward = (state - action) * (action - self.env.costs[0])
                    adv_action = int(self.chooseAdversaryAction(state))
                    next_state = int(state + (adv_action - action)/2)
                    next_state_index = next_state - lowestState
                    if(stage==self.numStages-1):
                        opt_value_next = 0
                    else:
                        opt_value_next = max(self.Qtable[next_state_index, : , stage+1])
                    new_value = (1-self.discountFactor)*reward + self.discountFactor * opt_value_next
                    if new_value != 0:
                        Qtable_error[stateIndex,actionIndex, stage] = (new_value - self.Qtable[stateIndex,actionIndex, stage])/new_value
                    if new_value == 0 and self.Qtable[stateIndex,actionIndex, stage] != 0:
                        print("Div by zero error") # This should not occur
        return Qtable_error
    
    
    def chooseAdversaryAction(self, state):
        return self.myopic(self.env.costs[1],400-state)
        

    
    def myopic(self, cost, demand):
        return (cost + demand)/2
    
    def payoff(self, cost, demand, price):
        return (demand - price)*(price - cost)
       
    def updateDemand(self, demand, pricePair):
        newDemand = demand + 0.5*(pricePair[1]- pricePair[0])
        return newDemand
    
    def utilityOfActions(self, actions):
        agentDemand = 200
        opponentDemand = 200
        totalPayoff = 0
        delta = 1/self.discountFactor
        for i in range(self.numStages):
            delta = delta * self.discountFactor
            agentPrice = int(actions[i])
            opponentPrice = int(self.myopic(self.env.costs[1],opponentDemand))
            totalPayoff += self.payoff(self.env.costs[0],agentDemand,agentPrice) * delta
            agentDemand = int(self.updateDemand(agentDemand, [agentPrice,opponentPrice]))
            opponentDemand = 400 - agentDemand
        return totalPayoff
            

    


                



        
