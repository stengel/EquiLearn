# Ed, Galit and Katerina 
# Relates to the Q-Learning approach.
# Testing out the Q-table against a given opponent

import numpy as np  


class Test():
    

    def __init__(self, Model, Qtable, discountFactor) -> None:

        
        self.env = Model
        self.Qtable = Qtable.Q_table 
        self.numStates=Qtable.num_States
        self.numActions=Qtable.num_Actions
        self.discountFactor = discountFactor  
        self.episodeLength = Model.T
        
    
    def chooseAdver(self, state):
        return int((400 - state + self.env.costs[1])/2) 
    

    def totalPayoff(self):
        
        utility = 0
        actions = [0]*self.episodeLength
        state = 200
        delta = 1/self.discountFactor
        
        for i in range(self.episodeLength):
            delta = delta * self.discountFactor
            row = self.Qtable[int(state -(200-self.numStates/2))]
            action = np.argmax(row) + int((state + self.env.costs[0])/2) - self.numActions + 1
            actions[i] = action
            utility += (state-action)*(action-self.env.costs[0]) * delta
            advAction = self.chooseAdver(state)
            state = int(state+.5*(advAction-action))
        return utility, np.transpose(actions)

    
    def error(self):
        
        Qtable_error = np.zeros((self.numStates, self.numActions))
        lowestState = int(200-(self.numStates)/2)
        
        for stateIndex in range(self.numStates):
            for actionIndex in range(self.numActions):
                
                state = stateIndex + lowestState
                monopoly_price = int((state + self.env.costs[0])/2) 
                action = actionIndex + monopoly_price - self.numActions + 1

                reward = (state - action) * (action - self.env.costs[0])
                adv_action = self.chooseAdver(state)
                next_state = int(state + (adv_action - action)/2)


                next_state_index = next_state - lowestState
                opt_value_next = max(self.Qtable[next_state_index])
                new_value = (1-self.discountFactor)*reward + self.discountFactor * opt_value_next
                Qtable_error[stateIndex,actionIndex] = (new_value - self.Qtable[stateIndex,actionIndex])/new_value
        return Qtable_error
    
    
    

    
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
        for i in range(self.episodeLength):
            delta = delta * self.discountFactor
            agentPrice = int(actions[i])
            opponentPrice = int(self.myopic(self.env.costs[1],opponentDemand))
            totalPayoff += self.payoff(self.env.costs[0],agentDemand,agentPrice) * delta
            agentDemand = int(self.updateDemand(agentDemand, [agentPrice,opponentPrice]))
            opponentDemand = 400 - agentDemand
        return totalPayoff
            

    


                



        

