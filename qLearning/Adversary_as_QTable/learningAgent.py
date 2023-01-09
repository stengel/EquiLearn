# Ed, Kat, Galit
# ReinforceAlgorithm Class: Solver.

import numpy as np 

# printoptions: output limited to 2 digits after decimal point
np.set_printoptions(precision=2, suppress=False)


class LearningAlgorithm():
    """
        Model Solver.
    """
    def __init__(self, Model, Qtable, numberEpisodes, discountFactor) -> None:
        
        self.env = Model
        self.Qtable = Qtable.Q_table
        self.learning_Rate = Qtable.learning_Rate
        self.numberEpisodes = numberEpisodes
        self.gamma = discountFactor
        self.numStates=Qtable.num_States #Qtable dimen - number of states x number of actions
        if self.numStates % 2 ==0 :
            print('num of states should be odd')
        self.numActions=Qtable.num_Actions
        if self.numActions % 2 ==1 :
            print('num of actions should be even')
        self.lowestState = 200-(self.numStates - 1)/2  
        self.highestState = 200+(self.numStates - 1)/2 
        
        
    def resetQtable(self):
        self.Qtable, self.learning_Rate = self.Qtable.reset()
        
        """
        This function will eventually choose and adversary agent to play against
        and according to that return an action played by that adversary. 
        This could be chosen according to some existing equilibrium dist.
        """
    def chooseAdversaryAction(self, state):
        adversary = self.env.adversary
        lowestAdversaryState = 200 - (len(adversary) -1)/2
        adversaryStateIndex = int((400 - state) - lowestAdversaryState) # Declare as int for index reasons
        row = adversary[adversaryStateIndex]
        adversaryActionIndex = np.argmax(row)
        action = adversaryActionIndex + int((400-state) + self.env.costs[1])/2 - len(row) + 1
        # Highest action is the myopic action. Subtract length of row and add one to get to lowest action.
        return action 

    def StateInd(self, state):                 # computes state index in Qtable
        return int(state - self.lowestState) # Declare as int for index reasons
        #return min(int(state -(200-self.numStates/2)), self.numStates - 1)

    def ActionInd(self, monPrice, action):      # computes action index in Qtable
        return int(action - (monPrice - self.numActions + 1)) #monPrice should have index numActions - 1

    def alpha_n(self, n):                       # defines the Qlearning rate
        return self.learning_Rate[0]/(n+self.learning_Rate[1])
    """
    Now the Q-learning itself 
    """
    def  solver(self):
        
        self.resetQtable

        for episode in range(self.numberEpisodes):

            state = np.random.randint(self.lowestState, self.highestState + 1) 
            monPrice = int((state + self.env.costs[0]) / 2) 
            action = np.random.randint(monPrice-self.numActions+1, monPrice + 1)
            advAction = self.chooseAdversaryAction(state)
            reward = (state-action)*(action-self.env.costs[0])

            next_state = int(state+.5*(advAction-action))

            opt_value_next = max(self.Qtable[self.StateInd(next_state)])

            # updating the Qtable
            qvalue = (1-self.alpha_n(episode)) * \
                   self.Qtable[self.StateInd(state), self.ActionInd(monPrice, action)]\
                   + self.alpha_n(episode) * \
                   ((1-self.gamma)*reward + self.gamma*opt_value_next)
            self.Qtable[self.StateInd(state), self.ActionInd(monPrice, action)]=qvalue


