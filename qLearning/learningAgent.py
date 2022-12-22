# Ed, Kat, Galit
# ReinforceAlgorithm Class: Solver.

import numpy as np #repeated

# printoptions: output limited to 2 digits after decimal point
np.set_printoptions(precision=2, suppress=False)


class LearningAlgorithm():
    """
        Model Solver.
    """
    #def __init__(self, Model, neuralNet, numberIterations, numberEpisodes, discountFactor) -> None:
    def __init__(self, Model, Qtable, numberEpisodes, discountFactor) -> None:
        self.env = Model
        self.env.adversaryReturns = np.zeros(numberEpisodes)
        self.returns = np.zeros( numberEpisodes)
        self.numberEpisodes = numberEpisodes
        self.episodesMemory = list()
        self.gamma = discountFactor
        self.Qtable=Qtable
        self.numStates=len(Qtable) #Qtable dimen - number of states x number of actions
        if self.numStates % 2 ==1 :
            print('num of states should be even')
        self.numActions=len(Qtable[0])
        if self.numActions % 2 ==1 :
            print('num of actions should be even')

        self.lowestState = int(200-(self.numStates)/2)
        self.highestState = int(200+(self.numStates)/2 - 1)

        """
        This function will eventually choose and adversary agent to play against
        and according to that return an action played by that adversary. 
        This could be chosen according to some existing equilibruim dist.
        """
    def chooseAdver(self, state):
        return int((400 - state + self.env.costs[1])/2) + 1 #myopic

    def StateInd(self, state):                 # computes state index in Qtable
        return int(state -(200-self.numStates/2))
        #return min(int(state -(200-self.numStates/2)), self.numStates - 1)

    def ActionInd(self, monprice, action):      # computes action index in Qtable
        return int(action-(monprice-self.numActions+1))

    def alpha_n(self, n):                       # defines the Qlearning rate
        return 50000/(n+60000)
    """
    Now the Q-learning itself 
    """
    def  solver(self):
        for episode in range(self.numberEpisodes):

            state = np.random.randint(self.lowestState, self.highestState + 1)

            monprice = int((state + self.env.costs[0]) / 2) + 1
            action = np.random.randint(monprice-self.numActions+1, monprice+1)
            advAction = self.chooseAdver(state)
            reward = (state-action)*(action-self.env.costs[0])

            #print(state,action,monprice)

            next_state = int(state+.5*(advAction-action))

            #print(state, action, monprice, next_state)
            #print(self.StateInd(state), self.ActionInd(monprice, action), monprice, self.StateInd(next_state))
            #print(self.highestState)

            opt_value_next = max(self.Qtable[self.StateInd(next_state)])

            # updating the Qtable
            qvalue = (1-self.alpha_n(episode)) * \
                   self.Qtable[self.StateInd(state), self.ActionInd(monprice, action)]\
                   + self.alpha_n(episode) * \
                   ((1-self.gamma)*reward + self.gamma*opt_value_next)
            self.Qtable[self.StateInd(state), self.ActionInd(monprice, action)]=qvalue


