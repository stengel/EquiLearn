from enum import Enum
import torch
import numpy as np 
from environmentModel import Model, AdversaryModes

totalDemand = 400
lowCost=57
highCost=71
totalStages=25

class MainGame():
    """
    strategies play against each other and dill the matrix of payoff, then the equilibria would be computed using Lemke algorithm
    """
    

    _strategies=[]
    _matrix=None

    def __init__(self) -> None:
        pass

    def reset_matrix(self):
        self._matrix=np.zeros((len(self._strategies),len(self._strategies),2))

    def fill_matrix(self):
        n=len(self._strategies)
        for low in range(n):
            for high in range(n):
                self.update_matrix_entry(low,high)
        pass
    def update_matrix_entry(self,lowIndex,highIndex):
        stratL=self._strategies[lowIndex].reset()
        stratH=self._strategies[highIndex].reset()
        self._matrix[lowIndex][highIndex]=stratL.play(stratH)
        pass

    def write_all_matrix(self):
        pass
    def compute_equilibria(self):
        pass

class Strategy():
    """
    strategies can be static or they can come from neural nets or Q-tables.

    """
    _type=None
    _name=None
    _neural_net=None
    _q_table=None
    _static_index=None

    def __init__(self,strategyType,name,staticIndex=None,neuralNet=None,qTable=None) -> None:
        """
        Based on the type of strategy, the index or neuralnet or q-table should be given as input
        """
        self._type=strategyType
        self._name=name
        self._neural_net=neuralNet
        self._static_index=staticIndex
        self._q_table=qTable
        

    def reset(self):
        pass

    def play(self, adversary):
        """ 
        self is low cost and gets adversary(high cost) strategy and they play
        output: tuple (payoff of low cost, payoff of high cost)
        """
        if self._type==StrategyType.static and adversary._type==StrategyType.static:
            self.play_static_static(adversary._static_index)
        elif self._type==StrategyType.static and adversary._type==StrategyType.neural_net:
            pass
        elif self._type==StrategyType.neural_net and adversary._type==StrategyType.static:
            pass

    def play_static_static(self,adversaryIndex):
        """
        self is the low cost player
        """

        selfIndex=self._static_index

        lProbs=torch.zeros(len(AdversaryModes))
        lProbs[selfIndex]=1

        hProbs=torch.zeros(len(AdversaryModes))
        hProbs[adversaryIndex]=1

        lGame= Model(totalDemand = 400, 
               tupleCosts = (highCost,lowCost),
              totalStages = 25, adversaryProbs=lProbs, advHistoryNum=0)
        hGame = Model(totalDemand = 400, 
               tupleCosts = (lowCost,highCost),
              totalStages = 25, adversaryProbs=hProbs, advHistoryNum=0)
        lGame.reset()
        hGame.reset()
        
        for i in range(totalStages):
            lAction= lGame.adversaryChoosePrice()
            hAction= hGame.adversaryChoosePrice()
            lGame.updatePricesProfitDemand([hAction,lAction])
            hGame.updatePricesProfitDemand([lAction,hAction])
            lGame.stage+=1
            hGame.stage+=1

            print("\nl: ", lAction, " h: ", hAction, "\nprofits: ", hGame.profit,"\ndemand: ", hGame.demandPotential,"\nprices:",hGame.prices)
        profits=np.array(hGame.profit)
        returns=profits.sum(axis=1)
        print(returns)

        pass



class StrategyType(Enum):
    static=0
    q_table=1
    neural_net=2
