# Katerina, Ed and Galit (and Tommy!)
# Relates to the Q-Learning approach.
# Contains DemandPotentialGame Class and the Model of the DemandPotentialGame Class.

from enum import Enum
import numpy as np  # numerical python

# printoptions: output limited to 2 digits after decimal point
np.set_printoptions(precision=2, suppress=False)


class DemandPotentialGame():
    """
        Fully defines demand Potential Game. It contains game rules, memory and agents strategies.
    """

    def __init__(self, totalDemand, tupleCosts, totalStages) -> None:
        self.totalDemand = totalDemand
        self.costs = tupleCosts
        self.T = totalStages
        # first index is always player
        self.demandPotential = None  # two lists for the two players
        self.prices = None  # prices over T rounds
        self.profit = None  # profit in each of T rounds
        self.stage = None

    def resetGame(self):
        """
        Method resets game memory: Demand Potential, prices, profits
        """
        self.demandPotential = [[0] * self.T, [0] * self.T]  # two lists for the two players
        self.prices = [[0] * self.T, [0] * self.T]  # prices over T rounds
        self.profit = [[0] * self.T, [0] * self.T]  # profit in each of T rounds
        self.demandPotential[0][0] = self.totalDemand / 2  # initialise first round 0
        self.demandPotential[1][0] = self.totalDemand / 2

    def profits(self, player=0):
        """
        Computes profits. Player 0 is the learning agent.
        """
        return self.profit[player][self.stage]

    def updatePricesProfitDemand(self, pricePair):
        """
        Updates Prices, Profit and Demand Potential Memory.
        Parameters.
        pricePair: Pair of prices from the Learning agent and adversary.
        """

        for player in [0, 1]:
            price = pricePair[player]
            self.prices[player][self.stage] = price
            self.profit[player][self.stage] = (self.demandPotential[player][self.stage] - price) * (
                        price - self.costs[player])
            if self.stage < self.T - 1:
                self.demandPotential[player][self.stage + 1] = \
                    self.demandPotential[player][self.stage] + (pricePair[1 - player] - price) / 2

    def monopolyPrice(self, player, t):  # myopic monopoly price
        """
            Computes Monopoly prices.
        """
        return (self.demandPotential[player][self.stage] + self.costs[player]) / 2

    """
    The following adversary strategies have been changed from the policy gradient method to remove 
    the last period effect.
    """


class Model(DemandPotentialGame):
    """
        Defines the Problem's Model. It is assumed a Markov Decision Process is defined.
        The class is a Child from the Demand Potential Game Class.
        The reason: Model is a conceptualisation of the Game.
    """

    def __init__(self, totalDemand, tupleCosts, totalStages, initState, adversary) -> None:
        super().__init__(totalDemand, tupleCosts, totalStages)

        self.rewardFunction = self.profits
        self.initState = initState
        self.episodesMemory = list()
        self.done = False
        self.adversary = adversary



