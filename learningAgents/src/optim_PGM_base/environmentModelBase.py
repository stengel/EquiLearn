# Francisco, Sahar, Edward
# Contains Game Class and Model of the Game Class.

from enum import Enum
import torch
import torch.nn as nn
from torch.distributions import Categorical
import sys  # Not used?
import numpy as np  # numerical python
import globals as gl



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
        self.demandPotential = [
            [0]*(self.T+1), [0]*(self.T+1)]  # two lists for the two players
        self.prices = [[0]*self.T, [0]*self.T]  # prices over T rounds
        self.myopicPrices = [[0]*self.T, [0]*self.T]  # prices over T rounds
        self.profit = [[0]*self.T, [0]*self.T]  # profit in each of T rounds
        self.demandPotential[0][0] = self.totalDemand / \
            2  # initialize first round 0
        self.demandPotential[1][0] = self.totalDemand/2

        self.our_target_demand = (
            (self.totalDemand + self.costs[1]-self.costs[0])/2)  # target demand
        self.target_price = (self.our_target_demand+self.costs[0])/2

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
            self.myopicPrices[player][self.stage] = myopic(self, player)
            price = pricePair[player]
            self.prices[player][self.stage] = price
            self.profit[player][self.stage] = (
                self.demandPotential[player][self.stage] - price)*(price - self.costs[player])
            if self.stage < self.T-1:
                self.demandPotential[player][self.stage + 1] = \
                    self.demandPotential[player][self.stage] + \
                    (pricePair[1-player] - price)/2

    def myopic(self, player=0):
        """
            Adversary follows Myopic strategy
        """
        return monopolyPrice(demand=self.demandPotential[player][self.stage], cost=self.costs[player])


class Model(DemandPotentialGame):
    """
        Defines the Problem's Model. It is assumed a Markov Decision Process is defined. The class is a Child from the Demand Potential Game Class.
        The reason: Model is a conceptualization of the Game.
    """

    def __init__(self, totalDemand, tupleCosts, totalStages, advMixedStrategy, stateAdvHistory=0) -> None:
        super().__init__(totalDemand, tupleCosts, totalStages)
        """ adversary is a MixedStrategy"""
        self.rewardFunction = self.profits

        self.episodesMemory = list()
        self.done = False

        # number of previous adversary's action we consider in the state
        self.stateAdvHistory = stateAdvHistory
        self.advMixedStrategy = advMixedStrategy

    def reset(self):
        """
            Reset Model Instantiation. 
        """
        reward = 0
        self.stage = 0
        self.done = False
        self.resetGame()
        self.resetAdversary()
        return self.get_state(0), reward, self.done

    def resetAdversary(self):
        self.adversaryStrategy = self.advMixedStrategy.set_adversary_strategy()

    def adversaryChoosePrice(self):
        """
            Strategy followed by the adversary.
        """

        return self.adversaryStrategy.play(environment=self, player=1)

    def step(self, price):
        """
        Transition Function. 
        Parameters:
        - action: Price
        - state: list in the latest stage (stage ,Demand Potential, Agent's Price, Adversary's price hisotry)
        """

        adversaryPrice = self.adversaryChoosePrice()
        p = self.myopic()
        # myopicPrice = self.myopic()
        self.updatePricesProfitDemand(
            [price, adversaryPrice])

        done = (self.stage == self.T-1)

        reward = self.rewardFunction()
        self.stage += 1

        return self.get_state(self.stage), reward, done

    def get_state(self, stage, player=0, adv_hist=None):

        num_adv_hist = adv_hist if (
            adv_hist is not None) else self.stateAdvHistory

        return define_state(stage, self.totalDemand, self.T, self.costs[player], self.costs[1-player], self.prices[player], self.prices[1-player], self.demandPotential[player], num_adv_hist)

    


class StrategyType(Enum):
    static = 0
    neural_net = 1


class Strategy():
    """
    strategies can be static or they can come from neural nets. If NN, policy is nn.policy o.w. the static function
    """
    type = None
    env = None
    name = None
    nn = None
    nn_hist = None
    policy = None

    def __init__(self, strategyType, NNorFunc, name, firstPrice=132) -> None:
        """
        Based on the type of strategy, the neuralnet or the Strategy Function  should be given as input. FirstPrice just applies to static strategies
        """
        self.type = strategyType
        self.name = name
        # self._env = environment

        if strategyType == StrategyType.neural_net:
            self.nn = NNorFunc
            self.policy = NNorFunc.policy
            self.nn_hist = NNorFunc.adv_hist
        else:
            self.policy = NNorFunc
            self._firstPrice = firstPrice

    def reset(self):
        pass

    def play(self, environment, player=0):
        """
            Computes the action to be played in the environment, nn.step_action is the step size for pricing less than myopic
        """
        self.env = environment
        if self.type == StrategyType.neural_net:
            state = self.env.getState(
                self.env.stage, player, advHist=self.nn_hist)
            normState = self.env.normalizeState(state=state)
            probs = self.policy(normState)
            distAction = Categorical(probs)
            action = distAction.sample()
            return self.env.compute_price(action=action.item(), actionStep=self.nn.action_step, player=player)

        else:
            return self.policy(self.env, player, self._firstPrice)

    def play_against(self, env, adversary):
        """ 
        self is player 0 and adversary is layer 1. The environment should be specified. action_step for the neural netwroks should be set.
        output: tuple (payoff of low cost, payoff of high cost)
        """
        self.env = env

        state, reward, done = env.reset()
        while env.stage < (env.T):
            prices = [0, 0]
            prices[0], prices[1] = self.play(env, 0), adversary.play(env, 1)
            env.updatePricesProfitDemand(prices)
            env.stage += 1

        return [sum(env.profit[0]), sum(env.profit[1])]

    def to_mixed_strategy(self):
        """
        Returns a MixedStrategy, Pr(self)=1
        """
        mix = MixedStrategy(probablitiesArray=torch.ones(1),
                            strategiesList=[self])

        return mix


class MixedStrategy():
    _strategies = []
    _strategyProbs = None

    def __init__(self, strategiesList=[], probablitiesArray=None) -> None:
        self._strategies = strategiesList
        self._strategyProbs = probablitiesArray

    def set_adversary_strategy(self):
        if len(self._strategies) > 0:
            adversaryDist = Categorical(torch.tensor(self._strategyProbs))
            strategyInd = (adversaryDist.sample()).item()
            return self._strategies[strategyInd]
        else:
            print("adversary's strategy can not be set!")
            return None

    def __str__(self) -> str:
        s = ""
        for i in range(len(self._strategies)):
            if self._strategyProbs[i] > 0:
                s += f"{self._strategies[i]._name}-{self._strategyProbs[i]:.2f},"
        return s


def myopic(env, player, firstprice=0):
    """
        Adversary follows Myopic strategy
    """
    return env.myopic(player)


def const(env, player, firstprice):  # constant price strategy
    """
        Adversary follows Constant strategy
    """
    if env.stage == env.T-1:
        return env.myopic(player)
    return firstprice


def imit(env, player, firstprice):  # price imitator strategy
    if env.stage == 0:
        return firstprice
    if env.stage == env.T-1:
        return env.myopic(player)
    return env.prices[1-player][env.stage-1]


def fight(env, player, firstprice):  # simplified fighting strategy
    if env.stage == 0:
        return firstprice
    if env.stage == env.T-1:
        return env.myopic(player)
    # aspire = [ 207, 193 ] # aspiration level for demand potential
    aspire = [0, 0]
    for i in range(2):
        aspire[i] = (env.totalDemand-env.costs[player] +
                     env.costs[1-player])/2

    D = env.demandPotential[player][env.stage]
    Asp = aspire[player]
    if D >= Asp:  # keep price; DANGER: price will never rise
        return env.prices[player][env.stage-1]
    # adjust to get to aspiration level using previous
    # opponent price; own price has to be reduced by twice
    # the negative amount D - Asp to getenv.demandPotential to Asp
    P = env.prices[1-player][env.stage-1] + 2*(D - Asp)
    # never price to high because even 125 gives good profits
    # P = min(P, 125)
    aspire_price = (env.totalDemand+env.costs[0]+env.costs[1])/4
    P = min(P, int(0.95*aspire_price))

    return P


def fight_lb(env, player, firstprice):
    P = env.fight(player, firstprice)
    # never price less than production cost
    P = max(P, env.costs[player])
    return P

# sophisticated fighting strategy, compare fight()
# estimate *sales* of opponent as their target


def guess(env, player, firstprice):  # predictive fighting strategy
    if env.stage == 0:
        env.aspireDemand = [(env.totalDemand/2 + env.costs[1]-env.costs[0]),
                            (env.totalDemand/2 + env.costs[0]-env.costs[1])]  # aspiration level
        env.aspirePrice = (env.totalDemand+env.costs[0]+env.costs[1])/4
        # first guess opponent sales as in monopoly ( sale= demand-price)
        env.saleGuess = [env.aspireDemand[0]-env.aspirePrice,
                         env.aspireDemand[1]-env.aspirePrice]

        return firstprice

    if env.stage == env.T-1:
        return env.myopic(player)

    D = env.demandPotential[player][env.stage]
    Asp = env.aspireDemand[player]

    if D >= Asp:  # keep price, but go slightly towards monopoly if good
        pmono = env.myopic(player)
        pcurrent = env.prices[player][env.stage-1]
        if pcurrent > pmono:  # shouldn't happen
            return pmono
        elif pcurrent > pmono-7:  # no change
            return pcurrent
        # current low price at 60%, be accommodating towards "collusion"
        return .6 * pcurrent + .4 * (pmono-7)

    # guess current *opponent price* from previous sales
    prevsales = env.demandPotential[1 -
                                    player][env.stage-1] - env.prices[1-player][env.stage-1]
    # adjust with weight alpha from previous guess
    alpha = .5
    newsalesguess = alpha * env.saleGuess[player] + (1-alpha)*prevsales
    # update
    env.saleGuess[player] = newsalesguess
    guessoppPrice = env.totalDemand - D - newsalesguess
    P = guessoppPrice + 2*(D - Asp)

    if player == 0:
        P = min(P, 125)
    if player == 1:
        P = min(P, 130)
    return P


def define_state(stage, total_demand, total_stages, agent_cost, adv_cost, agent_prices, adv_prices, agent_demands, num_adv_hist):
    # [one-hote encoding of stage, our demand, our price, adversary's price history]

    stageEncode = [0]*total_stages
    if stage < total_stages:
        stageEncode[stage] = 1

    if stage == 0:
        state = stageEncode + \
            [total_demand/2,
                ((total_demand/2) + agent_cost)/2] + ([0]*num_adv_hist)

    else:
        # check last stageeee demand
        state = stageEncode+[agent_demands[stage], agent_prices[stage-1]]
        if (num_adv_hist > 0):
            adv_history = [0]*num_adv_hist
            j = num_adv_hist-1
            for i in range(stage-1, max(-1, stage-1-num_adv_hist), -1):
                adv_history[j] = adv_prices[i]
                j -= 1
            state += adv_history

    return torch.tensor(state, dtype=torch.float32)


def monopolyPrice(demand, cost):  # myopic monopoly price
    """
        Computes Monopoly prices.
    """
    return (demand + cost) / 2
    # return (self.demandPotential[player][self.stage] + self.costs[player])/2

