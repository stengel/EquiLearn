# Francisco, Sahar, Edward
# Contains Game Class and Model of the Game Class.

from enum import Enum
import torch
import torch.nn as nn
from torch.distributions import Categorical
import sys  # Not used?
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
            self.myopicPrices[player][self.stage]=myopic(self,player)
            price = pricePair[player]
            self.prices[player][self.stage] = price
            self.profit[player][self.stage] = (
                self.demandPotential[player][self.stage] - price)*(price - self.costs[player])
            if self.stage < self.T-1:
                self.demandPotential[player][self.stage + 1] = \
                    self.demandPotential[player][self.stage] + \
                    (pricePair[1-player] - price)/2

    def monopolyPrice(self, player):  # myopic monopoly price
        """
            Computes Monopoly prices.
        """
        return (self.demandPotential[player][self.stage] + self.costs[player])/2

    def myopic(self, player=0):
        """
            Adversary follows Myopic strategy
        """
        return self.monopolyPrice(player)


class Model(DemandPotentialGame):
    """
        Defines the Problem's Model. It is assumed a Markov Decision Process is defined. The class is a Child from the Demand Potential Game Class.
        The reason: Model is a conceptualization of the Game.
    """

    def __init__(self, totalDemand, tupleCosts, totalStages, advMixedStrategy,stateAdvHistory=0) -> None:
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
        return self.getState(0), reward, self.done

    def resetAdversary(self):
        self.adversaryStrategy = self.advMixedStrategy.set_adversary_strategy()

    def adversaryChoosePrice(self):
        """
            Strategy followed by the adversary.
        """

        return self.adversaryStrategy.play(environment=self, player=1)

        # if self.adversaryStrategy == AdversaryModes.constant_132:
        #     return self.const(player=1, price=132)
        # elif self.adversaryStrategy == AdversaryModes.constant_95:
        #     return self.const(player=1, price=95)
        # elif self.adversaryStrategy == AdversaryModes.imitation_128:
        #     return self.imit(player=1, firstprice=128)
        # elif self.adversaryStrategy == AdversaryModes.imitation_132:
        #     return self.imit(player=1, firstprice=132)
        # elif self.adversaryStrategy == AdversaryModes.fight_100:
        #     return self.fight(player=1, firstprice=100)
        # elif self.adversaryStrategy == AdversaryModes.fight_125:
        #     return self.fight(player=1, firstprice=125)
        # elif self.adversaryStrategy == AdversaryModes.fight_lb_125:
        #     return self.fight_lb(player=1, firstprice=125)
        # elif self.adversaryStrategy == AdversaryModes.fight_132:
        #     return self.fight(player=1, firstprice=132)
        # elif self.adversaryStrategy == AdversaryModes.fight_lb_132:
        #     return self.fight_lb(player=1, firstprice=132)
        # elif self.adversaryStrategy == AdversaryModes.guess_125:
        #     return self.fight(player=1, firstprice=125)
        # elif self.adversaryStrategy == AdversaryModes.guess_132:
        #     return self.fight(player=1, firstprice=132)
        # else:
        #     return self.myopic(player=1)

    def step(self, price, mode=1):
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

        return self.getState(self.stage), reward, done

    def getState(self, stage, player=0, advHist=None):
        # [one-hote encoding of stage, our demand, our price, adversary's price history]

        stageEncode = [0]*self.T
        if stage < self.T:
            stageEncode[stage] = 1

        hist = advHist if (advHist is not None) else self.stateAdvHistory
        if stage == 0:
            state = stageEncode + \
                [self.totalDemand/2,
                    ((self.totalDemand/2) + self.costs[player])/2] + ([0]*hist)

        else:
            # check last stageeee demand
            state = stageEncode+[self.demandPotential[player]
                                 [stage], self.prices[player][stage-1]]
            if (hist > 0):
                advHistory = [0]*hist
                j = hist-1
                for i in range(stage-1, max(-1, stage-1-hist), -1):
                    advHistory[j] = self.prices[1-player][i]
                    j -= 1
                state += advHistory
            # elif(player==1 and self.advNN.adv_hist>0):
            #     advHistory = [0]*self.advNN.adv_hist
            #     j = self.advNN.adv_hist-1
            #     for i in range(stage-1, max(-1, stage-1-self.advNN.adv_hist), -1):
            #         advHistory[j] = self.prices[1-player][i]
            #         j -= 1
            #     state+=advHistory

        return torch.tensor(state, dtype=torch.float32)

    def normalizeState(self, state, mode=1):
        # [stage one-hot encoded, agent's demand potential, agent's last price, history of adversary's prices]

        if mode == 1:
            normalized = [0]*len(state)
            for i in range(self.T):
                normalized[i] = state[i]
            for i in range(self.T, len(state)):
                normalized[i] = state[i]/(self.totalDemand)
            return torch.tensor(normalized)
        elif mode == 2:

            normalized = torch.zeros(len(state))
            for i in range(self.T):
                normalized[i] = state[i]

            normalized[self.T] = -self.our_target_demand + \
                state[self.T]  # demand

            for i in range(self.T+1, len(state)):
                normalized[i] = -(self.target_price) + \
                    state[i]  # both players' prices
            return normalized
        elif mode == 3:
            return nn.functional.normalize(state, p=2.0, dim=0)
        elif mode == 4:
            normalized = [0]*len(state)
            for i in range(self.T):
                normalized[i] = state[i]
            for i in range(self.T, len(state)):
                normalized[i] = (state[i]-self.costs[0]) / \
                    (self.totalDemand)
            return torch.tensor(normalized)
        
    def compute_price(self,action,actionStep,player=0):
        return self.myopic(player=player) - (actionStep* action)

# class AdversaryModes(Enum):
#     myopic = 0
#     constant_132 = 1
#     constant_95 = 2
#     imitation_132 = 3
#     imitation_128 = 4
#     fight_132 = 5
#     fight_lb_132 = 6
#     fight_125 = 7
#     fight_lb_125 = 8
#     fight_100 = 9
#     guess_132 = 10
#     guess_125 = 11

class StrategyType(Enum):
    static = 0
    neural_net = 1


class Strategy():
    """
    strategies can be static or they can come from neural nets. If NN, policy is nn.policy o.w. the static function
    """
    _type = None
    _env = None
    _name = None
    _nn = None
    _nnHist = None
    _policy = None

    def __init__(self, strategyType, NNorFunc, name, firstPrice=132) -> None:
        """
        Based on the type of strategy, the neuralnet or the Strategy Function  should be given as input. FirstPrice just applies to static strategies
        """
        self._type = strategyType
        self._name = name
        # self._env = environment

        if strategyType == StrategyType.neural_net:
            self._nn = NNorFunc
            self._policy = NNorFunc.policy
            self._nnHist = NNorFunc.adv_hist
        else:
            self._policy = NNorFunc
            self._firstPrice = firstPrice
    
    def reset(self):
        pass

    def play(self, environment, player=0):
        """
            Computes the action to be played in the environment, nn.step_action is the step size for pricing less than myopic
        """
        self._env = environment
        if self._type == StrategyType.neural_net:
            state = self._env.getState(
                self._env.stage, player, advHist=self._nnHist)
            normState = self._env.normalizeState(state=state)
            probs = self._policy(normState)
            distAction = Categorical(probs)
            action = distAction.sample()
            return  self._env.compute_price(action=action.item(),actionStep=self._nn.action_step, player=player)

        else:
            return self._policy(self._env, player, self._firstPrice)

    def play_against(self, env, adversary):
        """ 
        self is player 0 and adversary is layer 1. The environment should be specified. action_step for the neural netwroks should be set.
        output: tuple (payoff of low cost, payoff of high cost)
        """
        self._env=env

        state, reward, done = env.reset()
        while env.stage<(env.T):
            prices=[0,0]
            prices[0],prices[1]=self.play(env,0),adversary.play(env,1)
            env.updatePricesProfitDemand(prices)
            env.stage += 1
        
        return [sum(env.profit[0]), sum(env.profit[1])]

    def to_mixed_strategy(self):
        """
        Returns a MixedStrategy, Pr(self)=1
        """
        mix=MixedStrategy(probablitiesArray=torch.ones(1),strategiesList=[self])
        
        return mix        






class MixedStrategy():
    _strategies = []
    _strategyProbs = None

    def __init__(self,strategiesList=[], probablitiesArray=None) -> None:
        self._strategies=strategiesList
        self._strategyProbs=probablitiesArray

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
            if self._strategyProbs[i]>0:
                s += f"{self._strategies[i]._name}-{self._strategyProbs[i]:.2f},"
        return s


def myopic(env, player, firstprice=0):
    """
        Adversary follows Myopic strategy
    """
    return env.monopolyPrice(player)


def const(env, player, firstprice):  # constant price strategy
    """
        Adversary follows Constant strategy
    """
    if env.stage == env.T-1:
        return env.monopolyPrice(player)
    return firstprice


def imit(env, player, firstprice):  # price imitator strategy
    if env.stage == 0:
        return firstprice
    if env.stage == env.T-1:
        return env.monopolyPrice(player)
    return env.prices[1-player][env.stage-1]


def fight(env, player, firstprice):  # simplified fighting strategy
    if env.stage == 0:
        return firstprice
    if env.stage == env.T-1:
        return env.monopolyPrice(player)
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
        return env.monopolyPrice(player)

    D = env.demandPotential[player][env.stage]
    Asp = env.aspireDemand[player]

    if D >= Asp:  # keep price, but go slightly towards monopoly if good
        pmono = env.monopolyPrice(player)
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
