from enum import Enum
import torch
from torch.distributions import Categorical
import torch.nn as nn
import numpy as np
from environmentModelBase import Model, MixedStrategy, Strategy, StrategyType
import environmentModelBase as em
from learningBase import ReinforceAlgorithm
from neuralNetworkSimple import NNBase
from fractions import Fraction
import bimatrix

totalDemand = 400
lowCost = 57
highCost = 71
totalStages = 25
adversaryHistroy = 3
lr = 0.000005
gamma = 1
numActions = 3
actionStep = 3
numStochasticIter = 10

# episodes for learning the last stage, then for 2nd to last stage 2*numEpisodes. In total:300*numEpisodes
numEpisodes = 4000
numEpisodesReset = numEpisodes
# increase in num of episodes for each adv in support
episodeIncreaseAdv = 1000


class BimatrixGame():
    """
    strategies play against each other and fill the matrix of payoff, then the equilibria would be computed using Lemke algorithm
    """

    _strategies_low = []
    _strategies_high = []
    _matrix_A = None
    _matrix_B = None

    def __init__(self, lowCostStrategies, highCostStrategies) -> None:
        self._strategies_low = lowCostStrategies
        self._strategies_high = highCostStrategies

    def reset_matrix(self):
        self._matrix_A = np.zeros(
            (len(self._strategies_low), len(self._strategies_high)))
        self._matrix_B = np.zeros(
            (len(self._strategies_low), len(self._strategies_high)))

    def fill_matrix(self):
        self.reset_matrix()

        for low in range(len(self._strategies_low)):
            for high in range(len(self._strategies_high)):
                self.update_matrix_entry(low, high)

    def update_matrix_entry(self, lowIndex, highIndex):
        stratL = self._strategies_low[lowIndex]
        stratH = self._strategies_high[highIndex]
        stratL.reset()
        stratH.reset()

        env = Model(totalDemand=totalDemand, tupleCosts=(lowCost, highCost),
                    totalStages=totalStages, advMixedStrategy=stratH.to_mixed_strategy())
        [self._matrix_A[lowIndex][highIndex], self._matrix_B[lowIndex]
            [highIndex]] = stratL.play_against(env, stratH)
        pass

    def write_all_matrix(self):
        # print("A: \n", self._matrix_A)
        # print("B: \n", self._matrix_B)

        output = f"{len(self._matrix_A)} {len(self._matrix_A[0])}\n\n"

        for matrix in [self._matrix_A, self._matrix_B]:
            for i in range(len(self._matrix_A)):
                for j in range(len(self._matrix_A[0])):
                    output += f"{matrix[i][j]:7.0f} "
                output += "\n"
            output += "\n"

        with open(f".\games\game{len(self._matrix_A)}x{len(self._matrix_A[0])}.txt", "w") as out:
            out.write(output)
        with open("game.txt", "w") as out:
            out.write(output)

    def add_low_cost_row(self, rowA, rowB):
        self._matrix_A=np.append(self._matrix_A,[rowA],axis=0)
        self._matrix_B=np.append(self._matrix_B,[rowB],axis=0)

    def add_high_cost_col(self, colA, colB):
        self._matrix_A=np.append(self._matrix_A,[colA],axis=1)
        self._matrix_B=np.append(self._matrix_B,[colB],axis=1)
        # for j in range(len(self._matrix_A)):
        #     self._matrix_A[j].append(colA[j])
        #     self._matrix_B[j].append(colB[j])

    def compute_equilibria(self):
        self.write_all_matrix()
        game = bimatrix.bimatrix("game.txt")
        equilibrium = game.tracing(100)
        low_cost_probs, high_cost_probs, low_cost_support, high_cost_support = recover_probs(
            equilibrium)
        low_cost_probabilities = return_distribution(
            len(self._strategies_low), low_cost_probs, low_cost_support)
        high_cost_probabilities = return_distribution(
            len(self._strategies_high), high_cost_probs, high_cost_support)
        low_cost_payoff = np.matmul(low_cost_probabilities, np.matmul(
            self._matrix_A, np.transpose(high_cost_probabilities)))
        high_cost_payoff = np.matmul(low_cost_probabilities, np.matmul(
            self._matrix_B, np.transpose(high_cost_probabilities)))
        return low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff


def recover_probs(test):
    low_cost_probs, high_cost_probs, rest = test.split(")")
    low_cost_probs = low_cost_probs.split("(")[1]
    _, high_cost_probs = high_cost_probs.split("(")
    high_cost_probs = [float(Fraction(s)) for s in high_cost_probs.split(',')]
    low_cost_probs = [float(Fraction(s)) for s in low_cost_probs.split(',')]
    _, low_cost_support, high_cost_support = rest.split('[')
    high_cost_support, _ = high_cost_support.split(']')
    high_cost_support = [int(s) for s in high_cost_support.split(',')]
    low_cost_support, _ = low_cost_support.split(']')
    low_cost_support = [int(s) for s in low_cost_support.split(',')]
    return low_cost_probs, high_cost_probs, low_cost_support, high_cost_support


def return_distribution(number_players, cost_probs, cost_support):
    player_probabilities = [0] * number_players
    for index, support in enumerate(cost_support):
        player_probabilities[support] = cost_probs[support]
    return player_probabilities


def run_tournament(number_rounds):
    global numEpisodes
    equilibria = []
    low_cost_players = [Strategy(StrategyType.static, NNorFunc=em.myopic, name="myopic"), Strategy(
        StrategyType.static, NNorFunc=em.const, name="const", firstPrice=132), Strategy(StrategyType.static, NNorFunc=em.myopic, name="guess", firstPrice=132)]
    high_cost_players = [Strategy(StrategyType.static, NNorFunc=em.myopic, name="myopic"), Strategy(
        StrategyType.static, NNorFunc=em.const, name="const", firstPrice=132), Strategy(StrategyType.static, NNorFunc=em.myopic, name="guess", firstPrice=132)]
    bimatrixGame = BimatrixGame(low_cost_players, high_cost_players)
    # bimatrixGame.reset_matrix()
    bimatrixGame.fill_matrix()
    for round in range(number_rounds):
        print("Round", round, " of ", number_rounds)
        update = False

        low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff = bimatrixGame.compute_equilibria()

        acceptable, agentPayoffs, advPayoffs, low_cost_player = training([lowCost, highCost], advMixedStrategy=MixedStrategy(
                strategiesList=high_cost_players, probablitiesArray=high_cost_probabilities),targetPayoff= low_cost_payoff)
        if acceptable:
            update = True
            low_cost_players.append(low_cost_player)
            bimatrixGame.add_low_cost_row(agentPayoffs, advPayoffs)
            equilibria.append(
                [low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff])
            print("equi: ", low_cost_probabilities, high_cost_probabilities,
                  low_cost_payoff, high_cost_payoff, f"\nlow cost player {low_cost_player._name} added")

            low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff = bimatrixGame.compute_equilibria()

        acceptable, agentPayoffs, advPayoffs, high_cost_player = training(
            [highCost, lowCost], advMixedStrategy=MixedStrategy(probablitiesArray= low_cost_probabilities, strategiesList=low_cost_players), targetPayoff= high_cost_payoff)

        if acceptable:
            update = True
            high_cost_players.append(high_cost_player)
            bimatrixGame.add_high_cost_col(advPayoffs, agentPayoffs)
            equilibria.append(
                [low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff])
            print("equi: ", low_cost_probabilities, high_cost_probabilities,
                  low_cost_payoff, high_cost_payoff, f"\nhigh cost player {high_cost_player._name} added")

            low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff = bimatrixGame.compute_equilibria()

        if update:
            numEpisodes = numEpisodesReset
        else:
            numEpisodes += 1000


def training(costs, advMixedStrategy, targetPayoff):
    """
    trains a neuralnet against adversaries. if the expected payoff of new agent is greater than payoff, returns acceptable=true and the new strategy and payoff to be added to the the strategies and matrix.
    """
    acceptable = False

    game = Model(totalDemand, costs, totalStages,
                 advMixedStrategy=advMixedStrategy, stateAdvHistory=adversaryHistroy)
    neuralNet = NNBase(num_input=game.T+2+game.stateAdvHistory,
                       lr=lr, num_actions=numActions, action_step=actionStep, adv_hist=game.stateAdvHistory)

    numberEpisodes = numEpisodes+episodeIncreaseAdv * \
        (support_count(advMixedStrategy._strategyProbs))
    algorithm = ReinforceAlgorithm(
        game, neuralNet, numberIterations=1, numberEpisodes=numberEpisodes, discountFactor=gamma)
    algorithm.solver(print_step=None, options=[
                     1, 10000, 1, 1], converge_break=True)
    a=algorithm.returns[0][-1]
    print(f"{neuralNet.nn_name} is trained against {str(advMixedStrategy)}")

    agentPayoffs = np.zeros(len(advMixedStrategy._strategies))
    advPayoffs = np.zeros(len(advMixedStrategy._strategies))
    expectedPayoff = 0
    for strategyIndex in range(len(advMixedStrategy._strategies)):
        if advMixedStrategy._strategyProbs[strategyIndex] > 0:
            returns = algorithm.playTrainedAgent(adversary=(
                (advMixedStrategy._strategies[strategyIndex]).to_mixed_strategy()), iterNum=numStochasticIter)
            agentPayoffs[strategyIndex] = torch.mean(returns[0])
            advPayoffs[strategyIndex] = torch.mean(returns[1])
            expectedPayoff += (agentPayoffs[strategyIndex]) * \
                (advMixedStrategy._strategyProbs[strategyIndex])
    if expectedPayoff > targetPayoff:
        acceptable = True
        # compute the payoff against all adv strategies, to be added to the matrix
        for strategyIndex in range(len(advMixedStrategy._strategies)):
            if advMixedStrategy._strategyProbs[strategyIndex] == 0:
                returns = algorithm.playTrainedAgent(adversary=(
                    advMixedStrategy._strategies[strategyIndex].to_mixed_strategy()), iterNum=numStochasticIter)
                agentPayoffs[strategyIndex] = torch.mean(returns[0])
                advPayoffs[strategyIndex] = torch.mean(returns[1])

    return acceptable, agentPayoffs, advPayoffs, Strategy(strategyType=StrategyType.neural_net, NNorFunc=neuralNet, name=neuralNet.nn_name)


def support_count(list):
    """
    gets a list and returns the number of elements that are greater than zero
    """
    counter = 0
    for item in list:
        if item > 0:
            counter += 1
    return counter
    # # did not understand what Ed did here, I rewrote it
    # episode_counter = 0
    # number_rounds = 1
    # number_episodes_per_round = int(number_episodes / number_rounds)
    # for round_ in range(number_rounds):
    #     algorithm.epsilon_greedy_learning(
    #         number_episodes_per_round, episode_counter, number_episodes)
    #     episode_counter += number_episodes_per_round
    # agent = Qtable.to_policy_table(costs[0])
    # payoff = 0
    # if costs[0] < costs[1]:
    #     for index, adversary in enumerate(adversaries):
    #         _, _, payoff_against_adversary, _ = new_equilibrium(
    #             [agent], [adversary], discount_factor, costs, total_stages, [200, 200])
    #         payoff += (payoff_against_adversary *
    #                    adversary_probabilities[index])
    # else:
    #     for index, adversary in enumerate(adversaries):
    #         _, _, _, payoff_against_adversary = new_equilibrium(
    #             [adversary], [agent], discount_factor, [costs[1], costs[0]], total_stages, [200, 200])
    #         payoff += (payoff_against_adversary *
    #                    adversary_probabilities[index])
    # return Qtable.to_policy_table(costs[0]), payoff

# class Strategy():
#     """
#     strategies can be static or they can come from neural nets

#     """
#     _type = None
#     _name = None
#     _neural_net = None
#     _q_table = None
#     _static_index = None

#     def __init__(self, strategyType, name, staticIndex=None, neuralNet=None, history_num=0, qTable=None) -> None:
#         """
#         Based on the type of strategy, the index or neuralnet or q-table should be given as input
#         """
#         self._type = strategyType
#         self._name = name
#         self._neural_net = neuralNet
#         self._neural_net_history = history_num
#         self._static_index = staticIndex
#         self._q_table = qTable
#         self.stochastic_iters = 1

#     def reset(self):
#         pass

#     def play(self, adversary):
#         """
#         self is low cost and gets adversary(high cost) strategy and they play
#         output: tuple (payoff of low cost, payoff of high cost)
#         """
#         if self._type == StrategyType.static and adversary._type == StrategyType.static:
#             return self.play_static_static(adversary._static_index)
#         elif self._type == StrategyType.static and adversary._type == StrategyType.neural_net:
#             returns = torch.zeros(2, self.stochastic_iters)
#             for iter in range(self.stochastic_iters):

#                 game = Model(totalDemand=400,
#                              tupleCosts=(highCost, lowCost),
#                              totalStages=25, adversaryProbs=None, stateAdvHistory=adversary._neural_net_history)
#                 algorithm = ReinforceAlgorithm(
#                     game, neuralNet=None, numberIterations=3, numberEpisodes=1_000_000, discountFactor=1)

#                 algorithm.policy = adversary._neural_net.policy
#                 returns[1][iter], returns[0][iter] = algorithm.playTrainedAgent(
#                     AdversaryModes(self._static_index), 1)

#             return [torch.mean(returns[0]), torch.mean(returns[1])]

#         elif self._type == StrategyType.neural_net and adversary._type == StrategyType.static:
#             returns = torch.zeros(2, self.stochastic_iters)
#             for iter in range(self.stochastic_iters):
#                 game = Model(totalDemand=400,
#                              tupleCosts=(lowCost, highCost),
#                              totalStages=25, adversaryProbs=None, stateAdvHistory=self._neural_net_history)
#                 algorithm = ReinforceAlgorithm(
#                     game, neuralNet=None, numberIterations=3, numberEpisodes=1_000_000, discountFactor=1)

#                 algorithm.policy = self._neural_net.policy
#                 returns[0][iter], returns[1][iter] = algorithm.playTrainedAgent(
#                     AdversaryModes(adversary._static_index), 1)

#             return [torch.mean(returns[0]), torch.mean(returns[1])]

#         elif self._type == StrategyType.neural_net and adversary._type == StrategyType.neural_net:
#             returns = torch.zeros(2, self.stochastic_iters)
#             for iter in range(self.stochastic_iters):
#                 advprobs = torch.zeros(len(AdversaryModes))
#                 advprobs[10] = 1
#                 game = Model(totalDemand=400, tupleCosts=(lowCost, highCost), totalStages=25, adversaryProbs=advprobs,
#                              stateAdvHistory=self._neural_net_history, advNN=adversary._neural_net)
#                 algorithm = ReinforceAlgorithm(
#                     game, neuralNet=None, numberIterations=3, numberEpisodes=1_000_000, discountFactor=1)
#                 algorithm.policy = self._neural_net.policy

#                 state, reward, done = game.reset()
#                 while not done:
#                     prevState = state
#                     normPrevState = game.normalizeState(
#                         prevState)
#                     probs = self._neural_net.policy(normPrevState)
#                     distAction = Categorical(probs)
#                     action = distAction.sample()

#                     state, reward, done = game.step(
#                         prevState, action.item())
#                 returns[0][iter], returns[1][iter] = sum(
#                     game.profit[0]), sum(game.profit[1])

#             return [torch.mean(returns[0]), torch.mean(returns[1])]
#         else:
#             print("play is not implemented!")

#     def play_static_static(self, adversaryIndex):
#         """
#         self is the low cost player
#         """

#         selfIndex = self._static_index

#         lProbs = torch.zeros(len(AdversaryModes))
#         lProbs[selfIndex] = 1

#         hProbs = torch.zeros(len(AdversaryModes))
#         hProbs[adversaryIndex] = 1

#         lGame = Model(totalDemand=400,
#                       tupleCosts=(highCost, lowCost),
#                       totalStages=25, adversaryProbs=lProbs, stateAdvHistory=0)
#         hGame = Model(totalDemand=400,
#                       tupleCosts=(lowCost, highCost),
#                       totalStages=25, adversaryProbs=hProbs, stateAdvHistory=0)
#         lGame.reset()
#         hGame.reset()

#         for i in range(totalStages):
#             lAction = lGame.adversaryChoosePrice()
#             hAction = hGame.adversaryChoosePrice()
#             lGame.updatePricesProfitDemand([hAction, lAction])
#             hGame.updatePricesProfitDemand([lAction, hAction])
#             lGame.stage += 1
#             hGame.stage += 1

#             # print("\nl: ", lAction, " h: ", hAction, "\nprofits: ", hGame.profit,"\ndemand: ", hGame.demandPotential,"\nprices:",hGame.prices)
#         profits = np.array(hGame.profit)
#         returns = profits.sum(axis=1)
#         return returns


# class StrategyType(Enum):
#     static = 0
#     q_table = 1
#     neural_net = 2
