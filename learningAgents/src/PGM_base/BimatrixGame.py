from enum import Enum
import globals as gl
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

# totalDemand = 400
# lowCost = 57
# highCost = 71
# totalStages = 25
# adversaryHistroy = 3
# lr = 0.000005
# gamma = 1
# numActions = 3
# actionStep = 3
# numStochasticIter = 10

# # episodes for learning the last stage, then for 2nd to last stage 2*numEpisodes. In total:300*numEpisodes
# numEpisodes = 3000
# numEpisodesReset = numEpisodes
# # increase in num of episodes for each adv in support
# episodeIncreaseAdv = 1000


class BimatrixGame():
    """
    strategies play against each other and fill the matrix of payoff, then the equilibria would be computed using Lemke algorithm
    """

    _strategies_low = []
    _strategies_high = []
    _matrix_A = None
    _matrix_B = None

    def __init__(self, lowCostStrategies, highCostStrategies) -> None:
        # globals.initialize()
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

        env = Model(totalDemand=gl.totalDemand, tupleCosts=(gl.lowCost, gl.highCost),
                    totalStages=gl.totalStages, advMixedStrategy=stratH.to_mixed_strategy())
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
        self._matrix_A = np.append(self._matrix_A, [rowA], axis=0)
        self._matrix_B = np.append(self._matrix_B, [rowB], axis=0)

    def add_high_cost_col(self, colA, colB):
        self._matrix_A = np.hstack((self._matrix_A, np.atleast_2d(colA).T))
        self._matrix_B = np.hstack((self._matrix_B, np.atleast_2d(colB).T))
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
        
        low_prob_str = ", ".join(map("{0:.2f}".format, low_cost_probabilities))
        high_prob_str = ", ".join(map("{0:.2f}".format, high_cost_probabilities))
        print(
            f"equi: [{low_prob_str}], [{high_prob_str}], {low_cost_payoff:.2f}, {high_cost_payoff:.2f}")
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


def training(costs, advMixedStrategy, targetPayoff):
    """
    trains a neuralnet against adversaries. if the expected payoff of new agent is greater than payoff, returns acceptable=true and the new strategy and payoff to be added to the the strategies and matrix.
    """
    acceptable = False

    game = Model(gl.totalDemand, costs, gl.totalStages,
                 advMixedStrategy=advMixedStrategy, stateAdvHistory=gl.adversaryHistroy)
    neuralNet = NNBase(num_input=game.T+2+game.stateAdvHistory,
                       lr=gl.lr, num_actions=gl.numActions, action_step=gl.actionStep, adv_hist=game.stateAdvHistory)

    numberEpisodes = gl.numEpisodes+gl.episodeIncreaseAdv * \
        (support_count(advMixedStrategy._strategyProbs))
    algorithm = ReinforceAlgorithm(
        game, neuralNet, numberIterations=1, numberEpisodes=numberEpisodes, discountFactor=gl.gamma)
    algorithm.solver(print_step=None, options=[
                     1, 10000, 1, 1], converge_break=True)
    a = algorithm.returns[0][-1]
    print(f"{neuralNet.nn_name} is trained against {str(advMixedStrategy)}")

    agentPayoffs = np.zeros(len(advMixedStrategy._strategies))
    advPayoffs = np.zeros(len(advMixedStrategy._strategies))
    expectedPayoff = 0
    for strategyIndex in range(len(advMixedStrategy._strategies)):
        if advMixedStrategy._strategyProbs[strategyIndex] > 0:
            returns = algorithm.playTrainedAgent(adversary=(
                (advMixedStrategy._strategies[strategyIndex]).to_mixed_strategy()), iterNum=gl.numStochasticIter)
            agentPayoffs[strategyIndex] = torch.mean(returns[0])
            advPayoffs[strategyIndex] = torch.mean(returns[1])
            expectedPayoff += (agentPayoffs[strategyIndex]) * \
                (advMixedStrategy._strategyProbs[strategyIndex])
    if expectedPayoff > targetPayoff:
        acceptable = True
        algorithm.write_nn_data(("low" if costs[0]<costs[1] else "high"))
        # compute the payoff against all adv strategies, to be added to the matrix
        for strategyIndex in range(len(advMixedStrategy._strategies)):
            if advMixedStrategy._strategyProbs[strategyIndex] == 0:
                returns = algorithm.playTrainedAgent(adversary=(
                    advMixedStrategy._strategies[strategyIndex].to_mixed_strategy()), iterNum=gl.numStochasticIter)
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


def run_tournament_random(number_rounds):
    equilibria = []
    neuralNet = NNBase(num_input=gl.totalStages+2+gl.adversaryHistroy,
                       lr=gl.lr, num_actions=gl.numActions, action_step=gl.actionStep, adv_hist=gl.adversaryHistroy)
    neuralNet.reset()
    randStrategy = Strategy(StrategyType.neural_net,
                            NNorFunc=neuralNet, name="nnRandom")
    low_cost_players = [randStrategy]
    high_cost_players = [randStrategy]
    bimatrixGame = BimatrixGame(low_cost_players, high_cost_players)
    # bimatrixGame.reset_matrix()
    bimatrixGame.fill_matrix()
    low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff = bimatrixGame.compute_equilibria()
    for round in range(number_rounds):
        print("Round", round, " of ", number_rounds)

        update = False

        
        acceptable, agentPayoffs, advPayoffs, low_cost_player = training([gl.lowCost, gl.highCost], advMixedStrategy=MixedStrategy(
            strategiesList=high_cost_players, probablitiesArray=high_cost_probabilities), targetPayoff=low_cost_payoff)
        if acceptable:
            update = True
            low_cost_players.append(low_cost_player)
            bimatrixGame.add_low_cost_row(agentPayoffs, advPayoffs)
            equilibria.append(
                [low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff])
            print(f"low cost player {low_cost_player._name} added")

            low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff = bimatrixGame.compute_equilibria()
            

        acceptable, agentPayoffs, advPayoffs, high_cost_player = training(
            [gl.highCost, gl.lowCost], advMixedStrategy=MixedStrategy(probablitiesArray=low_cost_probabilities, strategiesList=low_cost_players), targetPayoff=high_cost_payoff)

        if acceptable:
            update = True
            high_cost_players.append(high_cost_player)
            bimatrixGame.add_high_cost_col(advPayoffs, agentPayoffs)
            equilibria.append(
                [low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff])
            print(f"high cost player {high_cost_player._name} added")

            low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff = bimatrixGame.compute_equilibria()

        if update:
            gl.numEpisodes = gl.numEpisodesReset
        else:
            gl.numEpisodes += 1000
    return equilibria, bimatrixGame
