import globals as gl
import torch
import numpy as np
from environmentModelBase import Model
from learningBase import ReinforceAlgorithm, MixedStrategy, Strategy, StrategyType
from neuralNetworkSimple import NNBase
from fractions import Fraction
import bimatrix
from multiprocessing import Pool
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

    low_strategies = []
    high_strategies = []
    matrix_A = None
    matrix_B = None

    def __init__(self, lowCostStrategies, highCostStrategies) -> None:
        # globals.initialize()
        self.low_strategies = lowCostStrategies
        self.high_strategies = highCostStrategies

    def reset_matrix(self):
        self.matrix_A = np.zeros(
            (len(self.low_strategies), len(self.high_strategies)))
        self.matrix_B = np.zeros(
            (len(self.low_strategies), len(self.high_strategies)))

    def fill_matrix(self):
        self.reset_matrix()

        for low in range(len(self.low_strategies)):
            for high in range(len(self.high_strategies)):
                self.update_matrix_entry(low, high)

    def update_matrix_entry(self, lowIndex, highIndex):
        stratL = self.low_strategies[lowIndex]
        stratH = self.high_strategies[highIndex]
        stratL.reset()
        stratH.reset()

        env = Model(totalDemand=gl.total_demand, tupleCosts=(gl.low_cost, gl.high_cost),
                    totalStages=gl.total_stages, advMixedStrategy=stratH.to_mixed_strategy())
        payoffs = [stratL.play_against(env, stratH)
                   for _ in range(gl.num_stochastic_iter)]
        
        mean_payoffs=(np.mean(np.array(payoffs), axis=0)).tolist()

        self.matrix_A[lowIndex][highIndex], self.matrix_B[lowIndex][highIndex] = mean_payoffs[0], mean_payoffs[1]

    def write_all_matrix(self):
        # print("A: \n", self._matrix_A)
        # print("B: \n", self._matrix_B)

        output = f"{len(self.matrix_A)} {len(self.matrix_A[0])}\n\n"

        for matrix in [self.matrix_A, self.matrix_B]:
            for i in range(len(self.matrix_A)):
                for j in range(len(self.matrix_A[0])):
                    output += f"{matrix[i][j]:7.0f} "
                output += "\n"
            output += "\n"

        with open(f".\games\game{len(self.matrix_A)}x{len(self.matrix_A[0])}.txt", "w") as out:
            out.write(output)
        with open("game.txt", "w") as out:
            out.write(output)

    def add_low_cost_row(self, rowA, rowB):
        self.matrix_A = np.append(self.matrix_A, [rowA], axis=0)
        self.matrix_B = np.append(self.matrix_B, [rowB], axis=0)

    def add_high_cost_col(self, colA, colB):
        self.matrix_A = np.hstack((self.matrix_A, np.atleast_2d(colA).T))
        self.matrix_B = np.hstack((self.matrix_B, np.atleast_2d(colB).T))
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
            len(self.low_strategies), low_cost_probs, low_cost_support)
        high_cost_probabilities = return_distribution(
            len(self.high_strategies), high_cost_probs, high_cost_support)
        low_cost_payoff = np.matmul(low_cost_probabilities, np.matmul(
            self.matrix_A, np.transpose(high_cost_probabilities)))
        high_cost_payoff = np.matmul(low_cost_probabilities, np.matmul(
            self.matrix_B, np.transpose(high_cost_probabilities)))

        low_prob_str = ", ".join(map("{0:.2f}".format, low_cost_probabilities))
        high_prob_str = ", ".join(
            map("{0:.2f}".format, high_cost_probabilities))
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


def training(costs, adv_mixed_strategy, target_payoff):
    """
    trains a neuralnet against adversaries. if the expected payoff of new agent is greater than payoff, returns acceptable=true and the new strategy and payoff to be added to the the strategies and matrix.
    """
    gl.initialize()
    acceptable = False

    game = Model(gl.total_demand, costs, gl.total_stages,
                 advMixedStrategy=adv_mixed_strategy, stateAdvHistory=gl.num_adv_history)
    neural_net = NNBase(num_input=game.T+2+game.stateAdvHistory,
                        lr=gl.lr, num_actions=gl.num_actions)

    number_episodes = gl.num_episodes+gl.episode_adv_increase * \
        (support_count(adv_mixed_strategy._strategyProbs))
    algorithm = ReinforceAlgorithm(
        game, neural_net, numberIterations=1, numberEpisodes=number_episodes, discountFactor=gl.gamma)
    algorithm.solver()
    a = algorithm.returns[0][-1]
    print(f"{neural_net.name} is trained against {str(adv_mixed_strategy)}")

    agent_payoffs = np.zeros(len(adv_mixed_strategy._strategies))
    adv_payoffs = np.zeros(len(adv_mixed_strategy._strategies))
    expected_payoff = 0
    for strategy_index in range(len(adv_mixed_strategy._strategies)):
        if adv_mixed_strategy._strategyProbs[strategy_index] > 0:
            returns = algorithm.play_trained_agent(adversary=(
                (adv_mixed_strategy._strategies[strategy_index]).to_mixed_strategy()), iterNum=gl.num_stochastic_iter)
            agent_payoffs[strategy_index] = torch.mean(returns[0])
            adv_payoffs[strategy_index] = torch.mean(returns[1])
            expected_payoff += (agent_payoffs[strategy_index]) * \
                (adv_mixed_strategy._strategyProbs[strategy_index])
    if expected_payoff > target_payoff:
        acceptable = True
        algorithm.write_nn_data(("low" if costs[0] < costs[1] else "high"))
        # compute the payoff against all adv strategies, to be added to the matrix
        for strategy_index in range(len(adv_mixed_strategy._strategies)):
            if adv_mixed_strategy._strategyProbs[strategy_index] == 0:
                returns = algorithm.play_trained_agent(adversary=(
                    adv_mixed_strategy._strategies[strategy_index].to_mixed_strategy()), iterNum=gl.num_stochastic_iter)
                agent_payoffs[strategy_index] = torch.mean(returns[0])
                adv_payoffs[strategy_index] = torch.mean(returns[1])

    return [acceptable, agent_payoffs, adv_payoffs, Strategy(strategyType=StrategyType.neural_net, NNorFunc=neural_net, name=neural_net.name)]


def support_count(list):
    """
    gets a list and returns the number of elements that are greater than zero
    """
    counter = 0
    for item in list:
        if item > 0:
            counter += 1
    return counter


# def run_tournament_random(number_rounds):
#     equilibria = []
#     neural_net = NNBase(num_input=gl.total_stages+2+gl.num_adv_history,
#                         lr=gl.lr, num_actions=gl.num_actions)
#     neural_net.reset()
#     rand_strategy = Strategy(StrategyType.neural_net,
#                              NNorFunc=neural_net, name="nnRandom")
#     low_cost_players = [rand_strategy]
#     high_cost_players = [rand_strategy]
#     bimatrix_game = BimatrixGame(low_cost_players, high_cost_players)
#     # bimatrixGame.reset_matrix()
#     bimatrix_game.fill_matrix()
#     low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff = bimatrix_game.compute_equilibria()
#     for round in range(number_rounds):
#         print("Round", round, " of ", number_rounds)

#         update = False
#         # inputs = [([gl.low_cost, gl.high_cost], MixedStrategy(strategiesList=high_cost_players, probablitiesArray=high_cost_probabilities), low_cost_payoff), ([
#         #     gl.high_cost, gl.low_cost], MixedStrategy(probablitiesArray=low_cost_probabilities, strategiesList=low_cost_players), high_cost_payoff)]
#         # with Pool() as pool:
#         #     results = pool.starmap(training, inputs)

#         # [low_acceptable, low_agent_payoffs,
#         #     low_adv_payoffs, low_cost_player] = results[0]
#         # [high_acceptable, high_agent_payoffs,
#         #     high_adv_payoffs, high_cost_player] = results[1]


#         # training([gl.low_cost, gl.high_cost], adv_mixed_strategy=MixedStrategy(
#         #     strategiesList=high_cost_players, probablitiesArray=high_cost_probabilities), target_payoff=low_cost_payoff)
#         acceptable, agent_payoffs, adv_payoffs, low_cost_player = training([gl.low_cost, gl.high_cost], adv_mixed_strategy=MixedStrategy(
#             strategiesList=high_cost_players, probablitiesArray=high_cost_probabilities), target_payoff=low_cost_payoff)
#         if acceptable:
#             update = True
#             low_cost_players.append(low_cost_player)
#             bimatrix_game.add_low_cost_row(agent_payoffs, adv_payoffs)
#             equilibria.append(
#                 [low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff])
#             print(f"low cost player {low_cost_player.name} added")

#             low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff = bimatrix_game.compute_equilibria()


#         acceptable, agent_payoffs, adv_payoffs, high_cost_player = training(
#             [gl.high_cost, gl.low_cost], adv_mixed_strategy=MixedStrategy(probablitiesArray=low_cost_probabilities, strategiesList=low_cost_players), target_payoff=high_cost_payoff)

#         if acceptable:
#             update = True
#             high_cost_players.append(high_cost_player)
#             bimatrix_game.add_high_cost_col(adv_payoffs, agent_payoffs)
#             equilibria.append(
#                 [low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff])
#             print(f"high cost player {high_cost_player.name} added")

#             low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff = bimatrix_game.compute_equilibria()

#         if update:
#             gl.num_episodes = gl.num_episodes_reset
#         else:
#             gl.num_episodes += 1000
#     return equilibria, bimatrix_game

def run_tournament(bimatrix_game, number_rounds):
    equilibria = []

    # bimatrixGame.reset_matrix()
    bimatrix_game.fill_matrix()
    low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff = bimatrix_game.compute_equilibria()
    for round in range(number_rounds):
        print("Round", round, " of ", number_rounds)

        update = False
        inputs = [([gl.low_cost, gl.high_cost], MixedStrategy(strategiesList=bimatrix_game.high_strategies, probablitiesArray=high_cost_probabilities), low_cost_payoff), ([
            gl.high_cost, gl.low_cost], MixedStrategy(probablitiesArray=low_cost_probabilities, strategiesList=bimatrix_game.low_strategies), high_cost_payoff)]
        
        with Pool(2) as pool:
            results = pool.starmap(training, inputs)

        [low_acceptable, low_agent_payoffs,
            low_adv_payoffs, low_cost_player] = results[0]
        [high_acceptable, high_agent_payoffs,
            high_adv_payoffs, high_cost_player] = results[1]

        # [low_acceptable, low_agent_payoffs, low_adv_payoffs, low_cost_player] = training([gl.low_cost, gl.high_cost], MixedStrategy(
        #     strategiesList=bimatrix_game.high_strategies, probablitiesArray=high_cost_probabilities), low_cost_payoff)
        # [high_acceptable, high_agent_payoffs, high_adv_payoffs, high_cost_player] = training([
        #     gl.high_cost, gl.low_cost], MixedStrategy(probablitiesArray=low_cost_probabilities, strategiesList=bimatrix_game.low_strategies), high_cost_payoff)

        # training([gl.low_cost, gl.high_cost], adv_mixed_strategy=MixedStrategy(
        #     strategiesList=high_cost_players, probablitiesArray=high_cost_probabilities), target_payoff=low_cost_payoff)
        if low_acceptable:
            update = True
            bimatrix_game.low_strategies.append(low_cost_player)
            bimatrix_game.add_low_cost_row(low_agent_payoffs, low_adv_payoffs)

            print(f"low cost player {low_cost_player.name} added")

            # low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff = bimatrix_game.compute_equilibria()

        # acceptable, agent_payoffs, adv_payoffs, high_cost_player = training(
        #     [gl.high_cost, gl.low_cost], adv_mixed_strategy=MixedStrategy(probablitiesArray=low_cost_probabilities, strategiesList=low_cost_players), target_payoff=high_cost_payoff)

        if high_acceptable:
            update = True
            bimatrix_game.high_strategies.append(high_cost_player)
            if low_acceptable:
                high_adv_payoffs = np.append(high_adv_payoffs, 0)
                high_agent_payoffs = np.append(high_agent_payoffs, 0)

            bimatrix_game.add_high_cost_col(
                high_adv_payoffs, high_agent_payoffs)
            if low_acceptable:
                bimatrix_game.update_matrix_entry(
                    len(bimatrix_game.low_strategies)-1, len(bimatrix_game.high_strategies)-1)
            # equilibria.append(
            #     [low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff])
            print(f"high cost player {high_cost_player.name} added")

            # low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff = bimatrix_game.compute_equilibria()

        if update:
            equilibria.append(
                [low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff])
            low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff = bimatrix_game.compute_equilibria()
            gl.num_episodes = gl.num_episodes_reset
        else:
            gl.num_episodes += 1000
    return equilibria
