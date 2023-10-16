import BimatrixGame as BG
import globals as gl
import torch
import numpy as np
from environmentModelBase import Model, MixedStrategy, Strategy, StrategyType
import environmentModelBase as em
from learningBase import ReinforceAlgorithm
from neuralNetworkSimple import NNBase

# def myopic19(env, player, firstprice=0):
#     """
#         Adversary follows Myopic strategy
#     """
#     return env.monopolyPrice(player)-19

np.random.seed(0)
gl.initialize()

number_rounds=50

l1=NNBase(lr=gl.lr, num_input=gl.totalStages+2+gl.adversaryHistroy,num_actions=gl.numActions,adv_hist=gl.adversaryHistroy,action_step=gl.actionStep)
l1.load("low,1684386202")
# l2=NNBase(lr=gl.lr, num_input=gl.totalStages+2+gl.adversaryHistroy,num_actions=gl.numActions,adv_hist=gl.adversaryHistroy,action_step=gl.actionStep)
# l2.load("low,1684332617")
# h1=NNBase(lr=gl.lr, num_input=gl.totalStages+2+gl.adversaryHistroy,num_actions=gl.numActions,adv_hist=gl.adversaryHistroy,action_step=gl.actionStep)
# h1.load("high,1684261807")
equilibria = []
low_cost_players = [
                    # Strategy(StrategyType.static, NNorFunc=myopic19, name="myopic19")
                    # Strategy(StrategyType.static, NNorFunc=em.const, name="const", firstPrice=132), 
                    # Strategy(StrategyType.static, NNorFunc=em.guess, name="guess", firstPrice=132),
                    Strategy(StrategyType.neural_net, NNorFunc=l1, name=l1.nn_name)
                    # Strategy(StrategyType.neural_net, NNorFunc=l2, name=l2.nn_name)
                   ]
high_cost_players = [
                    Strategy(StrategyType.static, NNorFunc=em.myopic, name="myopic") ,
                    Strategy(StrategyType.static, NNorFunc=em.const, name="const", firstPrice=132), 
                    Strategy(StrategyType.static, NNorFunc=em.guess, name="guess", firstPrice=132),
                    # Strategy(StrategyType.neural_net, NNorFunc=h1, name=h1.nn_name),
                   ]
bimatrixGame = BG.BimatrixGame(low_cost_players, high_cost_players)
    # bimatrixGame.reset_matrix()
bimatrixGame.fill_matrix()
print(bimatrixGame._matrix_A)
print(bimatrixGame._matrix_B)
# low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff = bimatrixGame.compute_equilibria()
# for round in range(number_rounds):
#     print("Round", round, " of ", number_rounds)

#     update = False


#     acceptable, agentPayoffs, advPayoffs, low_cost_player = BG.training([gl.lowCost, gl.highCost], advMixedStrategy=MixedStrategy(
#         strategiesList=high_cost_players, probablitiesArray=high_cost_probabilities), targetPayoff=low_cost_payoff)
#     if acceptable:
#         update = True
#         low_cost_players.append(low_cost_player)
#         bimatrixGame.add_low_cost_row(agentPayoffs, advPayoffs)
#         equilibria.append(
#             [low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff])
#         print(f"low cost player {low_cost_player._name} added")

#         low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff = bimatrixGame.compute_equilibria()


#     acceptable, agentPayoffs, advPayoffs, high_cost_player = BG.training(
#         [gl.highCost, gl.lowCost], advMixedStrategy=MixedStrategy(probablitiesArray=low_cost_probabilities, strategiesList=low_cost_players), targetPayoff=high_cost_payoff)

#     if acceptable:
#         update = True
#         high_cost_players.append(high_cost_player)
#         bimatrixGame.add_high_cost_col(advPayoffs, agentPayoffs)
#         equilibria.append(
#             [low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff])
#         print(f"high cost player {high_cost_player._name} added")

#         low_cost_probabilities, high_cost_probabilities, low_cost_payoff, high_cost_payoff = bimatrixGame.compute_equilibria()

    # if update:
    #     gl.numEpisodes = gl.numEpisodesReset
    # else:
    #     gl.numEpisodes += 1000