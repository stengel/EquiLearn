# Francisco, Sahar, Edward
# ReinforceAlgorithm Class: Solver.
import environmentModel as em
import torch
import torch.nn as nn
from torch.distributions import Categorical
import sys
import numpy as np  # numerical python
import pandas as pd
from matplotlib import pyplot as plt
# printoptions: output limited to 2 digits after decimal point
np.set_printoptions(precision=2, suppress=False)


class Solver():

    def __init__(self, numberEpisodes, Model, discountFactor, numberIterations):
        self.numberEpisodes = numberEpisodes
        self.env = Model
        self.gamma = discountFactor
        self.numberIterations = numberIterations
        self.bestPolicy = None

    def runBestPolicy(self):
        """
            Run best policy from the Reinforcement Learning Algorithm. It needs to be used after training.
        """

        state, reward, done = self.env.reset()
        returns = 0
        while not done:
            prev_state = state
            probs = self.bestPolicy(prev_state)
            distAction = Categorical(probs)
            action = distAction.sample()

            state, reward, done = self.env.step(prev_state, action.item())
            returns = returns + reward

        return returns


class ReinforceAlgorithm(Solver):
    """
        Model Solver.
    """

    def __init__(self, Model, neuralNet, numberIterations, numberEpisodes, discountFactor) -> None:
        super().__init__(numberEpisodes, Model, discountFactor, numberIterations)

        self.env.adversaryReturns = np.zeros(numberEpisodes)
        self.neuralNetwork = neuralNet
        self.policy = None
        self.optim = None
        self.bestAverageRetu = 0

        self.returns = []
        self.loss = []
        # self.returns = np.zeros((numberIterations, numberEpisodes))
        # self.loss = np.zeros((numberIterations, numberEpisodes))

    def resetPolicyNet(self):
        """
            Reset Policy Neural Network.
        """
        self.policy, self.optim = self.neuralNetwork.reset()

    def savePolicy(self):
        pass

    def solver(self, print_step=10_000, prob_break_limit_ln=-0.001):
        """
            Method that performs Monte Carlo Policy Gradient algorithm. 
        """

        for iteration in range(self.numberIterations):
            self.resetPolicyNet()

            self.meanStageValue = torch.zeros(self.env.T)
            self.returns.append([])
            self.loss.append([])

            for episode in range(self.numberEpisodes):

                episodeMemory = list()
                state, reward, done = self.env.reset()
                returns = 0
                probs_lst = []
                while not done:
                    prevState = state
                    normPrevState = self.normalizeState(prevState)
                    probs = self.policy(normPrevState)
                    distAction = Categorical(probs)
                    probs_lst.append(probs)
                    action = distAction.sample()

                    # if episode % 1000 == 0:
                    #     print("-"*30)
                    #     print("probs= ", probs)

                    state, reward, done = self.env.step(
                        prevState, action.item())
                    returns = returns + reward
                    episodeMemory.append((normPrevState, action, reward))

                states = torch.stack([item[0] for item in episodeMemory])
                actions = torch.tensor([item[1] for item in episodeMemory])
                rewards = torch.tensor([item[2] for item in episodeMemory])

                action_probs = torch.stack(probs_lst)
                action_dists = Categorical(action_probs)

                action_logprobs = action_dists.log_prob(actions)

                discReturns = self.returnsComputation(rewards, episodeMemory)

                if episode == 0:
                    self.meanStageValue = discReturns

                baseDiscReturns = discReturns-self.meanStageValue

                # is it a good idea? to get over the big weights in NN
                baseDiscReturns /= 1000

                loss = - (torch.sum(baseDiscReturns*action_logprobs)) / \
                    len(episodeMemory)

                shouldBreak = torch.all((action_logprobs > (
                    prob_break_limit_ln*torch.ones(self.env.T))))
                

                if (episode % print_step == 0) or shouldBreak:
                    print("-"*50)
                    print(episode, "  adversary: ", self.env.adversaryMode)
                    print("  actions: ", actions)
                    
                    print("loss= ", loss, "  , return= ", returns)
                    # print("states= ", states)
                    print("probs of actions: ", torch.exp(action_logprobs))
                    print("shouldBreak:", shouldBreak.item())
                    # print("actionProbsDist",action_probs)
                    # print("action_dists",action_dists)
                    # print("action_logprobs",action_logprobs)
                    print("baseDiscReturns/1000=", baseDiscReturns)
                    # print("discReturns=", discReturns)
                    print("meanStageValue= ", self.meanStageValue)
                    
                if episode != 0:
                    self.meanStageValue = (
                        (self.meanStageValue*episode)+discReturns)/(episode+1)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # sum of the our player's rewards  rounds 0-25
                # self.returns[iteration][episode] = returns
                # self.loss[iteration][episode] = loss
                self.returns[iteration].append(returns)
                self.loss[iteration].append(loss.item())

                # all probs >0.999 means coverged? break
                if shouldBreak:
                    self.returns[iteration] = self.returns[iteration][0:episode]
                    self.loss[iteration] = self.loss[iteration][0:episode]
                    break

            # averageRetu = (
            #     (self.returns[iteration]).sum())/(self.numberEpisodes)
            # if (self.bestPolicy is None) or (averageRetu > self.bestAverageRetu):
            #     self.bestPolicy = self.policy
            #     self.bestAverageRetu = averageRetu

            plt.plot(self.returns[iteration])
            plt.show()

    def returnsComputation(self, rewards, episodeMemory):
        """
        Method computes vector of returns for every stage. The returns are the cumulative rewards from that stage.
        """
        return torch.tensor([torch.sum(rewards[i:] * (self.gamma ** torch.arange(0, (len(episodeMemory)-i)))) for i in range(len(episodeMemory))])

    def normalizeState(self, state):
        normalized = [0]*len(state)
        normalized[0] = (state[0]+1)/(self.env.T)
        for i in range(1, len(state)):
            normalized[i] = state[i]/(self.env.totalDemand)
        return torch.tensor(normalized)
        

    def playTrainedAgent(self, advMode, iterNum):
        advProbs = torch.zeros(len(em.AdversaryModes))
        advProbs[int(advMode.value)] = 1
        game = em.Model(totalDemand=self.env.totalDemand,
                        tupleCosts=self.env.costs,
                        totalStages=self.env.T, advHistoryNum=self.env.advHistoryNum, adversaryProbs=advProbs)
        returns = np.zeros(iterNum)
        for episode in range(iterNum):

            episodeMemory = list()
            state, reward, done = game.reset()
            retu = 0

            while not done:
                prevState = state
                normPrevState = self.normalizeState(prevState)
                probs = self.neuralNetwork(normPrevState)
                distAction = Categorical(probs)
                action = distAction.sample()

                state, reward, done = game.step(
                    prevState, action.item())
                retu = retu + reward
                episodeMemory.append((normPrevState, action, reward))

            states = torch.stack([item[0] for item in episodeMemory])
            actions = torch.tensor([item[1] for item in episodeMemory])
            rewards = torch.tensor([item[2] for item in episodeMemory])

            print(f"episode {episode} return= {retu} \n\t actions: {actions}")

            # sum of the our player's rewards  rounds 0-25
            returns[episode] = retu
        plt.plot(returns)
        plt.show()
