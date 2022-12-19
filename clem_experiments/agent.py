#base class for Agents - data and strategy
#OOP will make it easier to generalise to RL agents

class Agent:
    name = ""
    def __init__(self):
        pass

    def reset(self):
        pass

    def choose_price(self, d, cost, t): #prev opponent price is "measured environment" for RL agent
        raise NotImplementedError

#Mixed strategy
class MixedStrategy(Agent):
    #TODO: think what name of mixed strat should be?
    def __init__(self, strategies, rand, weights = None): #TODO: what is strategy type
        self.strategies = strategies
        self.current_strategy = rand #TODO: random choice. if weights is not None etc.

    def reset(self):
        pass #TODO

    def choose_price(self, d, cost, t):
        return self.current_strategy.choose_price(d, cost)

class LearningAgent(Agent):
    pass #TODO

class FiniteDifferenceAgent(LearningAgent):
    def __init__(self, agent, learnable_parameters):
        pass