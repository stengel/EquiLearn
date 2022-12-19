#base class for Agents - data and strategy
#OOP will make it easier to generalise to RL agents

class Agent:
    name = ""
    def __init__(self, parameters): #TODO parameter class
        self.parameters = parameters

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
        return self.current_strategy.choose_price(d, cost, t)

class LearningAgent(Agent):
    pass #TODO

class FiniteDifferenceAgent(LearningAgent):
    def __init__(self, agent, learnable_parameters, learning_rate):
        self.agent = agent
        self.learnable_parameters = learnable_parameters
        self.learning_rate = learning_rate
        
        self.gradients = dict.fromkeys(learnable_parameters, 0)
    
    def reset(self):
        self.agent.reset()

    def choose_price(self, d, cost, t):
        return self.agent.choose_price(d, cost, t)

    def increment_parameter(self, parameter, increment):
        self.agent.parameters[parameter] += increment

    def step(self):
        for parameter in self.learnable_parameters:
            self.increment_parameter(parameter, self.gradients[parameter] * self.learning_rate)