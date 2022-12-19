from agent import LearningAgent

#temporary, to help figure out correct structure for FiniteDifferenceAgent

#Uses finite difference gradient descent to optimise a single parameter
class FiniteDifferenceGuess(LearningAgent):
    name = "FiniteDifferenceGuess"
    def __init__(self, agent, learning_rate = 0.1):
        self.agent = agent
        self.learning_rate = learning_rate

        self.gradient = 0
    
    def reset(self):
        self.agent.reset()

    def choose_price(self, d, cost, t):
        return self.agent.choose_price(d, cost, t)

    def increment_learnable_parameter(self, increment):
        self.agent.start_price += increment

    def step(self):
        self.increment_learnable_parameter(self.gradient * self.learning_rate)
    

class FiniteDifferenceGuess2(LearningAgent):
    pass