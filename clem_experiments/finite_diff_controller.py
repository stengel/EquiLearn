from duopoly_game import DuopolyGame
from agent import FiniteDifferenceAgent
from hard_coded_agents import Myopic, Imit, Guess

game = DuopolyGame(25)

def train_low_cost(learning_agent, opponent, epsilon = 1e-4):
    for _ in range(100):
        profit, _ = game.run(learning_agent, opponent)

        for parameter in learning_agent.learnable_parameters:
            learning_agent.increment_parameter(parameter, epsilon)
            profit_delta, _ = game.run(learning_agent, opponent)
            learning_agent.gradients[parameter] = (profit_delta - profit) / epsilon
            learning_agent.increment_parameter(parameter, -epsilon)

        learning_agent.reset()
        learning_agent.step()

def train_high_cost(learning_agent, opponent, epsilon = 1e-5):
    for _ in range(100):
        _, profit = game.run(opponent, learning_agent)

        for parameter in learning_agent.learnable_parameters:
            learning_agent.increment_parameter(parameter, epsilon)
            _, profit_delta, = game.run(opponent, learning_agent)
            learning_agent.gradients[parameter] = (profit_delta - profit) / epsilon
            learning_agent.increment_parameter(parameter, -epsilon)
        learning_agent.reset()
        learning_agent.step()

low_cost = FiniteDifferenceAgent(Imit({"start_price": 110}), ["start_price"], 0.1)
high_cost = FiniteDifferenceAgent(Imit({"start_price": 130}), ["start_price"], 0.1)


print(game.run(low_cost, high_cost))
train_low_cost(low_cost, high_cost)
print(game.run(low_cost, high_cost))
train_high_cost(high_cost, low_cost)
print(game.run(low_cost, high_cost))


print(low_cost.agent.parameters)
print(high_cost.agent.parameters)