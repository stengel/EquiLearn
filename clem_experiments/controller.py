from duopoly_game import DuopolyGame
from learning_guess import FiniteDifferenceGuess
from hard_coded_agents import Myopic, Guess

game = DuopolyGame(25)

def train_low_cost(learning_agent, opponent, epsilon = 1e-7):
    for _ in range(100):        
        learning_agent.increment_learnable_parameter(-epsilon)
        profit1, _ = game.run(learning_agent, opponent)
        
        learning_agent.increment_learnable_parameter(2*epsilon)
        profit2, _ = game.run(learning_agent, opponent)
        
        learning_agent.increment_learnable_parameter(-epsilon)
        learning_agent.reset()
        
        learning_agent.gradient = (profit2 - profit1) / (2*epsilon)
        learning_agent.step()

def train_high_cost(learning_agent, opponent, epsilon = 1e-7):
    for _ in range(100):        
        learning_agent.increment_learnable_parameter(-epsilon)
        _, profit1 = game.run(opponent, learning_agent)
        
        learning_agent.increment_learnable_parameter(2*epsilon)
        _, profit2 = game.run(opponent, learning_agent)
        
        learning_agent.increment_learnable_parameter(-epsilon)
        learning_agent.reset()
        
        learning_agent.gradient = (profit2 - profit1) / (2*epsilon)
        learning_agent.step()

trainee = FiniteDifferenceGuess(Guess(125, 207, 61, 125), 0.1)

nice_trainer = Myopic()
mean_trainer = Guess(130, 193, 75, 130)

trainee2 = FiniteDifferenceGuess(Guess(130, 193, 75, 130), 0.1)

for _ in range(5):
    print(game.run(trainee, trainee2))
    train_low_cost(trainee, trainee2)
    print(game.run(trainee, trainee2))
    train_high_cost(trainee2, trainee)


print(trainee.agent.start_price)
print(trainee2.agent.start_price)