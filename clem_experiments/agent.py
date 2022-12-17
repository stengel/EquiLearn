#base class for Agents - data and strategy
#OOP will make it easier to generalise to RL agents

class Agent:
    name = ""
    def __init__(self):
        pass

    def choose_price(self, d, cost, t): #prev opponent price is "measured environment" for RL agent
        raise NotImplementedError

    def reset(self):
        pass

#Mixed strategy
class MixedStrategy(Agent):
    name = "" #TODO: think what name of mixed strat should be?
    def __init__(self, strategies, rand, weights = None): #TODO: what is strategy type
        self.strategies = strategies
        self.current_strategy = rand #TODO: random choice. if weights is not None etc.

    def choose_price(self, d, cost, t):
        return self.current_strategy.choose_price(d, cost)

class LearningAgent(Agent):
    pass

#basic hard coded strategies, from Bernhard's play.py
class Myopic(Agent):
    name = "Myopic"
    def __init__(self):
        pass

    def choose_price(self, d, cost, t):
        return (d+cost)*0.5

class Const(Agent):
    def __init__(self, c):
        self.c = c #constant price

    def choose_price(self, d, cost, t):
        return self.c

class Imit(Agent):
    def __init__(self, start_price):
        self.start_price = start_price

        self.reset()

    def reset(self):
        self.prev_price = self.start_price
        self.prev_d = 200

    def choose_price(self, d, cost, t):
        prev_op_price = self.prev_price + 2*(d - self.prev_d)  #infer previous opponent price
        self.prev_price = prev_op_price
        self.prev_d = d
        return prev_op_price

class Fight(Agent): #for some reason doesn't match bernhard's play.py, so wouldn't use this!
    def __init__(self, start_price, aspiration_level, alpha = 0.5):
        self.start_price = start_price
        self.aspiration_level = aspiration_level
        self.alpha = alpha
    
        self.reset()

    def reset(self):
        self.prev_price = self.start_price
        self.prev_d = 200

    def choose_price(self, d, cost, t):
        if d >= self.aspiration_level:
            self.prev_d = d
            return self.prev_price
        else:
            prev_op_price = self.prev_price + (d - self.prev_d)  #infer previous opponent price
            next_price = min(prev_op_price + 2*(d - self.aspiration_level), 125)
            self.prev_price = next_price
            return next_price

class Guess(Agent):
    def __init__(self, start_price, aspiration_level, op_sale_guess, max_price, alpha = 0.5):
        self.start_price = start_price
        self.aspiration_level = aspiration_level
        self.init_op_sale_guess = op_sale_guess
        self.max_price = max_price
        self.alpha = alpha

        self.reset()

    def reset(self):
        self.op_sale_guess = self.init_op_sale_guess
        self.prev_price = self.start_price
        self.prev_d = 200

    def choose_price(self, d, cost, t):
        if t == 0:
            return self.start_price

        next_price = 0 #placeholder
        if d >= self.aspiration_level:
            if self.prev_price > (d+cost)*0.5 - 7:
                next_price = self.prev_price
            else:
                next_price = 0.6 * self.prev_price + 0.4 * ((d+cost)*0.5 - 7)
        else:
            prev_op_d = 400 - d
            prev_op_price = self.prev_price + (d - self.prev_d)  #infer previous opponent price
            self.op_sale_guess = self.alpha * self.op_sale_guess + (1 - self.alpha) * (prev_op_d - prev_op_price)
            op_price_guess = prev_op_d - self.op_sale_guess
            next_price = min(op_price_guess + 2*(d - self.aspiration_level), self.max_price)

        self.prev_price = next_price
        self.prev_d = d
        return next_price
