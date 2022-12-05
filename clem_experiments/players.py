#base class for players - data and strategy
#OOP will make it easier to generalise to RL agents

class Player:
    def __init__(self, cost, d):
        self.cost = cost
        self.start_d = d
        self.start_price = 0
        self.d = []
        self.price = []

    def start(self):
        self.d = [self.start_d]
        self.price = [self.start_price]

    def choose_price(self, op_price): #prev opponent price is "measured environment" for RL agent
        raise NotImplementedError

    def monopoly_price(self):
        if self.d:
            return (self.cost + self.d[-1]) * 0.5
        else:
            return (self.cost + self.start_d) * 0.5 #TODO: make the start effects nicer and more general

#basic hard coded strategies, from Bernhard's play.py
class Myopic(Player):
    def __init__(self, cost, d):
        super().__init__(cost, d)
        self.start_price = self.monopoly_price()

    def choose_price(self, _):
        return self.monopoly_price()

class Const(Player):
    def __init__(self, cost, d, c):
        super().__init__(cost, d)
        self.c = c #constant price
        self.start_price = c

    def choose_price(self, _):
        return self.c

class Imit(Player):
    def __init__(self, cost, d, start_price):
        super().__init__(cost, d)
        self.start_price = start_price

    def choose_price(self, op_price):
        return op_price

class Fight(Player):
    def __init__(self, cost, d, start_price, aspiration_level, alpha = 0.5):
        super().__init__(cost, d)
        self.start_price = start_price
        self.aspiration_level = aspiration_level
        self.alpha = alpha

    def choose_price(self, op_price):
        if self.d[-1] >= self.aspiration_level:
            return self.price[-1]
        else:
            return min(op_price + 2*(self.d[-1] - self.aspiration_level), 125)

class Guess(Player):
    def __init__(self, cost, d, start_price, aspiration_level, op_sale_guess, max_price, alpha = 0.5):
        super().__init__(cost, d)
        self.start_price = start_price
        self.aspiration_level = aspiration_level
        self.init_op_sale_guess = op_sale_guess
        self.op_sale_guess = op_sale_guess
        self.max_price = max_price
        self.alpha = alpha

    def start(self):
        self.d = [self.start_d]
        self.price = [self.start_price]
        self.op_sale_guess = self.init_op_sale_guess

    def choose_price(self, op_price):
        next_p = 0 #placeholder
        if self.d[-1] >= self.aspiration_level:
            if self.price[-1] > self.monopoly_price() - 7:
                next_p = self.price[-1]
            else:
                next_p = 0.6 * self.price[-1] + 0.4 * (self.monopoly_price() - 7)
        else:
            op_d = 400 - self.d[-1]
            self.op_sale_guess = self.alpha * self.op_sale_guess + (1 - self.alpha) * (op_d - op_price)
            op_price_guess = op_d - self.op_sale_guess
            next_p = min(op_price_guess + 2*(self.d[-1] - self.aspiration_level), self.max_price)
        return next_p

#Mixed strategy
class MixedStrategy(Player):
    def __init__(self, strategies, rand, weights = None): #TODO: what is strategy type
        self.strategies = strategies
        self.rand = rand #TODO: seed
        self.current_strategy = rand #TODO: random choice. if weights is not None etc.
        self.cost = self.current_strategy.cost
        self.d = self.current_strategy.d

    def choose_price(self, op_price):
        return self.current_strategy.choose_price(op_price)
