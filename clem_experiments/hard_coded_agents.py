from agent import Agent

#basic hard coded strategies, from Bernhard's play.py
class Myopic(Agent):
    name = "Myopic"
    def __init__(self, parameters):
        super().__init__(parameters)

    def choose_price(self, d, cost, t):
        return (d+cost)*0.5

class Const(Agent):
    def __init__(self, parameters):
        super().__init__(parameters)

    def choose_price(self, d, cost, t):
        return self.parameters["c"]

class Imit(Agent):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.reset()

    def reset(self):
        self.prev_price = self.parameters["start_price"]
        self.prev_d = 200

    def choose_price(self, d, cost, t):
        prev_op_price = self.prev_price + 2*(d - self.prev_d)  #infer previous opponent price
        self.prev_price = prev_op_price
        self.prev_d = d
        return prev_op_price

#Best hard-coded strategy found
class Guess(Agent):
    def __init__(self, parameters): #start_price, aspiration_level, init_op_sales_guess, max_price, step_size, alpha
        super().__init__(parameters)
        self.reset()

    def reset(self):
        self.op_sales_guess = self.parameters["init_op_sales_guess"]
        self.prev_price = self.parameters["start_price"]
        self.prev_d = 200

    def choose_price(self, d, cost, t):
        if t == 0:
            return self.parameters["start_price"]

        next_price = 0 #placeholder
        if d >= self.parameters["aspiration_level"]:
            if self.prev_price > (d+cost)*0.5 - self.parameters["step_size"]:
                next_price = self.prev_price
            else:
                next_price = 0.6 * self.prev_price + 0.4 * ((d+cost)*0.5 - self.parameters["step_size"])
        else:
            prev_op_d = 400 - d
            prev_op_price = self.prev_price + (d - self.prev_d)  #infer previous opponent price
            self.op_sales_guess = self.parameters["alpha"] * self.op_sales_guess \
                + (1 - self.parameters["alpha"]) * (prev_op_d - prev_op_price)
            op_price_guess = prev_op_d - self.op_sales_guess
            next_price = min(op_price_guess + 2*(d - self.parameters["aspiration_level"]), self.parameters["max_price"])

        self.prev_price = next_price
        self.prev_d = d
        return next_price


# #Fight not working properly, but it is also a bad strategy so I didn't get round to fixing it. 
# class Fight(Agent):
#     def __init__(self, start_price, aspiration_level, alpha = 0.5):
#         self.start_price = start_price
#         self.aspiration_level = aspiration_level
#         self.alpha = alpha
    
#         self.reset()

#     def reset(self):
#         self.prev_price = self.start_price
#         self.prev_d = 200

#     def choose_price(self, d, cost, t):
#         if d >= self.aspiration_level:
#             self.prev_d = d
#             return self.prev_price
#         else:
#             prev_op_price = self.prev_price + (d - self.prev_d)  #infer previous opponent price
#             next_price = min(prev_op_price + 2*(d - self.aspiration_level), 125)
#             self.prev_price = next_price
#             return next_price
