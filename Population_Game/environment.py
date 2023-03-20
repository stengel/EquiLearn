# Katerina, Ed and Galit (and Tommy!)
# Relates to the Q-Learning approach.
# Contains DemandPotentialGame Class and the Model of the DemandPotentialGame Class.

import numpy as np  # numerical python
# printoptions: output limited to 2 digits after decimal point
np.set_printoptions(precision=2, suppress=False)


class Model():
    
    def __init__(self, total_demand, costs, total_stages, adversaries, adversary_probabilities):
        self.total_demand = total_demand
        self.costs = costs
        self.total_stages = total_stages
        self.demand_potentials = None  
        self.prices = None  
        self.profits = None  
        self.stage = None
        self.initial_state = [0, total_demand/2, 0, 0]
        self.done = False
        self.adversary_probabilities = adversary_probabilities
        self.adversaries = adversaries
        self.adversary = None
        
        
    def reset_game(self):
        """
        Method resets game memory: Demand Potential, prices, profits
        """
        self.demand_potentials = [[0]*(self.total_stages), [0]*(self.total_stages)]  
        self.prices = [[0]*self.total_stages, [0]*self.total_stages]  
        self.profits = [[0]*self.total_stages, [0]*self.total_stages]  
        self.demand_potentials[0][0] = self.total_demand / 2  
        self.demand_potentials[1][0] = self.total_demand/2
        

    def update_prices_profit_demand(self, price_pair):

        for player in [0, 1]:
            price = int(price_pair[player])
            self.prices[player][self.stage] = price
            self.profits[player][self.stage] = int((
                self.demand_potentials[player][self.stage] - price)*(price - self.costs[player]))
        if self.stage < self.total_stages-1:
                self.demand_potentials[0][self.stage + 1] = \
                    int(self.demand_potentials[0][self.stage] + (price_pair[1] - price_pair[0])/2)
                self.demand_potentials[1][self.stage + 1] = 400 - self.demand_potentials[0][self.stage + 1]
                

    def reset(self):
        reward = 0
        self.stage = 0
        self.done = False
        self.reset_game()
        self.reset_adversary()
        return self.initial_state, reward, self.done

    def reset_adversary(self):
        adversary_index = np.random.choice(range(len(self.adversaries)), 1, p= self.adversary_probabilities)[0]
        self.adversary = self.adversaries[adversary_index]

    
    def adversary_choose_price(self):
        adversary_demand_potential = int(self.demand_potentials[1][self.stage])
        number_actions = len(self.adversary[adversary_demand_potential,:,self.stage])
        if self.stage == 0:
            adversary_previous_action_index = 0
        else: 
            adversary_previous_action = int(self.prices[1][self.stage -1])
            adversary_previous_myopic = int((self.demand_potentials[1][self.stage -1] + self.costs[1])/2)
            adversary_previous_action_index = adversary_previous_action - adversary_previous_myopic + number_actions - 1
        adversary_action = int(self.adversary[adversary_demand_potential][adversary_previous_action_index][self.stage])
        return adversary_action
    

    def step(self, state, action, action_index):
        """
        Transition Function. 
        Parameters:
        - action: Price
        - state: tupple in the latest stage (stage ,Demand Potential, Adversary Action)
        """
        adversary_action = int(self.adversary_choose_price())
        self.update_prices_profit_demand([action, adversary_action])

        done = (self.stage == self.total_stages-1)

        
        if not done:
            new_state = [self.stage+1, self.demand_potentials[0][self.stage + 1], action_index, adversary_action] 
        else:
            new_state = [self.stage+1, 0, action_index, adversary_action] 

        reward = self.profits[0][self.stage]
        self.stage = self.stage + 1

        return new_state, reward, done
