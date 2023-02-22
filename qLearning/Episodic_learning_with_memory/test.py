# Ed, Galit and Katerina 
# Relates to the Q-Learning approach.
# Testing out the Q-table against a given opponent

import numpy as np
from environment import AdversaryModes


class Test():
    

    def __init__(self, Model, Qtable, discount_factor, adversary_probabilities) -> None:

        
        self.env = Model
        self.Qtable = Qtable.Q_table 
        self.number_demands=Qtable.number_demands
        self.number_actions=Qtable.number_actions
        self.number_stages=Qtable.number_stages
        self.discount_factor = discount_factor  
        self.adversary_probabilities = adversary_probabilities
        self.adversary = None
        
    
    
    def set_adversary(self):
        options = list(range(len(self.adversary_probabilities)))
        adversary_index = int(np.random.choice(options, 1, p= self.adversary_probabilities))
        self.adversary = AdversaryModes(adversary_index)
        new_probabilities = [0]*len(self.adversary_probabilities)
        new_probabilities[adversary_index] = 1
        self.env.adversary_probabilities = new_probabilities
            

    def total_payoff(self):

        self.set_adversary()
        
        delta = 1
        utility = 0
        adversary_utility = 0
        actions = [0]*self.number_stages
        adversary_actions = [0]*self.number_stages
        demands = [0]*self.number_stages
        state_vector, reward, done = self.env.reset()
        demand = state_vector[1]
        previous_action = state_vector[2]
        
        for stage in range(self.number_stages):
            demands[stage] = demand
            if demand >= self.number_demands:
                print("max demand reached")
                demand = self.number_demands - 1
            if demand < 0:
                print("min demand reached")
                demand = 0
            demand_index = int(demand)
            action_index = np.argmax(self.Qtable[demand_index, :, previous_action, stage])
            action = action_index + int((demand + self.env.costs[0])/2) - self.number_actions + 1
            actions[stage] = action
            utility += (demand-action)*(action-self.env.costs[0]) * delta
            adversary_demand = 400 - demand
            state_vector, _, _ = self.env.step(state_vector, action, action_index)
            demand = state_vector[1]
            previous_action = state_vector[2]
            adversary_actions[stage] = state_vector[3]
            adversary_utility += (adversary_demand-adversary_actions[stage])*(adversary_actions[stage]-self.env.costs[1]) * delta
            delta *= self.discount_factor
        return utility, adversary_utility, np.transpose(actions), np.transpose(adversary_actions), np.transpose(demands)


    def q_learning(self, state_action_value, optimal_next_value, reward, alpha, gamma): 
        target = reward + gamma * optimal_next_value
        return state_action_value + alpha * (target - state_action_value)
    
    def error(self, number_tests):
        
        self.set_adversary()
        
        errors = np.zeros(number_tests)
        
        for test in range(number_tests):
            state, reward, done = self.env.reset()
            stage, demand, previous_action, _ = state
        
        
            while not done:
                if demand >= self.number_demands:
                    print("max demand reached")
                    demand = self.number_demands - 1
                if demand < 0:
                    print("min demand reached")
                    demand = 0
                demand_index = int(demand)
                action_index = np.random.randint(0, self.number_actions)
                previous_action = int(previous_action)
                action = action_index + int((demand + self.env.costs[0])/2) - self.number_actions + 1
                state, reward, done = self.env.step(state, action, action_index)
                stage, demand, previous_action, _ = state
                
                if done: 
                    optimal_next_value = 0
                else:
                    optimal_next_value = max(self.Qtable[demand,:, action_index, stage])
                
                new_value = reward + self.discount_factor * optimal_next_value
                if new_value != 0:
                    errors[test] += (new_value - self.Qtable[demand_index, action_index, previous_action, stage-1])/new_value
                if new_value == 0 and self.Qtable[demand_index,action_index, previous_action, stage-1] != 0:
                    print("Div by zero error") # This should not occur
        return errors.mean()/self.number_stages
    
        

    
    def myopic(self, cost, demand):
        return (cost + demand)/2
    
    def payoff(self, cost, demand, price):
        return (demand - price)*(price - cost)
       
    def update_demand(self, demand, price_pair):
        new_demand = demand + 0.5*(price_pair[1]- price_pair[0])
        return new_demand
    
    def utility_of_actions(self, actions):
        agent_demand = 200
        opponent_demand = 200
        total_payoff = 0
        delta = 1/self.discount_factor
        for i in range(self.number_stages):
            delta = delta * self.discount_factor
            agent_price = int(actions[i])
            opponent_price = int(self.myopic(self.env.costs[1],opponent_demand))
            total_payoff += self.payoff(self.env.costs[0],agent_demand,agent_price) * delta
            agent_demand = int(self.update_demand(agent_demand, [agent_price,opponent_price]))
            opponent_demand = 400 - agent_demand
        return total_payoff
            

    


                



        

