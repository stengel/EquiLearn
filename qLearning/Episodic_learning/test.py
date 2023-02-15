# Ed, Galit and Katerina 
# Relates to the Q-Learning approach.
# Testing out the Q-table against a given opponent

import numpy as np
from environment import AdversaryModes


class Test():
    

    def __init__(self, Model, Qtable, discount_factor, adversary_probabilities) -> None:

        
        self.env = Model
        self.Qtable = Qtable.Q_table 
        self.number_states=Qtable.number_states
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
        
    
    def best_responses(self):
        states = [0]* self.number_states
        best_responses = [0]* self.number_states
        for i in range(self.number_states):
            demand = int((200-self.number_states/2)) + i
            states[i] = demand
            action = np.argmax(self.Qtable[i]) + int((demand + self.env.costs[0])/2) - self.number_actions + 1
            best_responses[i] = action
        return states, best_responses
            

    def total_payoff(self):
        
        self.set_adversary()
        
        delta = 1/self.discount_factor
        utility = 0
        adversary_utility = 0
        actions = [0]*self.number_stages
        adversary_actions = [0]*self.number_stages
        demands = [0]*self.number_stages
        state_vector, reward, done = self.env.reset()
        demand = state_vector[1]
        
        for stage in range(self.number_stages):
            delta = delta * self.discount_factor
            demands[stage] = demand
            if (int(demand -(200-self.number_states/2)) > len(self.Qtable) -1):
                print("max action reached")
                demand = int((200-self.number_states/2) + len(self.Qtable) -1)
            if (int(demand -(200-self.number_states/2)) < 0):
                print("min action reached")
                demand = int((200-self.number_states/2))
            demand_index = int(demand -(200-self.number_states/2))
            action_index = np.argmax(self.Qtable[demand_index, :, stage])
            action = action_index + int((demand + self.env.costs[0])/2) - self.number_actions + 1
            actions[stage] = action
            utility += (demand-action)*(action-self.env.costs[0]) * delta
            adversary_demand = 400 - demand
            state_vector, reward, done = self.env.step(state_vector, action)
            demand = state_vector[1]
            adversary_actions[stage] = state_vector[2]
            adversary_utility += (adversary_demand-adversary_actions[stage])*(adversary_actions[stage]-self.env.costs[1]) * delta
        return utility, adversary_utility, np.transpose(actions), np.transpose(adversary_actions), np.transpose(demands)

    
    def error(self):
        
        Qtable_error = np.zeros((self.number_states, self.number_actions, self.number_stages))
        lowest_state = int(200-(self.number_states - 1)/2)
        
        for state_index in range(self.number_states):
            for action_index in range(self.number_actions):
                for stage in range(self.number_stages):
                    state = state_index + lowest_state
                    monopoly_price = int((state + self.env.costs[0])/2) 
                    action = action_index + monopoly_price - self.number_actions + 1
                    reward = (state - action) * (action - self.env.costs[0])
                    adv_action = int(self.choose_adversary_action(state))
                    next_state = int(state + (adv_action - action)/2)
                    next_state_index = next_state - lowest_state
                    if(stage==self.number_stages-1):
                        optimal_next_value = 0
                    else:
                        optimal_next_value = max(self.Qtable[next_state_index, : , stage+1])
                    new_value = (1-self.discount_factor)*reward + self.discount_factor * optimal_next_value
                    if new_value != 0:
                        Qtable_error[state_index,action_index, stage] = (new_value - self.Qtable[state_index,action_index, stage])/new_value
                    if new_value == 0 and self.Qtable[state_index,action_index, stage] != 0:
                        print("Div by zero error") # This should not occur
        return Qtable_error
    
    
    def choose_adversary_action(self, state):
        return self.myopic(self.env.costs[1],400-state)
        

    
    def myopic(self, cost, demand):
        return (cost + demand)/2
    
    def payoff(self, cost, demand, price):
        return (demand - price)*(price - cost)
       
    def update_demand(self, demand, price_pair):
        newDemand = demand + 0.5*(price_pair[1]- price_pair[0])
        return newDemand
    
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
            

    


                



        

