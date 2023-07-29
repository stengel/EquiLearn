# Francisco, Sahar, Edward
# Contains environment class.

from enum import Enum
import numpy as np 
import gymnasium as gym
from gymnasium import spaces

class env(gym.Env):


    def __init__(self, total_demand, tuple_costs, total_stages, adversary_probabilities):
        
        # Actions that we can take: From 0 to 49 below the myopic price
        self.action_space = spaces.Discrete(50)
        
        # State space
        self.observation_space = spaces.MultiDiscrete([total_stages, total_demand])

        self.total_demand = total_demand
        self.costs = tuple_costs
        self.total_stages = total_stages
        self.demand_potential = None # two lists for the two players
        self.prices = None # prices over rounds
        self.profit = None  # profit in each round
        self.stage = None
        self.reward_function = self.profits
        self.initial_state = [0, total_demand/2] 
        self.done = False
        self.adversary_probabilities = adversary_probabilities


    def reset(self, seed = None, options = None):

        self.demand_potential = [[0]*(self.total_stages),[0]*(self.total_stages)] # two lists for the two players
        self.prices = [[0]*self.total_stages,[0]*self.total_stages]  # prices over rounds
        self.profit = [[0]*self.total_stages,[0]*self.total_stages]  # profit in each round
        self.demand_potential[0][0] = self.total_demand/2 # initialise first round 0
        self.demand_potential[1][0] = self.total_demand/2
        self.stage = 0
        self.done = False
        self.reset_adversary()

        return np.array(self.initial_state).astype(np.int64), {}
    

    def reset_adversary(self):

        index = np.random.choice(len(self.adversary_probabilities), 1, p = self.adversary_probabilities)
        self.adversary_mode = AdversaryModes(index)
        
        
    def step(self, action):

        """
        Transition Function. 
        Parameters:
        - action: Price
        - state: tupple in the latest stage (Demand Potential, Price)
        """

        adversary_action = self.adversary_choose_price()
        self.update_prices_profit_demand( [self.myopic() - action, adversary_action] ) 

        done = (self.stage == self.total_stages -1)

        if not done:
            new_state = [self.stage + 1, self.demand_potential[0][self.stage + 1]] 
            reward = self.reward_function(0)
        else:
            new_state = [self.stage + 1,0]
            reward = self.reward_function(0)
        
        self.stage = self.stage + 1  

        info = {}

        return (np.array(new_state).astype(np.int64), reward, done, False, info)

    def profits(self, player):
        """
        Computes profits.
        """
        return self.profit[player][self.stage]

    def update_prices_profit_demand(self, price_pair):
        """
        Updates Prices, Profit and Demand Potential Memory.
        Parameters. 
        price_pair: Pair of prices from the learning agent and adversary.
        """

        for player in [0,1]:
            price = price_pair[player]
            self.prices[player][self.stage] = price
            self.profit[player][self.stage] = (self.demand_potential[player][self.stage] - price) * (price - self.costs[player])
            if self.stage < self.total_stages - 1 :
                self.demand_potential[player][ self.stage + 1] = \
                    self.demand_potential[player][self.stage] + (price_pair[1-player] - price)/2
                

    def monopoly_price(self, player): # myopic monopoly price 
        """
            Computes Monopoly prices.
        """
        return (self.demand_potential[player][self.stage] + self.costs[player])/2 



    def adversary_choose_price(self): 
        """
            Strategy followed by the adversary.
        """

        if self.adversary_mode== AdversaryModes.constant_132:
            return self.const(player=1,price=132)
        elif self.adversary_mode== AdversaryModes.constant_95:
            return self.const(player=1,price=95)
        elif self.adversary_mode== AdversaryModes.imitation_128:
            return self.imit(player=1,firstprice=128)
        elif self.adversary_mode== AdversaryModes.imitation_132:
            return self.imit(player=1,firstprice=132)
        elif self.adversary_mode== AdversaryModes.fight_100:
            return self.fight(player=1,firstprice=100)
        elif self.adversary_mode== AdversaryModes.fight_125:
            return self.fight(player=1,firstprice=125)
        elif self.adversary_mode== AdversaryModes.fight_lb_125:
            return self.fight_lb(player=1,firstprice=125)
        elif self.adversary_mode== AdversaryModes.fight_132:
            return self.fight(player=1,firstprice=132)
        elif self.adversary_mode== AdversaryModes.fight_lb_132:
            return self.fight_lb(player=1,firstprice=132)
        elif self.adversary_mode== AdversaryModes.guess_125:
            return self.fight(player=1,firstprice=125)
        elif self.adversary_mode== AdversaryModes.guess_132:
            return self.fight(player=1,firstprice=132)
        else:
            return self.myopic(player = 1)



    def myopic(self, player = 0): 
        """
            Adversary follows Myopic strategy
        """
        return self.monopoly_price(player)    

    def const(self, player, price): # constant price strategy
        """
            Adversary follows Constant strategy
        """    
        if self.stage == self.total_stages-1:
            return self.monopoly_price(player, self.stage)
        return price

    def imit(self, player, firstprice): # price imitator strategy
        if self.stage == 0:
            return firstprice
        if self.stage == self.total_stages-1:
            return self.monopoly_price(player, self.stage)
        return self.prices[1-player][self.stage-1] 

    def fight(self, player, first_price): # simplified fighting strategy
        if self.stage == 0:
            return first_price
        if self.stage == self.total_stages-1:
            return self.monopoly_price(player, self.stage)
        #aspire = [ 207, 193 ] # aspiration level for demand potential
        aspire=[0,0]
        for i in range(2):
            aspire[i]= (self.total_demand-self.costs[player]+self.costs[1-player])/2
        
        D =self.demand_potential[player][self.stage] 
        Asp = aspire [player]
        if D >= Asp: # keep price; DANGER: price will never rise
            return self.prices[player][self.stage-1] 
        # adjust to get to aspiration level using previous
        # opponent price; own price has to be reduced by twice
        # the negative amount D - Asp to getself.demandPotential to Asp 
        P = self.prices[1-player][self.stage-1] + 2*(D - Asp) 
        # never price to high because even 125 gives good profits
        # P = min(P, 125)
        aspire_price= (self.total_demand+self.costs[0]+self.costs[1])/4
        P= min(P, int(0.95*aspire_price))

        return P

    def fight_lb(self, player, first_price):
        P = self.fight(player,first_price)
        # never price less than production cost
        P=max(P, self.costs[player])
        return P

    # sophisticated fighting strategy, compare fight()
    # estimate *sales* of opponent as their target, kept between
    # calls in global variable oppsaleguess[]. Assumed behavior
    # of opponent is similar to this strategy itself.
    oppsaleguess = [61, 75] # first guess opponent sales as in monopoly
    def guess(self, player, first_price): # predictive fighting strategy
        if self.stage == 0:
            self.oppsaleguess[0] = 61 # always same start 
            self.oppsaleguess[1] = 75 # always same start 
            return first_price

        if self.stage == self.total_stages-1:
            return self.monopoly_price(player, self.stage)
        aspire = [ 207, 193 ] # aspiration level
        D =self.demand_potential[player][self.stage] 
        Asp = aspire [player]

        if D >= Asp: # keep price, but go slightly towards monopoly if good
            pmono = self.monopoly_price(player, self.stage)
            pcurrent = self.prices[player][self.stage-1] 
            if pcurrent > pmono: # shouldn't happen
                return pmono
            if pcurrent > pmono-7: # no change
                return pcurrent
            # current low price at 60%, be accommodating towards "collusion"
            return .6 * pcurrent + .4 * (pmono-7)

        # guess current *opponent price* from previous sales
        prevsales =self.demand_potential[1-player][t-1] - self.prices[1-player][t-1] 
        # adjust with weight alpha from previous guess
        alpha = .5
        newsalesguess = alpha * self.oppsaleguess[player] + (1-alpha)*prevsales
        # update
        self.oppsaleguess[player] = newsalesguess 
        guessoppPrice = 400 - D - newsalesguess 
        P = guessoppPrice + 2*(D - Asp) 
        
        if player == 0:
            P = min(P, 125)
        if player == 1:
            P = min(P, 130)
        return P    
    
    def render(self):
        pass

    def close(self):
        pass



class AdversaryModes(Enum):
    myopic=0
    constant_132=1
    constant_95=2
    imitation_132=3
    imitation_128=4
    fight_132=5
    fight_lb_132=6
    fight_125=7
    fight_lb_125=8
    fight_100=9
    guess_132=10
    guess_125=11
    
    
    
    
