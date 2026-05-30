##Polymatrix Generators
A library of polymatrix game generators to complement the existing suite of game generators
(including the GAMUT library and the bimatrix-games library). The game classes implemented are as
follows:

####Two-player Bayesian games

- **Item Bidding Auctions**: These are combinatorial auctions with item bidding. The type of a
  bidder gives the valuation function for the different subsets of the available items.
- **Multi-unit Auctions**: This is a special case of combinatorial auctions where the items being
  sold are identical. The type of a player gives the valuation function representing the player's
  marginal value for recieving a copy of an item.
- **Blotto Games**: The players in this game have a given number of soldiers that are to be assigned
  to n hills. Each player has a value for each of the different hills, which is received if the
  player has assigned more soldiers to the hill than the other player. The utility for each player
  is then the sum of the valuation gotten on all hills.
- **Adjusted Winner**: In these games, the players wish to split a set of divisible items. The
  players have preferences for the items expressed by numerical values where the sum of the values
  over all items sums up to 1. In a similar fashion to Blotto games, the players have to divide a
  number of points across the available set of items.

####Multiplayer Polymatrix games

- **Coordination/Zero-sum Games**: In this category of games, each edge of the underlying graph is
  either a coordination game (A, A) or a zero-sum game (A, -A).
- **Groupwise Zero-sum Games**: The players are partitioned into groups so that the edges going
  across groups are zero-sum while those within the same group are coordination games.
- **Strictly Competitive Games**: A two-player bimatrix game is strictly competitive if for every
  pair of mixed strategy profiles s and s', we have that: if the payoff of one player is better in s
  than s', then the payoff of the other player is worse in s than in s'.
- **Weighted Cooperation Games**: Each player chooses a colour for a set of available colours (which
  might not be the same set for each player). The payoff of a player is the number of neighbours who
  choose the same colour.

###Generating Instances:
- **Global parameters**:

        -a actions : Indicates the number of actions available to each player
        -f filename : Output the generated instace into the specified file
        -g game : Specifies which game to generate
        -r seed : Specifies a random seed for the generator

- **Item Bidding Auctions**:

        -A auction type: Specifies the auction rule, 0 - First price, 1 - Second price, and 2 -
        All-pay 
        -g Itembidding : Indicates that the itembidding generators should be used
        -S : Indicates that the valuation of player 1 and 2 are the same
        -t types : Number of types per player
        -T tie : Indicates the tie breaking rule. 0 - First player, 1 - Second player, and 2 -
        Uniformly at random
        -v val : A string representing the valuation of all items for all player types, first set of
        t valuations are for the first player and the second set of t valuations are for the second
        player. For each type, the valuation function is the first value, followed by the valuation
        per item. The following examples show how different valuation functions can be created for
        three items:
            - Unit demand '0' : '0 2 3 4' is a unit demand valuation where the player values the
              first item at 2, the second at 3 and the third at 4.
            - Single minded '1': '1 1 0 1 3' is a single minded valuation where the player has
              a value of 3 for getting both the first and third items.
            - Additive '2': '2 2 3 4' is a an additive valuation function where the player values
              the first item at 2, the second at 3 and the third at 4.
            - Budget Additive '3': '3 2 3 4 5' is a budget additive valuation function where the
              player values the first item at 2, the second at 3, the third at 4 and has a budget of 5.

- **Multi-unit Auctions**:

        -A auction type: Specifies the auction rule, 0 - Discriminatory price, 1 - Uniform price, and 2 -
        All-pay 
        -g Multiunit: Indicates that the multi-unit generators should be used
        -t types : Number of types per player
        -v val : A string representing the valuation of all items for all player types. Each value
        represents the marginal valuation for receiving the next copy of the item. Example:
            - Additive valuation : '5 5 5 5' is an additive valuation function for four items where
              the value of a single copy is 5.
            - Subadditive valuation : '10 7 7 3' is a sub-additive valuation function for four
              items.


- **Graph structure parameters**:

        -G graphname : Specifies the graph structure to be used for the game
        -m m : Parameter used to determine the size of the graph
        -n n : Parameter used to determine the size of the graph

        Examples:

        For Complete graphs, m determines the number of nodes on the graph
        $ ./pm-gen ... -G Complete -m 10 -n 1

        For Cycle graphs, m determines the number of nodes on the graph
        $ ./pm-gen ... -G Cycle -m 10 -n 1
        
        For Grid graphs, m and n represent the number of rows and columns respectively.
        $ ./pm-gen ... -G Grid -m 5 -n 2

        For Tree graphs, m and n represent the branching and depth of the tree respectively
        $ ./pm-gen ... -G Tree -m 5 -n 3

- **Coordination/Zero-sum Games**:

        $ ./pm-gen -g CoordZero [graph options] -p
        -p p : A value within range [0, 1] representing the proportion of games that are zerosum

- **Group Zerosum Games**:
        
        $ ./pm-gen -g GroupZero [graph options] -p
        -p p : Indicates the number of groups

- **Strict Competition Games**:

        $ ./pm-gen -g StrictComp [graph options]

- **Weighted Cooperation Games**:

        $ ./pm-gen -g WeightCoop [graph options] -p
        -p p : A multiplier which indicates the total number of available strategies
