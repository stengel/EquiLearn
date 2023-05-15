import BimatrixGame

from environmentModelBase import Model, Strategy, StrategyType,MixedStrategy
from neuralNetworkSimple import NNBase
import environmentModelBase as em

# const132=Strategy(StrategyType.static,name="const132",staticIndex=1)
# const95=Strategy(StrategyType.static,"const95",staticIndex=2)
# myopic=Strategy(StrategyType.static,"myopic",staticIndex=0)

# mainGame=BimatrixGame()

# mainGame._strategies.append(const132)
# mainGame._strategies.append(const95)
# mainGame._strategies.append(myopic)
# mainGame.fill_matrix()
# mainGame.write_all_matrix()

# nnMyopic=NNBase(num_input=27, num_actions=50, adv_hist=0)
# nnMyopic.reset()
# nnMyopic.load("0,[1e-05,1][1, 10000, 1, 1],1682423487")
# nn1st=Strategy(StrategyType.neural_net,nnMyopic,"nnMyopic" )
# mainGame._strategies.append(nn1st)


# nnConst=NNBase(num_input=27, num_actions=50, adv_hist=0)
# nnConst.reset()
# nnConst.load("0,[1e-05,1][1, 10000, 1, 1],1682506150")
# mainGame._strategies.append(Strategy(StrategyType.neural_net,nnConst,"nnConst132" ))

# mainGame._strategies.append(Strategy(StrategyType.static,NNorFunc=em.const,name="staticConst132",firstPrice=132))
# mainGame._strategies.append(Strategy(StrategyType.static,NNorFunc=em.myopic,name="staticMyopic"))


# mainGame._strategies.append(Strategy(StrategyType.static,NNorFunc=em.guess,name="staticGuess132",firstPrice=132))
# mainGame.fill_matrix()

# mainGame.write_all_matrix()
BimatrixGame.run_tournament(10)