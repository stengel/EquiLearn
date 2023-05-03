from MainGame import MainGame
from MainGame import Strategy
from MainGame import StrategyType
from environmentModelBase import Model, AdversaryModes
from neuralNetworkSimple import NNBase

# const132=Strategy(StrategyType.static,name="const132",staticIndex=1)
# const95=Strategy(StrategyType.static,"const95",staticIndex=2)
# myopic=Strategy(StrategyType.static,"myopic",staticIndex=0)

mainGame=MainGame()

# mainGame._strategies.append(const132)
# mainGame._strategies.append(const95)
# mainGame._strategies.append(myopic)
# mainGame.fill_matrix()
# mainGame.write_all_matrix()

nnMyopic=NNBase(num_input=27, num_actions=50, adv_hist=0)
nnMyopic.reset()
nnMyopic.load("0,[1e-05,1][1, 10000, 1, 1],1682423487")
nn1st=Strategy(StrategyType.neural_net,"nnMyopic",neuralNet=nnMyopic,history_num=0 )
mainGame._strategies.append(nn1st)


nnConst=NNBase(num_input=27, num_actions=50, adv_hist=0)
nnConst.reset()
nnConst.load("0,[1e-05,1][1, 10000, 1, 1],1682506150")
mainGame._strategies.append(Strategy(StrategyType.neural_net,"nnConst",neuralNet=nnMyopic,history_num=0 ))

mainGame.fill_matrix()

mainGame.write_all_matrix()