from enum import Enum
import numpy as np
import classes as cl

class DoubleOracle():
    """
    this class is the data structure to represent the double oracle which is a bimatrix game with low-cost and high-cost strategies are trained and added to it based on the Nash equilibrium.

    """
    def __init__(self):
        cl.create_directories()

    
