import numpy as np
import torch

class controller:
    def __init__(self):
        self.normalize = False # or True, base on your need.

    def get_input(self, state):
        """
        state is a numpy array of shape (39,) that contains
        ["step", "USS1", "USS2", "USS3", "USS4", "ESS", "E", "KF_X1", "KF_X2", 
        "Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7", "Z8", "Z9", "Z10", "Z11", 
        "Z12", "Z13", "Z14", "Z15", "Z16", "Z17", "Z18", "Z19", "Z20", "Z21", 
        "Z22", "Z23", "Z24", "Z25", "Z26", "Z27", "Z28", "Z29", "Z30"]
        step ranges from 0 to 59.
        if you set self.normalize to True, then every feature in state 
        except step will be normalized to [-1, 1]. 
        Step will be normalized to [-1, 0.09] where -1 means step = 0, 0.09 means
        step = 59.
        the expected output is a numpy array of shape (4,),
        representing actions
        """
        return np.ones(4) * 0.5
