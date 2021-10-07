from controller import Controller
import numpy as np
import os,neat

class player_controller(Controller):
    """ 
    Mostly from demo_controller.py.
    
    Control player in the Evoman environment.
    """
    def __init__(self, config):
        self.config = config
        
    def control(self, inputs, genome):
        """
        Partly from https://neat-python.readthedocs.io/en/latest/xor_example.html
        
        Create a neat neural network using the given genome, get outputs weights 
        from neural network and decide what actions the player must take.
        """
        # creating and activating neural network using the genome and config
        net = neat.nn.FeedForwardNetwork.create(genome, self.config)
        output = net.activate(inputs)
        
        # weight of each action
        if output[0] > 0.5:
            left = 1
        else:
            left = 0
            
        if output[1] > 0.5:
            right = 1
        else:
            right = 0
            
        if output[2] > 0.5:
            jump = 1
        else:
            jump = 0
        if output[3] > 0.5:
             shoot = 1
        else:
             shoot = 0
 
        if output[4] > 0.5:
             release = 1
        else:
             release = 0
 
        return [left, right, jump, shoot, release]