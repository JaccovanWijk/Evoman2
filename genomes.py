# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:31:03 2021

@author: jacco
"""
import numpy as np

class genomes():
    def __init__(self, pop_size, n_genes):
        self.all_genomes = [genome(n_genes) for i in range(pop_size)]
        
        self.population = np.zeros(pop_size)
        for i in range(pop_size):
            self.population[i] = self.all_genomes[i].get_weights()
            
        self.sigmas = np.zeros(pop_size)
        for i in range(pop_size):
            self.sigmas[i] = self.all_genomes[i].get_sigma()
        
    def set_population(self):
        return
        
    def get_population(self):
        return self.population
    
    def get_all_genomes(self):
        return self.all_genomes
    
    def get_sigmas(self):
        return self.sigmas
          

class genome():
    def __init__(self, n_genes):
        self.weights = np.random.uniform(-1, 1, n_genes)
        # TODO: Choose range values of sigma, is 1 - 3 good?
        self.sigma = np.random.uniform(1, 3, n_genes)
    
    def get_sigma(self):
        return self.sigma
    
    def get_weights(self):
        return self.weights