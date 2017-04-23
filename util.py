# -*- coding: utf-8 -*-
"""
Created on Sun Apr 09 21:54:48 2017

@author: brynn
"""

# python libraries
import os

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib as mpl
import matplotlib.pyplot as plt

######################################################################
# global settings
######################################################################

mpl.lines.width = 2
mpl.axes.labelsize = 14


######################################################################
# classes
######################################################################

class Data :
    
    def __init__(self, X=None, y=None) :
        """
        Data class.
        
        Attributes
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        """
        
        # n = number of examples, d = dimensionality
        self.X = X
        self.y = y
    
    def load(self, filename) :
        """
        Load tsv file into X array of features and y array of labels.
        
        Parameters
        --------------------
            filename -- string, filename
        """
        
        # determine filename
        dir = os.path.dirname(__file__)
        current_working_dir = os.getcwd()

        # assumes a folder called data in the current directory
        data_dir = os.path.join(dir, 'data')
        os.chdir(data_dir)
        
        f = os.path.abspath(filename)
        
        # load data
        with open(f, 'r') as fid :
            data = np.loadtxt(fid, dtype=np.str, delimiter="\t")
        
        os.chdir(current_working_dir)
        
        # separate features and labels
        self.X = data[:,:-1]
        self.y = data[:,-1]

# helper functions
def load_data(filename) :
    """Load tsv file into Data class."""
    data = Data()
    data.load(filename)
    return data