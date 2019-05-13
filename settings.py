# -*- coding: utf-8 -*-
"""
Created on Mon May 13 09:33:18 2019

Define some global variables which define how the game is played, plus the
folder structure for storing data

@author: Sheldon
"""

#%% 

# Value of Blackjack. Might be interesting to see what happens if we create
# a goal that is higher than 21?
BLACKJACK_SCORE = 21

# number of different actions available to agent
NUM_ACTIONS = 3

# the multiplier applied to rewards when agent chooses to "double down"
ODDS_BOOST = 2          

# default 4. Number points before 21 where dealer stops taking new cards
# e.g. 4 means dealer WILL take a card at 16 but will NOT at 17
DEALER_STOP = 4         

# if player doubles down, this value is added to DEALER_STOP to find a
# more conservative stopping point. 
DOUBLE_BIAS = 0         


#%% Folder structure
# location to save/load pickled Q-matrices
Q_DATA_DIR = r'Q_matrix/'            

# location to export rewards per episode per agent to   
RLIST_DATA_DIR = r'rLists/'         

# location to export validation results to       
VAL_RESULTS_DIR = r'validation_results/'    

   

