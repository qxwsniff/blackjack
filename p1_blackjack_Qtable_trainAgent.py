# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 11:16:07 2019

Script to train an agent to play Blackjack via Q-learning. Sections relevant to experimentation:
- Global (environment) variables found in lines 419-437
- Reinforcement learning parameters found in lines 457-473

KEY SOURCES:
- Blackjack environment adapted from:
  https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py
- Q-learning implementation adapted from:
  https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

@author: Sheldon West
"""
#%% LOAD MODULES
import gym
from gym import spaces
from gym.utils import seeding

import pandas as pd
import numpy as np
import random
import datetime
import pickle
import os

from sklearn.model_selection import ParameterGrid

import matplotlib.pyplot as plt
import seaborn as sns

# load the Blackjack environment
from blackjack_env import *

#%% ANALYSIS FUNCTIONS
def mode(l):
    """ Find the mode of a given list l. """
    return max(set(l), key=l.count)

def meanR_lastN(r_list, n=200):
    """ Find mean over the last n items in a list """
    return np.mean(r_list[-n:])

def action_dict(a, short=False):
    if short:
        lookup = {0: "St", 1: "Hi", 2: "Db", 3: "Su", 4: "In"}
    else:
        lookup = {0: "Stick", 1: "Hit", 2: "Double", 3: "Surrender", 4: "Invest"}
    return lookup[a]

def show_ep(player_hand, dealer_hand, reward, ep_no, cumulative=True):
    """ Summarise the episode once it is complete """
    pl_bust = ""
    dl_bust = ""
    
    # show if anyone went bust
    if is_bust(player_hand):
        pl_bust = "XXX"
    if is_bust(dealer_hand):
        dl_bust = "XXX"
    
    # show who won
    if reward > 0:
        pl_win = "!!!"
        dl_win = ""
    elif reward < 0:
        pl_win = ""
        dl_win = "!!!"
    else:
        pl_win = "---"
        dl_win = "---"
        
    # update hands to show effects of an Ace being used as 11 (instead of 1)
    if usable_ace(player_hand):
        player_hand = convert_first_1_to_11(player_hand)
    if usable_ace(dealer_hand):
        dealer_hand = convert_first_1_to_11(dealer_hand)
        
    # display results
    print("--- SUMMARY OF EP #{} ---".format(ep_no))
    if cumulative:
        # simplest view
        print("Player:\t{} {}{}".format(np.cumsum(player_hand), pl_bust, pl_win))
        print("Dealer:\t{} {}{}".format(np.cumsum(dealer_hand), dl_bust, dl_win))
        print(player_hand)
        print(dealer_hand)
    else:
        # more detailed view, showing progression of cards from initial (visible) hands to final hands
        print("Player:\t{} >> {} >> {} {}{}".format(player_hand[:2], player_hand, sum_hand(player_hand),
                                                    pl_bust, pl_win))
        print("Dealer:\t{} >> {} >> {} {}{}".format(dealer_hand[:1], dealer_hand, sum_hand(dealer_hand),
                                                    dl_bust, dl_win))

def show_step(player_hand_old, player_hand_new, dealer_hand,
              s0, s1, a, Q0, Q1, r, d):
    """ Give a detailed view of each step in a hand """
    # current state summary and action
    print("""-----------------------
Player's old hand:\t{} (={})
Dealer's face-up card:\t{}
Player had useable:\t{}
Player action:\t\t{} ({})""".format(player_hand_old, s0[0], s0[1], s0[2], a, action_dict(a)))
    
    # effects of action
    if player_hand_old != player_hand_new:
        print("""Player got card:\t{}""".format(player_hand_new[-1]))
        
    # Dealer's eventual outcome
    print("Dealer's hand:\t{} (={})".format(dealer_hand, sum_hand(dealer_hand)))
    print("{}\n{}".format(s0, s1))
    
    print("r:{} | Done: {}".format(r, d))
    
    # show two Q matrices separately
    inspect_Q(Q0)
    inspect_Q(Q1)
    
    # show the difference
    Q_diff = Q1 - Q0
    inspect_Q(Q_diff)

def convert_first_1_to_11(l):
    """ Convert the first 1 in a hand to 11, e.g. to faciliate understanding of hands played involving an Ace """
    for idx, i in enumerate(l):
        if i == 1:
            l[idx] = 11
            break
    return l

def inspect_Q(Q_in):
    """ Inspect the Q matrix as heatmaps
    Split out each action's Q-values for all possible states"""
        
    # Heatmaps: Show side-by-side action Q-matrix, with two rows for each Usable Ace state
    color_limits = np.max([abs(np.min(Q_in)), np.max(Q_in)])
    i_labels = ['No useable ace', 'Useable ace']
    j_labels = ['Stick', 'Hit', 'Double', 'Surrender', 'Invest'][:NUM_ACTIONS]
    
    # new row of plots for each of the binary states Ace / no Ace
    for i in np.arange(0, len(i_labels)):    
        plt.figure()
        f, axes = plt.subplots(1,len(j_labels), sharex=False, sharey=False,
                                          figsize=(12,6))
        plt.title(i_labels[i])
        # show Q values for each possible action, across all hand values x initial dealer card
        for j in np.arange(0, len(j_labels)):
            # show each heat map, adding color bar for the last plot only
            showCbar = j == len(j_labels)-1
            # obtain the relevant Q values - hiding non-used states (4- & 22+ for player / 0 for dealer)
            g = sns.heatmap(Q_in[4:22,1:,i,j], cmap='seismic_r', cbar=showCbar,
                            vmin = -color_limits, vmax = color_limits, center=0, ax=axes[j])
            g.set_title("{}\n - {} - ".format(j_labels[j], i_labels[i]))
            g.set_xlabel('Dealer initial card')
            # set agent hand value range of 4-21 inclusive, dealer 2-11 inclusive
            g.set_yticklabels(np.arange(4,22), rotation=0)
            g.set_xticklabels(np.arange(1,11), rotation=0)  # where Ace = 1 for now
            # only show y label on first plot
            if j == 0:
                g.set_ylabel('Agent hand value')
                
        plt.show()

# %% LET AGENT PLAY BLACKJACK
# using code adapted from
# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0


def select_a(Q_row, policy="e", param=0.9):
    """ Select action from a given row of Q_values, for a given policy
    
    if policy == "e" then we will use e-greedy policy
    if policy == "b" then we will use boltzmann distribution
    """
    
    # choose action based on policy required
    if policy=="e":
        # use e-greedy policy, where param = epsilon
        if np.random.rand() < param:
            # pick by random choice
            a =  np.random.choice(np.arange(0, env.action_space.n))   
        else:
            # choose highest expected value action
            a = np.argmax(Q_row)
    else:
        # use boltzmann policy, where param = tau = inverse temperature
        # Based on: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-7-action-selection-strategies-for-exploration-d3a97b7cceaf
        try:
            # normalize wrt minimum values, ensuring no negatives
            Q_row_ajd = Q_row - Q_row.min()
            # calculate probabilities under boltzmann distribution
            Q_pr = np.exp(param * Q_row_ajd) / np.exp(param * Q_row_ajd).sum()
            action_value = np.random.choice(Q_pr,p=Q_pr)
            a = np.argmax(Q_pr == action_value)
        except:
            # uniform distribution if an error occurred
            print("Error in calculating Boltzmann distribution")
            a = np.random.choice(np.arange(0, env.action_space.n))                 


    return a
        

        
def doubleQ_select(Q_a, Q_b, s, policy="e", policy_param = 0.9):
    """
    'In our experiments, we calculated the average of the two Q values for each
    action and then performed e-greedy exploration with the resulting average
    Q values.' (Hado van Hasselt)
    
    Optional params:
        policy = "e" or "b" for e-greedy and boltzmann respectively
        policy_param = float, epsilon if e-greedy or inverse temperature (tau)
                       if boltzmann 
    """
    
    # take average Q values under consideration for given state
    Q_avg_row =  (Q_a[s] + Q_b[s])/2
    
    # choose action based on given policy and policy parameter
    a = select_a(Q_avg_row, policy, policy_param)
        
    return a

def doubleQ_update(Q_a, Q_b, r, s0, s1, a,
                   lr, y, prob_update_Q_a = 0.5):
    
    if np.random.rand() < prob_update_Q_a:
        # update Q_a matrix, taking copy before update for reference
        Q_old = np.copy(Q_a)
        optimal_a = np.argmax(Q_a[s1][a])
        Q_a[s0][a] = Q_a[s0][a] + lr*(r + y*Q_b[s1][optimal_a] - Q_a[s0][a])
        
    else:
        # update Q_b matrix, taking copy before update for reference
        Q_old = np.copy(Q_b)
        optimal_b = np.argmax(Q_b[s1][a])
        Q_b[s0][a] = Q_b[s0][a] + lr*(r + y*Q_a[s1][optimal_b] - Q_b[s0][a])
        
    return Q_a, Q_b, Q_old
        
def playGame(Q_a=None, Q_b = None, verbose=0,
             dbl_Q=0, lr=0.1, y=0.6,
             policy = "e", policy_param=0.7,
             num_episodes = 1, cb_over_num_eps=10000):
    
    # state space needs to be blackjack_score+11, e.g. ace on top of a score of 21
    if Q_a is None:
        Q_a = np.zeros([BLACKJACK_SCORE+11,11,2,env.action_space.n])

    # take a copy of Q matrix if we are using double Q learning
    if dbl_Q != 0:
        Q_b = np.copy(Q_a)
    
    # prepare a 'call-back' version of the Q-matrix
    # e.g. the one with best mean rewards over last N episodes
    Q_cb = Q_a.copy()
    best_meanR = -999999    # a very low bar to beat, e.g. so that a Q-matrix of 0s doesn't beat this!
    last_cb_ep = 0         # to record the last episode the call-back is made to
    a_freq = np.zeros(env.action_space.n)

    # if using Boltzmann policy, set annealing speed
    if policy=="b":
        tau_growth = (200/policy_param)**(1/num_episodes)

    #create lists to contain total rewards and steps per episode
    iList= []
    rList = []
    aList = []
    
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        
        # game will not exceed 20 moves, e.g. simple while loop with a ceiling out of caution
        while j < 20:
            j+=1

            # choose an acion per given policy        
            if dbl_Q==0:
                Q_row = Q_a[s]
                a = select_a(Q_row, policy, policy_param)  
            else:
                Q_avg_row =  (Q_a[s] + Q_b[s])/2
                a = select_a(Q_avg_row, policy, policy_param) 
                
            # log the actions chosen
            a_freq[a] += 1
            aList.append(a)

            #Get new state and reward from environment
            Q_old = np.copy(Q_a)
            player_hand_old = env.player.copy()
            s1,r,d,_ = env.step(a)
            
            #Update Q-Table with new knowledge, unless we are in a testing scenario where lrate == 0.0
            if lr > 0:
                if dbl_Q==0:
                    Q_a[s][a] = Q_a[s][a] + lr*(r + y*np.max(Q_a[s1]) - Q_a[s][a])
                else:
                    # Double-Q learning: use the second Q-matrix as estimate of expected returns
                    Q_a, Q_b, Q_old = doubleQ_update(Q_a, Q_b, r, s, s1, a, lr, y)
    
            # inspect the step & Q matrix on episode 100000 only
            if (i == 80000) & (verbose > 2):
                if ((dbl_Q==0)|(np.array_equal(Q_b, Q_old))):
                    Q_new = np.copy(Q_a)
                else:
                    Q_new = np.copy(Q_b) 
                show_step(player_hand_old, env.player, env.dealer, s, s1, a, Q_old, Q_new, r, d)
            
            # update rewards and state
            rAll += r
            s = s1

            if d == True:            
                #Reduce chance of random action as we train the model.
                
                # for e-greedy policy, reduce epsilon
                if policy=="e":
                    if policy_param >= 0.5:
                        policy_param *= 0.99999
                    else:
                        policy_param *= 0.9999
                        
                # for boltzmann policy, increase tau (inverse temperature)
                if policy=="b":
                    policy_param *= tau_growth
                
                # For each of the final 5 episodes, show the game for inspection
                if (i >= (num_episodes-5)) & (verbose > 1):
                    show_ep(env.player, env.dealer, r, i)
                
                break

        # append reward to list
        rList.append(rAll)
        
        # update the callback Q-matrix IIF better average over last N episodes
        meanR_lastN_eps = meanR_lastN(rList, n=cb_over_num_eps)
        if ((i > cb_over_num_eps) & (meanR_lastN_eps > best_meanR)):
            
            # take average of Q-values first if double Q learning
            if dbl_Q == 0:
                Q_cb = np.copy(Q_a)
            else:
                Q_cb = (Q_a + Q_b) / 2
            last_cb_ep = i
            best_meanR = meanR_lastN_eps
            
        # show some training feedback
        if verbose > 0:
            if i % 25000 == 0:
                print("Ep: {}\tMean reward over last {} episodes: {}. ({}-policy @ {})".format(i, cb_over_num_eps, meanR_lastN_eps,
                      policy, np.around(policy_param, decimals=5)))
        
    # show overall success history of agent 
    if verbose > 1:
        plt.figure(figsize=(12,4))
        plt.plot(np.cumsum(rList))
        plt.title('Performance vs. episodes')
        plt.ylabel('Total Returns')
        plt.xlabel('Episode number')
        plt.axvline(x=last_cb_ep, linestyle="--", color="black")        # show the last callback episode
        plt.show()
        
        pc_actions_taken = a_freq/a_freq.sum()
        print(pc_actions_taken)
        
    # before returning Q matrix, take average if double Q-learning
    if dbl_Q == 0:
        Q_out = np.copy(Q_a)
    else:
        Q_out = (Q_a + Q_b)/2
        
    return Q_out, rList, Q_cb, last_cb_ep
        
#%% GLOBAL VARIABLES

# import global variables from settings file
from settings import *

# flags to control what sections of code to execute
TRAINING = True       # Train new agents
TESTING = False        # Test agents - must have run on TRAINING at least once if True
LOAD_AGENTS = False   # Load agents - must have run on TRAINING at least once if True
AGENT_RESULTS_FILE = 'results_2019-03-05_0505_fix.csv' # name of CSV containing agent locations and their eval metrics

# create environment
env = BlackjackEnv()

# check required folders exist, create them if not
subfolders = [f.path[2:]+"/" for f in os.scandir() if f.is_dir() ]
for f in [Q_DATA_DIR, VAL_RESULTS_DIR, RLIST_DATA_DIR]:
    if str(f) not in subfolders:
        try:
            # create an empty folder here
            newpath = str(f)
            os.makedirs(newpath)
        except Exception as e:
            print(e)

#%% TRAIN AGENT TO PLAY BLACKJACK
# set up evaluation metric arrays
eval_metrics = []

# Set verbosity level
# 0: nothing | 1: validation metrics only | 2: validation, perf-vs-episodes, last 5 games | 3: inspect Q-matrix update
VERBOSE =2

# SET ML PARAMETERS
NUM_EPISODES_TRAIN = 500000                # number of episodes to train agent over
EVAL_LAST_N_EPS = 50000                    # evaluate agent with mean rewards over last (N) episodes  
NUM_TO_TEST = int(EVAL_LAST_N_EPS / 50)    # number of episodes to test agent over per round (learning OFF)
NUM_TEST_ROUNDS = 1000                     # number of rounds N (each of Z episodes) to test agent(s) over

grid = ParameterGrid({"dbl_Q" : [0, 2500, 5000],                 # frequency of second Q-table updates
                          "lr": [0.01, 0.05, 0.1],               # learning rate
                          "y": [0.2, 0.5, 0.8],                  # gamma
                          "e":  [0.99, 0.6, 0.2],                # epsilon
                          "num_episodes": [NUM_EPISODES_TRAIN],
                          "cb_over_num_eps": [EVAL_LAST_N_EPS],
                          "verbose": [VERBOSE]})
    
grid = ParameterGrid({"dbl_Q" : [0],                 # frequency of second Q-table updates
                          "lr": [0.01],               # learning rate
                          "y": [0.3],                  # gamma
                          "policy": "b",
                          "policy_param":  [1],     # epsilon / tau (inverse temperature)
                          "num_episodes": [NUM_EPISODES_TRAIN],
                          "cb_over_num_eps": [EVAL_LAST_N_EPS],
                          "verbose": [VERBOSE]})
    
# Start providing feedback on progress
print("---------------------------")
print("---------------------------\nCreated environment:")
print("Goal: {} | DealerStop: {} | DoubleBias: {} | DoubleBoost: {} | NumActions: {}".format(BLACKJACK_SCORE, DEALER_STOP, DOUBLE_BIAS, ODDS_BOOST, NUM_ACTIONS))
print("---------------------------")
   
# iterate through the grid search    
grid_size = len(grid)
if TRAINING:
    for p_idx, params in enumerate(grid):
        print("---------------------------\nNow training agent {}/{}".format(p_idx+1, grid_size))
        print("DblQ freq: {} | LR: {} | g: {} | pol: {} | pol_param: {}".format(params['dbl_Q'], params['lr'],
                                                              params['y'], params['policy'], np.round_(params['policy_param'], decimals=5)))
        
        # train agent using current parameters
        Q, rList, Q_CB, lastCB_ep = playGame(Q_a=None, Q_b=None, **params)
        
        # calculate evaluation metrics
        eval_final = meanR_lastN(rList, n=EVAL_LAST_N_EPS)
        eval_cb = meanR_lastN(rList[:lastCB_ep], n=EVAL_LAST_N_EPS)
        
        if VERBOSE>0:
            print("---------------------------\nValidation metrics\n---------------------------")
            print("After final episode, agent achieved mean reward of {} over last {} episodes".format(eval_final, EVAL_LAST_N_EPS))
            print("After episode {}, agent achieved mean reward of {} over last {} episodes".format(lastCB_ep, eval_cb, EVAL_LAST_N_EPS))
        
        # Pickle Q matrices for later user
        exportMe = True
        if exportMe:
            print("---------------------------\nPickling agents")
            export_name = 'Qmatrix_expID_{}_Fi_DblQ{}_lr{}_g{}_e{}.p'.format(p_idx, params['dbl_Q'], params['lr'],
                                                                        params['y'], params['policy'], params['policy_param'])
            name_fi = Q_DATA_DIR + export_name
            pickle.dump(Q , open(name_fi, "wb" ) )
            
            # Pickle the call back too, referencing the call-back episode in filename
            export_name_cb = 'Qmatrix_expID_{}_CB{}_DblQ{}_lr{}_g{}_e{}.p'.format(p_idx, lastCB_ep,
                                                                                      params['dbl_Q'], params['lr'],
                                                                                      params['y'], params['policy'], params['policy_param'])
            name_cb = Q_DATA_DIR + export_name_cb 
            pickle.dump(Q_CB, open(name_cb, "wb" ) )
        
            # pickle the rLists using for later analysis (same filenames but different folder)
            pickle.dump(rList , open(RLIST_DATA_DIR + export_name, "wb" ) )
            pickle.dump(rList[:lastCB_ep], open(RLIST_DATA_DIR + export_name_cb, "wb" ) )
            
            
        # form Vector for validation metrics, for both final agent and call-back agent
        eval_vec_fi = [name_fi, params['dbl_Q'], params['lr'], params['y'], params['policy'], params['policy_param'],
                       params['num_episodes'], params['cb_over_num_eps'], 0, eval_final]
        eval_vec_cb = [name_cb, params['dbl_Q'], params['lr'], params['y'], params['policy'], params['policy_param'],
                       params['num_episodes'], params['cb_over_num_eps'], lastCB_ep, eval_cb]
        eval_metrics.append(eval_vec_fi)
        eval_metrics.append(eval_vec_cb)
     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#%% VALIDATION
# Export the overall validation metrics table
if TRAINING:
    val_cols = ['agent_loc', 'dbl_Q', 'lr', 'y', 'policy', 'policy_param', 'training_eps', 'lastN', 'CB_index', 'meanR_lastN']
    val_df = pd.DataFrame(eval_metrics, columns=val_cols)
    timestamp = datetime.datetime.today().strftime('%Y-%m-%d_%H%M')
    val_df.to_csv('{}results_{}.csv'.format(VAL_RESULTS_DIR, timestamp), index=False)






#%% TESTING OF TOP 2
# Test the two agents with best training validation metrics
if not(LOAD_AGENTS):
    val_df = val_df.sort_values(by='meanR_lastN', ascending=False)
else:
    # load previous results and agent locations 
    RESULTS_LOC = VAL_RESULTS_DIR + AGENT_RESULTS_FILE
    val_df = pd.read_csv(RESULTS_LOC)
    val_df = val_df.sort_values(by='meanR_lastN', ascending=False)

# find the top two agents by meanR over lastN episodes
Q_top1_name = val_df.agent_loc.values[0]
Q_top2_name = val_df.agent_loc.values[1]

# load these agents
Q_top1 = pickle.load( open(Q_top1_name, "rb" ))
Q_top2 = pickle.load( open(Q_top2_name, "rb" ))

# Evalate two agents with learning switched OFF. Increase verbosity to study differences
qt1_rLists = []
qt2_rLists = []
qt1_rListsAll = []
qt2_rListsAll = []

if TESTING:   
    print("---------------------------\nTesting top 2 agents (Learning OFF)\n---------------------------")
    for nr in np.arange(0, NUM_TEST_ROUNDS):
        print("#1 VALIDATION AGENT PERFORMANCE (round {}/{}):".format(nr, NUM_TEST_ROUNDS))
        _, rList_Qt1, _2, lastCB_ep_CB = playGame(Q_in=Q_top1, verbose=0, lr=0.0, y=0.0, e=0.0, num_episodes=NUM_TO_TEST)
        print("#2 VALIDATION AGENT PERFORMANCE (round {}/{}):".format(nr, NUM_TEST_ROUNDS))
        _, rList_Qt2, _2, lastCB_ep_Fi = playGame(Q_in=Q_top2, verbose=0, lr=0.0, y=0.0, e=0.0, num_episodes=NUM_TO_TEST)
    
        # append the new rewards
        qt1_rLists.append(sum(rList_Qt1))
        qt2_rLists.append(sum(rList_Qt2))
        qt1_rListsAll.append(rList_Qt1) 
        qt2_rListsAll.append(rList_Qt2)
    
    if np.mean(qt1_rLists)  > np.mean(qt2_rLists):
        print("The #1 validation agent was shown to outperform the #2 validation agent")
    else:
        print("The #2 validation agent was shown to outperform the #1 validation agent")        
    
    # show summary of test score
    print("#1 mean: {}\t\tstD: {}\n#2 mean: {}\t\tstD: {}".format(np.mean(qt1_rLists), np.std(qt1_rLists),
                                                            np.mean(qt2_rLists), np.std(qt2_rLists)))
    print("#1 scores: {}\n#2 scores: {}".format(qt1_rLists, qt2_rLists))    

    # visualise results
    plt.figure()
    plt.hist(qt1_rLists, alpha=0.4, label='rank1', bins=20)
    plt.hist(qt2_rLists, alpha=0.4, label='rank2', bins=20)
    plt.axvline(x=0, linestyle="--", color="k")
    plt.title('Testing scores for top two agents')
    plt.xlabel('Total rewards')
    plt.ylabel('freq')
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 0.85))
    
    broke_even_1 = sum([x>=0 for x in qt1_rLists])
    broke_even_2 = sum([x>=0 for x in qt2_rLists])
    print("Agent 1 break-even rate:\t{}".format(broke_even_1/NUM_TEST_ROUNDS))
    print("Agent 2 break-even rate:\t{}".format(broke_even_2/NUM_TEST_ROUNDS))
    
    # show the last two rounds of performance on one chart
    plt.figure() #(figsize=(12,6))
    plt.title('Performance over final 200 testing games')
    plt.plot(np.cumsum(rList_Qt1), label='rank1')
    plt.plot(np.cumsum(rList_Qt2), label='rank2')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative rewards')
    plt.legend()