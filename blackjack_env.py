# -*- coding: utf-8 -*-
"""

Environment that controls the rules of this variant of blackjack. The original
version from openAI Gym only involes two actions, stick and hit. Here we also
give the agent the option to "double down", e.g. multiply the odds by 2 on the
condition of taking one more (and no more) cards.

@author: openAI Gym
Link: https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py

Edits: Sheldon West
    - added "Double Down" as an action
    - parameterized the point at which dealer stops drawing cards
"""

# load required modules
import gym
from gym import spaces
from gym.utils import seeding

# import some global vars 
from settings import *

def cmp(a, b):
    return float(a > b) - float(a < b)

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

def draw_card(np_random):
    return int(np_random.choice(deck))

def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]

def usable_ace(hand):  # Does this hand have a usable ace?
    return 1*(1 in hand and sum(hand) + 10 <= BLACKJACK_SCORE)

def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)

def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > BLACKJACK_SCORE

def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10] 


class BlackjackEnv(gym.Env):
    """Simple blackjack environment
    Blackjack is a card game where the goal is to obtain cards that sum to as
    near as possible to 21 without going over.  They're playing against a fixed
    dealer.
    Face cards (Jack, Queen, King) have point value 10.
    Aces can either count as 11 or 1, and it's called 'usable' at 11.
    This game is placed with an infinite deck (or with replacement).
    The game starts with each (player and dealer) having one face up and one
    face down card.
    The player can request additional cards (hit=1) until they decide to stop
    (stick=0) or exceed 21 (bust).
    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust the player wins.
    If neither player nor dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.  The reward for winning is +1,
    drawing is 0, and losing is -1.
    The observation of a 3-tuple of: the players current sum,
    the dealer's one showing card (1-10 where 1 is ace),
    and whether or not the player holds a usable ace (0 or 1).
    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto.
    http://incompleteideas.net/book/the-book-2nd.html
    """
    
    def __init__(self, natural=False):
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(BLACKJACK_SCORE+11), # player's score state space
            spaces.Discrete(11), # dealer's face-up card
            spaces.Discrete(2))) # whether player has a useable card
        self.seed()

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        # Start the first game
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        
        ## 0: STICK
        # stick: take no more cards. dealer plays out their hand, and score
        if action==0: 
            done = True
            while (sum_hand(self.dealer) < (BLACKJACK_SCORE - DEALER_STOP)):  
                self.dealer.append(draw_card(self.np_random))
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1:
                reward = 1.5
                
        ## 1: HIT
        # hit: add a card to players hand and return
        elif action==1:  
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1
            else:
                done = False
                reward = 0
                
        ## 2: DOUBLE DOWN
        # double down: take exactly one more card and double the odds
        elif action==2:  
            self.player.append(draw_card(self.np_random))
            if is_bust(self.player):
                done = True
                reward = -1 * ODDS_BOOST
            else:
                # if not bust then we enter the dealer's phase of drawing cards
                # given boosted odds, we can make the dealer more conservative
                done = True
                dbl_dealer_stop = BLACKJACK_SCORE - DEALER_STOP - DOUBLE_BIAS
                while (sum_hand(self.dealer) < dbl_dealer_stop):
                    self.dealer.append(draw_card(self.np_random))
                raw_reward = cmp(score(self.player), score(self.dealer))
                reward = ODDS_BOOST * raw_reward
                if self.natural and is_natural(self.player) and reward >= 1:
                    reward = ODDS_BOOST * 1.5
            
        return self._get_obs(), reward, done, {}
        
    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player))

    def reset(self):
        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)
        return self._get_obs()
