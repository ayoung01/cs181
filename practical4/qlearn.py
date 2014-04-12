import numpy as np
import numpy.random as npr
import sys

from time import gmtime, strftime

from SwingyMonkey import SwingyMonkey

BINSIZE = 38 # Number of pixels per bin
GAMMA = 0.3 # Discount factor
r = 0 # Distance to neighbors for imputation

class Learner:

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.epoch = 1
        self.imputed = 1
        self.Q = np.zeros((2,600/BINSIZE+1,400/BINSIZE+1,400/BINSIZE+1))
        self.k = np.zeros((2,600/BINSIZE+1,400/BINSIZE+1,400/BINSIZE+1)) # number of times action a has been taken from state s

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.epoch += 1

    def action_callback(self, state):

        '''Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.'''

        # You might do some learning here based on the current state and the last state.

        # You'll need to take an action, too, and return it.
        # Return 0 to swing and 1 to jump.

        '''
        Q matrix: 
        ndarray of dimensions A X D x T x M
        A: <action space: 0 or 1>
        D: <pixels to next tree trunk>
        T: <screen height of bottom of tree trunk>
        M: <screen height of bottom of monkey>
        '''

        # current state
        D = state['tree']['dist'] / BINSIZE
        if D < 0:  # Disregard trees that we've passed already
            D = 0
        T = state['tree']['top'] / BINSIZE
        M = state['monkey']['bot'] / BINSIZE

        # We never want to fall off the bottom or go through the roof
        def sanity_check(action):
            # if state['monkey']['bot'] < 25: # if we're too close to the bottom, jump
            #     return 1
            # if state['monkey']['top'] > 375: # if we're too close to the top, don't jump
            #     return 0
            return action

        def default_action():
            # if state['monkey']['bot'] < 50: # if we're too close to the bottom, jump
            #     return 1 if npr.rand() < 0.95 else 0
            # if state['monkey']['top'] > 300: # if we're too close to the top, don't jump
            #     return 1 if npr.rand() < 0.05 else 0
            # if state['monkey']['bot']-state['tree']['bot'] < 50 and state['tree']['dist'] > 0 and state['tree']['dist'] < 125: # if we're way too close to hitting bottom tree trunk, jump
            #     return 1
            return 1 if npr.rand() < 0.1 else 0

        new_action = default_action()
        if not self.last_action == None:
            # previous state
            d = self.last_state['tree']['dist'] / BINSIZE
            t = self.last_state['tree']['top'] / BINSIZE
            m = self.last_state['monkey']['bot'] / BINSIZE

            max_Q = np.max(self.Q[:,D,T,M])

            # if we've never seen this state before, try to impute from closest state
            if self.k[0][D,T,M] == self.k[1][D,T,M] == 0:
                neighbors = np.zeros((600/BINSIZE+1,400/BINSIZE+1,400/BINSIZE+1))
                for i in xrange(-r,r):
                    for j in xrange(-r,r):
                        for k in xrange(-r,r):
                            try:
                                if sum(self.k[:,D+i,T+j,M+k]) > 1:
                                    new_action = 1 if self.Q[1][D+i,T+j,M+k] > self.Q[0][D+i,T+j,M+k] else 0
                                    self.imputed +=1
                                    break
                            except:
                                pass
                
            else:
            # Take greedy action
                new_action = 1 if self.Q[1][D,T,M] > self.Q[0][D,T,M] else 0

            new_action = sanity_check(new_action)
            self.k[new_action][D,T,M] += 1
            ALPHA = 1/self.k[new_action][D,T,M]
            self.Q[self.last_action][d,t,m] += ALPHA*(self.last_reward+GAMMA*max_Q-self.Q[self.last_action][d,t,m])

        new_action = sanity_check(new_action)
        self.last_action = new_action
        self.last_state  = state

        return new_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward

iters = 1000
learner = Learner()
scores = []

for ii in xrange(iters):

    # Make a new monkey object.
    swing = SwingyMonkey(sound=False,            # Don't play sounds.
                         tick_length=1,          # Make game ticks super fast.
                         # Display the epoch on screen and % of Q matrix filled
                         text="Epoch %d " % (ii) + str(round(float(np.count_nonzero(learner.Q))*100/learner.Q.size,3)) + "%", 
                         action_callback=learner.action_callback,
                         reward_callback=learner.reward_callback)

    # Loop until you hit something.
    while swing.game_loop():
        pass

    # Keep track of the score for that epoch.
    scores.append(learner.last_state['score'])
    print 'score %d' % learner.last_state['score'], str(round(float(np.count_nonzero(learner.Q))*100/learner.Q.size,3)) + "%"

    # Reset the state of the learner.
    learner.reset()

print np.mean(scores)
print learner.imputed
# np.savetxt(strftime("out/%m-%d %H:%M:%S", gmtime())+'-'+str(BINSIZE)+'-'+str(GAMMA)+".txt", scores)
