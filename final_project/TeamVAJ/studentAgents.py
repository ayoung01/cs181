from distanceCalculator import Distancer
from game import Actions
from game import Directions
from observedState import ObservedState
import numpy as np
from numpy import linalg as lg
import random
import pickle
import heapq
from sklearn.externals import joblib

import sys

# less stdout.txt | grep Frobenius
# reverse search: ctrl-R

# TO RUN FOR 1 million iterations:
# set VERBOSE = False, DUMP = True
# python pacman.py -m 100001 -q -T TeamVAJ

# TO DEBUG VISUALLY:
# set VERBOSE = True, DUMP = False
# python pacman.py -T TeamVAJ

VERBOSE = True
DUMP = False # Whether we write Q matrix to file
RELEARN = True # If true we start with an empty Q matrix

SANITY_CHECK = False # run away from bad ghost if we are too close
RANDOM_DEFAULT_ACTION = False

USE_LEARNING_RESULTS = False # If false we always take default action

SAVE_FILE = 'Q0'
GAMMA = 1
ALPHA_POW = 1

if not VERBOSE:
    sys.stdout = open('stdout.txt', 'wb')
sys.stderr = open('stderr.txt', 'wb')

class BaseStudentAgent(object):
    """Superclass of agents students will write"""

    def registerInitialState(self, gameState):
        """Initializes some helper modules"""
        import __main__
        self.display = __main__._display
        self.distancer = Distancer(gameState.data.layout, False)
        self.firstMove = True

    def observationFunction(self, gameState):
        """ maps true state to observed state """
        return ObservedState(gameState)

    def getAction(self, observedState):
        """ returns action chosen by agent"""
        return self.chooseAction(observedState)

    def chooseAction(self, observedState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP

class TeamVAJAgent(BaseStudentAgent):
    """
    An example TeamAgent. After renaming this agent so it is called <YourTeamName>Agent,
    (and also renaming it in registerInitialState() below), modify the behavior
    of this class so it does well in the pacman game!
    """
    
    def __init__(self, *args, **kwargs):
        """
        arguments given with the -a command line option will be passed here
        """
        self.last_state  = None
        self.last_action = None
        self.last_score = 0
        self.bad_ghost = None
        self.good_ghost = None
        self.last_ghost_features = None
        self.step = 1
        if RELEARN:
            self.Q = np.zeros((3,4,3,4,3,4,2,4))
            self.k = np.zeros((3,4,3,4,3,4,2,4)) # num times action a has been taken from state s
        else:
            self.Q = pickle.load(open(SAVE_FILE,"rb"))
            self.k = pickle.load(open(SAVE_FILE + '_k',"rb"))
        
        
        self.ghost_predictor = joblib.load('ghost_predictor_lda.pkl')
        self.cap_predictor = joblib.load('capsule_predictor_lda.pkl')
    
    def registerInitialState(self, gameState):
        """
        Do any necessary initialization
        """
        # Here, you must replace "ExampleTeamAgent" with "<YourTeamName>Agent"
        super(TeamVAJAgent, self).registerInitialState(gameState)
        
        # Here, you may do any necessary initialization, e.g., import some
        # parameters you've learned, as in the following commented out lines
        # learned_params = cPickle.load("myparams.pkl")
        # learned_params = np.load("myparams.npy")        

    def chooseAction(self, observedState):
        '''
        good_dist: smallest number of steps to bad ghost <[0]:1,[1]:2-5,[2]:6+>
        good_dir: direction to reach tracked ghost <'0:N','1:E','2:S','3:W'>

        bad_dist: smallest number of steps to bad ghost <[0]:1,[1]:2-5,[2]:6+>
        bad_dir: direction to reach bad ghost <'0:N','1:E','2:S','3:W'>

        cap_dist: smallest number of steps to best capsule in world <[0]:1,[1]:2-5,[2]:6+>
        cap_dir: direction to reach best capsule in world <'0:N','1:E','2:S','3:W'>

        scared_ghost_present: <0,1>
        action space: <'0:N','1:E','2:S','3:W'> (we don't think stopping should ever be the optimal action)

        '''
        print '============================================='
        print 'Step number: ', self.step
        print '============================================='

        if self.step % 1000 == 0 and DUMP:
            try:
                old_Q = pickle.load(open(SAVE_FILE,"rb"))
                # Do comparison between old_Q and self.Q
                diff = lg.norm(self.Q - old_Q)
                # If we are not converged, save and keep going
                print '\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nFrobenius norm: ', diff
            except:
                pass
            f = open(SAVE_FILE,"wb")
            pickle.dump(self.Q,f)
            pickle.dump(self.k,open(SAVE_FILE+'_k',"wb"))
            f.close()

        # process current state variables
        pacmanPosition = observedState.getPacmanPosition()
        ghost_states = observedState.getGhostStates() # states have getPosition() and getFeatures() methods
        legalActs = observedState.getLegalPacmanActions()
        ghost_dists = np.array([self.distancer.getDistance(pacmanPosition,gs.getPosition()) 
                              for gs in ghost_states])
        ghost_quadrants = [observedState.getGhostQuadrant(gs) for gs in ghost_states]
        ghost_features = [gs.getFeatures()[0] for gs in ghost_states]
        directions = [Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST]


        # if we are just starting the game
        if self.last_action==None:
            # we identify the bad ghost by its first feature
            self.bad_ghost = ghost_states[ghost_quadrants.index(4)].getFeatures()[0]
            # print 'Just found our first ghost!', self.bad_ghost

        # check for the spawn of a new bad ghost
        elif not self.bad_ghost in ghost_features:
            # print 'current bad ghost: ', self.bad_ghost
            print '!!!! we don not know who the bad ghost is !!!!!'
            # print 'current ghost features: ', ghost_features
            # print 'self.last_ghost_features', self.last_ghost_features
            for gs in ghost_states:
                feat = gs.getFeatures()[0]
                if not feat in self.last_ghost_features:
                    if observedState.getGhostQuadrant(gs)==4:
                        print feat, 'is newly identified!'
                        self.bad_ghost = feat

        curr_score = observedState.getScore()
        scared_ghost_present = int(observedState.scaredGhostPresent())
        capsule_states = observedState.getCapsuleData()
        capsule_dists = np.array([self.distancer.getDistance(pacmanPosition,x[0]) 
                              for x in capsule_states])
        # if our score increased and the good ghost no longer exists, we must have eaten the good ghost!
        if curr_score - self.last_score > 0 and not self.good_ghost in ghost_features:
            if VERBOSE:
                print 'We ate the good ghost woohoo!'
            last_reward = 100
        elif curr_score - self.last_score < -1000:
            last_reward = -1000
        elif curr_score - self.last_score >1000:
            last_reward = 1200
        else:
            last_reward = -0.5
        print 'last_reward: ',last_reward


        def discretizeDistance(d):
            if d == 0:
                print "We shouldn't be caring about zero distance"
                return 0
            if d == 1:
                return 0
            if d <= 5:
                return 1
            return 2

        # return discretized distance from the Pacman to object in position pos
        def getDistance(pos):
            return discretizeDistance(self.distancer.getDistance(pacmanPosition,pos))

        # returns 0, 1, 2, or 3 corresponding to North, East, South, West
        def getDirection(pacman_pos, ghost_pos):
            x,y = ghost_pos[0]-pacman_pos[0], ghost_pos[1]-pacman_pos[1]
            if x==0 and y==0:
                # raise Exception("Object targeted in collision with Pacman")
                print 'Object colliding with Pacman'
                return random.choice([0,1,2,3])
            if abs(y) >= abs(x):
                if y >= 0:
                    return 0 # North
                else:
                    return 2 # South
            else:
                if x >=0:
                    return 1 # East
                else:
                    return 3 # West


        def getGoodGhostPos(ghost_states):
            gs_class_list = []
            gs_distance_list = []
            gs_direction_list = []
            gs_list = []
            for gs in ghost_states:
                # only need the first 8 features
                features = np.array(gs.getFeatures())[:8]
                if features[0] == self.bad_ghost:
                    # skip the bad ghost
                    continue
                if gs.getPosition() == pacmanPosition:
                    # if we are already eating the ghost, skip it
                    continue
                else:
                    # class of the good ghost
                    gs_class_list.append(self.ghost_predictor.predict(features))
                    gs_distance_list.append(getDistance(gs.getPosition()))
                    gs_direction_list.append(getDirection(pacmanPosition,gs.getPosition()))
                    gs_list.append(gs)

            # ranking of classes based on mean score, -1, 0, 1, 2
            gs_class_list = [-1 if gs_class == 3 else gs_class for gs_class in gs_class_list]

            # best ghost is the one with highest class ranking. In case of a tie in class ranking, take the ghost that's closer to the pacman
            best_gs_i = 0
            for i in xrange(1, len(gs_class_list)):
                if gs_class_list[i] > gs_class_list[best_gs_i]:
                    best_gs_i = i
                elif gs_class_list[i] == gs_class_list[best_gs_i]:
                    if gs_distance_list[i] < gs_distance_list[best_gs_i]:
                        best_gs_i = i
            best_gs = gs_list[best_gs_i]
            return best_gs.getPosition()

        def getGoodGhost(ghost_states):
            gs_class_list = []
            gs_distance_list = []
            gs_direction_list = []
            gs_list = []
            for gs in ghost_states:
                # only need the first 8 features
                features = np.array(gs.getFeatures())[:8]
                if features[0] == self.bad_ghost:
                    # skip the bad ghost
                    continue
                if gs.getPosition() == pacmanPosition:
                    # if we are already eating the ghost, skip it
                    continue
                else:
                    # class of the good ghost
                    gs_class_list.append(self.ghost_predictor.predict(features))
                    gs_distance_list.append(getDistance(gs.getPosition()))
                    gs_direction_list.append(getDirection(pacmanPosition,gs.getPosition()))
                    gs_list.append(gs)

            # ranking of classes based on mean score, -1, 0, 1, 2
            gs_class_list = [-1 if gs_class == 3 else gs_class for gs_class in gs_class_list]

            # best ghost is the one with highest class ranking. In case of a tie in class ranking, take the ghost that's closer to the pacman
            best_gs_i = 0
            for i in xrange(1, len(gs_class_list)):
                if gs_class_list[i] > gs_class_list[best_gs_i]:
                    best_gs_i = i
                elif gs_class_list[i] == gs_class_list[best_gs_i]:
                    if gs_distance_list[i] < gs_distance_list[best_gs_i]:
                        best_gs_i = i
            best_gs = gs_list[best_gs_i]
            self.good_ghost = best_gs.getFeatures()[0]
            return gs_distance_list[best_gs_i], gs_direction_list[best_gs_i]

        def getBadGhostPos(ghost_states):
            return ghost_states[ghost_features.index(self.bad_ghost)].getPosition()
        def getBadGhost(ghost_states):
            try:
                gs = ghost_states[ghost_features.index(self.bad_ghost)]
                bad_dist = getDistance(gs.getPosition())
                bad_dir = getDirection(pacmanPosition,gs.getPosition())
                if VERBOSE:
                    # print "Bad ghost position: ", gs.getPosition()
                    print 'BAD GHOST DIRECTION: ', directions[bad_dir],'BAD GHOST DISTANCE: ', bad_dist
            except:
                bad_dist = 0
                bad_dir = random.choice([0,1,2,3])

            return bad_dist, bad_dir

        #cap_class = 1 => good capsule
        #cap_class = 0 => bad capsule
        def getBestCapsulePos(capsule_states):
            capsule_states = observedState.getCapsuleData()
            capsule_dists = np.array([self.distancer.getDistance(pacmanPosition,x[0]) 
                                  for x in capsule_states])
            
            good_caps = []
            # process capsule locations and features to return distance, direction, type of best capsule
            for cs in capsule_states:
                pos = cs[0]
                features = cs[1]
                cap_class = self.cap_predictor.predict(features)
                if (cap_class):
                    good_caps.append(cs)

            # Get good capsule with best distance
            best_dist = 1000
            cap_dir = random.choice([0,1,2,3])
            best_pos = good_caps[0][0]
            for cs in good_caps:
                pos = cs[0]
                if pos==pacmanPosition:
                    continue
                if getDistance(pos) < best_dist:
                    cap_dir = getDirection(pacmanPosition, pos)
                    best_dist = getDistance(pos)
                    best_pos = pos

            # Corner case: no good capsules
            if len(good_caps) ==0:
                # handle case
                pass
            return best_pos

        def getBestCapsule(capsule_states):
            capsule_states = observedState.getCapsuleData()
            capsule_dists = np.array([self.distancer.getDistance(pacmanPosition,x[0]) 
                                  for x in capsule_states])
            
            best_cap = capsule_states[0]
            good_caps = []
            # process capsule locations and features to return distance, direction, type of best capsule
            for cs in capsule_states:
                pos = cs[0]
                features = cs[1]
                cap_class = self.cap_predictor.predict(features)
                if (cap_class):
                    good_caps.append(cs)

            # Get good capsule with best distance
            best_dist = 1000
            cap_dir = random.choice([0,1,2,3])
            for cs in good_caps:
                pos = cs[0]
                if pos==pacmanPosition:
                    continue
                if getDistance(pos) < best_dist:
                    best_cap = cs
                    cap_dir = getDirection(pacmanPosition, pos)
                    best_dist = getDistance(pos)

            # Corner case: no good capsules
            if len(good_caps) ==0:
                # handle case
                pass
            return best_dist, cap_dir

        good_dist,good_dir = getGoodGhost(ghost_states)
        bad_dist,bad_dir = getBadGhost(ghost_states)
        cap_dist,cap_dir = getBestCapsule(capsule_states)

        curr_state = good_dist,good_dir,bad_dist,bad_dir,cap_dist,cap_dir,scared_ghost_present
        
        if VERBOSE:
            print 'good ghost direction:',directions[good_dir]
            print "current state: ",curr_state
        
        def sanity_check(action):
            # if we are going to get eaten by the bad ghost, gtfo
            if getBadGhost(ghost_states)[0]<=2 and scared_ghost_present==0:
                if VERBOSE:
                    print "GTFO we're near!!!"
                try:
                    pos=ghost_states[ghost_features.index(self.bad_ghost)].getPosition()
                    best_action = random.choice(directions)
                    best_dist = -np.inf

                    for la in legalActs:
                        if la == Directions.STOP:
                            continue
                        successor_pos = Actions.getSuccessor(pacmanPosition,la)
                        new_dist = self.distancer.getDistance(successor_pos,pos)
                        if new_dist >= best_dist:
                            best_action = directions.index(la)
                            best_dist = new_dist
                        if VERBOSE:
                            print 'sanity check!', directions[action], 'changed to' , directions[best_action]

                    return best_action
                except:
                    print 'oops we messed up'
                    pass
            return action

        # returns a random legal action
        def default_action():
            # action = random.choice([0,1,2,3])
            action = good_dir
            if not scared_ghost_present:
                action = cap_dir
            elif good_dist >0 and observedState.getGhostQuadrant(ghost_states[ghost_features.index(self.bad_ghost)])!=4:
                print 'We are going after the bad ghost!'
                action = bad_dir
            # else go after good ghost


            while not directions[action] in legalActs:
                try:
                    # if we're going after a capsule, take an action that minimizes the distance to the capsule
                    best_dist = np.inf
                    for la in legalActs:
                        if la == Directions.STOP:
                            continue
                        if not scared_ghost_present:
                            successor_pos = Actions.getSuccessor(pacmanPosition,la)
                            new_dist = self.distancer.getDistance(successor_pos,getBestCapsulePos(capsule_states))
                        # if we're going after the bad ghost, take an action that minimizes distance to good ghost
                        elif good_dist==0:
                            successor_pos = Actions.getSuccessor(pacmanPosition,la)
                            new_dist = self.distancer.getDistance(successor_pos,getGoodGhostPos(ghost_states))
                        # if we're going after the bad ghost, take an action that minimizes distance to bad ghost
                        else:
                            print 'Going after bad ghost but we are stuck at a wall!'
                            successor_pos = Actions.getSuccessor(pacmanPosition,la)
                            new_dist = self.distancer.getDistance(successor_pos,getBadGhostPos(ghost_states))
                            # print la, new_dist
                        if new_dist < best_dist:
                            best_dist = new_dist
                            action = directions.index(la)
                            print 'new best distance', la, action
                    print 'Successfully circumvented the wall!', action
                except:
                    print 'Random action because we are a wall!'
                    action = random.choice([0,1,2,3])
            if SANITY_CHECK:
                return sanity_check(action)
            if RANDOM_DEFAULT_ACTION:
                action = random.choice([0,1,2,3])
                while not directions[action] in legalActs:
                    action = random.choice([0,1,2,3])
            if VERBOSE:
                print 'default action', directions[action]
            return action
        
        new_action = default_action()
        if USE_LEARNING_RESULTS:
            if not self.last_action == None: # if we're not at the very beginning of the step
                max_Q = np.max(self.Q[curr_state])

                # if we've seen this state before, take greedy action:
                print 'Q[curr_state]', self.Q[curr_state]
                if not sum(self.Q[curr_state])==0:
                    Q_N = self.Q[curr_state][0]
                    Q_E = self.Q[curr_state][1]
                    Q_S = self.Q[curr_state][2]
                    Q_W = self.Q[curr_state][3]
                    new_action = np.argmax([Q_N,Q_E,Q_S,Q_W])
                    print 'new action selected based on best Q-Value!', new_action

                self.k[curr_state][new_action] += 1
                ALPHA = 1/pow(self.k[curr_state][new_action], ALPHA_POW)
                self.Q[self.last_state][self.last_action] += ALPHA*(last_reward+GAMMA*max_Q-self.Q[self.last_state][self.last_action])
        self.last_action = new_action
        self.last_state  = curr_state
        self.last_score = curr_score
        print ghost_features
        self.last_ghost_features = ghost_features
        print str(round(float(np.count_nonzero(self.Q))*100/self.Q.size,3)) + "%"

        if not directions[new_action] in legalActs:
            if VERBOSE:
                print 'illegal action'
            new_action = default_action()
        if VERBOSE:
            print 'New action: ', directions[new_action], 'for step', self.step
        self.step+=1
        return directions[new_action]

