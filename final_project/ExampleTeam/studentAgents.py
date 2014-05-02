from distanceCalculator import Distancer
from game import Actions
from game import Directions
from observedState import ObservedState
import numpy as np
import random

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

class QLearnAgent(BaseStudentAgent):
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
        self.last_score = None
        self.bad_ghost = None
        # self.epoch = 1
        self.Q = np.zeros((5,7,4,4,7,4,7,4,2,2))
        # number of times action a has been taken from state s
        self.k = np.zeros((5,7,4,4,7,4,7,4,2,2))
    
    def registerInitialState(self, gameState):
        """
        Do any necessary initialization
        """
        # Here, you must replace "ExampleTeamAgent" with "<YourTeamName>Agent"
        super(ExampleTeamAgent, self).registerInitialState(gameState)
        
        # Here, you may do any necessary initialization, e.g., import some
        # parameters you've learned, as in the following commented out lines
        # learned_params = cPickle.load("myparams.pkl")
        # learned_params = np.load("myparams.npy")        

    def chooseAction(self, observedState):
        '''
        A: action space: <'0:N','1:E','2:S','3:W','4:STOP'>
        B: good_dist: smallest number of steps to tracked ghost <1:1,2:2,3:3,4:4,5:5,6:6-9,7:10+>
        C: good_dir: direction to reach tracked ghost <'0:N','1:E','2:S','3:W'>
        D: good_class: predicted class of tracked ghost <0,1,2,3>

        E: bad_dist: smallest number of steps to bad ghost <1:1,2:2,3:3,4:4,5:5,6:6-9,7:10+>
        F: bad_dir: direction to reach bad ghost <'0:N','1:E','2:S','3:W'>

        G: cap_dist: smallest number of steps to best capsule in world <1:1,2:2,3:3,4:4,5:5,6:6-9,7:10+>
        H: cap_dir: direction to reach best capsule in world <'0:N','1:E','2:S','3:W'>
        I: cap_type: predicted capsule type <0,1>

        J: scared_ghost_present: <0,1>
        '''
        # process current state variables
        pacmanPosition = observedState.getPacmanPosition()
        ghost_states = observedState.getGhostStates() # states have getPosition() and getFeatures() methods
        legalActs = [a for a in observedState.getLegalPacmanActions()]
        ghost_dists = np.array([self.distancer.getDistance(pacmanPosition,gs.getPosition()) 
                              for gs in ghost_states])
        ghost_quadrants = [observedState.getGhostQuadrant(gs) for gs in ghost_states]
        ghost_features = [gs.getFeatures[0] for gs in ghost_states]

        # if we are just starting the game
        if self.last_action==None:
            # we identify the bad ghost by its first feature
            self.bad_ghost = ghost_states[ghost_quadrants.index(4)].getFeatures[0]

        # check for the spawn of a new bad ghost
        if not self.bad_ghost in ghost_features:
            for gs in ghost_states:
                if not gs.getFeatures[0] in ghost_features:
                    self.bad_ghost = gs.getFeatures[0]

                    if ghost_states[ghost_quadrants.index(4)].getFeatures[0] != self.bad_ghost:
                        raise Exception("Bad ghost identified not in quadrant 4")
        
        capsule_data = observedState.getCapsuleData()
        curr_score = observedState.getScore()
        J = observedState.scaredGhostPresent()

        # returns 0, 1, 2, or 3 corresponding to North, East, South, West
        def getDirection(pacman_pos, ghost_pos):
            x,y = ghost_pos-pacman_pos
            if x==0 and y==0:
                raise Exception("Ghost targeted in collision with Pacman")
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

        def getGoodGhost(ghost_states):
            # process features to return distance, direction, class of best ghost
            # FEATURES:
            # Whether ghost is from quadrant 4
            # Given good ghost, need features 1-8
            return good_dist, good_dir, good_class

        def getBadGhost(ghost_states):
            gs = ghost_states[ghost_features.index(self.bad_ghost)]
            bad_dist = self.distancer.getDistance(pacmanPosition,gs.getPosition())
            bad_dir = getDirection(pacmanPosition,gs.getPosition())

            return bad_dist, bad_dir

        def getBestCapsule(capsule_data):
            # process capsule locations and features to return distance, direction, type of best capsule
            return cap_dist, cap_dir, cap_type

        B,C,D = getGoodGhost(ghost_states)
        E,F = getBadGhost(ghost_states)
        G,H,I = getBestCapsule(capsule_data)

        curr_state = B,C,D,E,F,G,H,I,J

        def default_action():
            return random.choice([0,1,2,3,4])

        last_reward = curr_score - self.last_score
        
        new_action = default_action()
        if not self.last_action == None: # if we're not at the very beginning of the epoch
            max_Q = np.max(self.Q[:,curr_state])

            # if we've seen this state before, take greedy action:
            if not sum(self.Q[:,curr_state])==0:
                Q_N = self.Q[0,curr_state]
                Q_E = self.Q[1,curr_state]
                Q_S = self.Q[2,curr_state]
                Q_W = self.Q[3,curr_state]
                Q_STOP = self.Q[4,curr_state]

                new_action = np.argmax(Q_N,Q_E,Q_S,Q_W,Q_STOP)

            self.k[new_action,curr_state] += 1
            ALPHA = 1/pow(self.k[new_action,curr_state], self.ALPHA_POW)

            self.Q[self.last_action, self.last_state] += ALPHA*(last_reward+self.GAMMA*max_Q-self.Q[self.last_action, self.last_state])
        self.last_action = new_action
        self.last_state  = curr_state
        self.last_score = curr_score

        return new_action


## Below is the class students need to rename and modify
class ExampleTeamAgent(BaseStudentAgent):
    """
    An example TeamAgent. After renaming this agent so it is called <YourTeamName>Agent,
    (and also renaming it in registerInitialState() below), modify the behavior
    of this class so it does well in the pacman game!
    """
    
    def __init__(self, *args, **kwargs):
        """
        arguments given with the -a command line option will be passed here
        """
        pass # you probably won't need this, but just in case
    
    def registerInitialState(self, gameState):
        """
        Do any necessary initialization
        """
        # Here, you must replace "ExampleTeamAgent" with "<YourTeamName>Agent"
        super(ExampleTeamAgent, self).registerInitialState(gameState)
        
        # Here, you may do any necessary initialization, e.g., import some
        # parameters you've learned, as in the following commented out lines
        # learned_params = cPickle.load("myparams.pkl")
        # learned_params = np.load("myparams.npy")        
    
    def chooseAction(self, observedState):
        """
        Here, choose pacman's next action based on the current state of the game.
        This is where all the action happens.
        
        This silly pacman agent will move away from the ghost that it is closest
        to. This is not a very good strategy, and completely ignores the features of
        the ghosts and the capsules; it is just designed to give you an example.
        """
        goodCapsules = observedState.getGoodCapsuleExamples()
        pacmanPosition = observedState.getPacmanPosition()
        ghost_states = observedState.getGhostStates() # states have getPosition() and getFeatures() methods
        legalActs = [a for a in observedState.getLegalPacmanActions()]
        ghost_dists = np.array([self.distancer.getDistance(pacmanPosition,gs.getPosition()) 
                              for gs in ghost_states])
        ghost_quadrants = [observedState.getGhostQuadrant(gs) for gs in ghost_states]
        print pacmanPosition

        # find the closest ghost by sorting the distances
        closest_idx = sorted(zip(range(len(ghost_states)),ghost_dists), key=lambda t: t[1])[0][0]
        # take the action that minimizes distance to the current closest ghost
        best_action = Directions.STOP
        best_dist = -np.inf
        for la in legalActs:
            if la == Directions.STOP:
                continue
            successor_pos = Actions.getSuccessor(pacmanPosition,la)
            new_dist = self.distancer.getDistance(successor_pos,ghost_states[closest_idx].getPosition())
            if new_dist > best_dist:
                best_action = la
                best_dist = new_dist
        return best_action

