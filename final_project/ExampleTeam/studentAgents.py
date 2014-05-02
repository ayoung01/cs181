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
        self.last_reward = None
        # self.epoch = 1
        self.Q = np.zeros((5,7,4,4,7,4,7,4,2))
        # number of times action a has been taken from state s
        self.k = np.zeros((5,7,4,4,7,4,7,4,2))
    
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
        A: action space: <'N','E','S','W', 'STOP'>
        B: good_dist: smallest number of steps to tracked ghost <1,2,3,4,5,6-9,10+>
        C: good_dir: direction to reach tracked ghost <'N','E','S','W'>
        D: good_class: predicted class of tracked ghost <0,1,2,3>

        E: bad_dist: smallest number of steps to bad ghost <1,2,3,4,5,6-9,10+>
        F: bad_dir: direction to reach bad ghost <'N','E','S','W'>

        G: cap_dist: smallest number of steps to best capsule in world <1,2,3,4,5,6-9,10+>
        H: cap_dir: direction to reach best capsule in world <'N','E','S','W'>
        I: cap_type: predicted capsule type <0,1>

        J: scared_ghost_present: <0,1>
        '''
        
        # process current state variables
        pacmanPosition = observedState.getPacmanPosition()
        ghost_states = observedState.getGhostStates() # states have getPosition() and getFeatures() methods
        legalActs = [a for a in observedState.getLegalPacmanActions()]
        ghost_dists = np.array([self.distancer.getDistance(pacmanPosition,gs.getPosition()) 
                              for gs in ghost_states])
        capsule_data = observedState.getCapsuleData()
        J = observedState.scaredGhostPresent()

        def getGoodGhost(ghost_states):
            # process features to return distance, direction, class of best ghost
            return good_dist, good_dir, good_class

        def getBadGhost(ghost_states):
            # process features to return distance, direction of bad ghost
            return bad_dist, bad_dir

        def getBestCapsule(capsule_data):
            # process capsule locations and features to return distance, direction, type of best capsule
            return cap_dist, cap_dir, cap_type

        B,C,D = getGoodGhost(ghost_states)
        E,F = getBadGhost(ghost_states)
        G,H,I = getBestCapsule(capsule_data)

        curr_state = B,C,D,E,F,G,H,I,J

        def default_action():
            return random.choice([Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP])

        # TODO: implement reward callback
        
        new_action = default_action()
        if not self.last_action == None: # if we're not at the very beginning of the epoch
            # previous state
            # process state parameters for previous state


            max_Q = np.max(self.Q[:,B,C,D,E,F,G,H,I,J])

            # if we've seen this state before, take greedy action:
            if not sum(self.Q[:,B,C,D,E,F,G,H,I,J])==0:
                Q_N = self.Q['N',B,C,D,E,F,G,H,I,J]
                Q_S = self.Q['S',B,C,D,E,F,G,H,I,J]
                Q_W = self.Q['W',B,C,D,E,F,G,H,I,J]
                Q_E = self.Q['E',B,C,D,E,F,G,H,I,J]
                Q_STOP = self.Q['STOP',B,C,D,E,F,G,H,I,J]
                new_action = '# max of above'

            self.k[new_action,B,C,D,E,F,G,H,I,J] += 1
            ALPHA = 1/pow(self.k[new_action,B,C,D,E,F,G,H,I,J], self.ALPHA_POW)

            self.Q[self.last_action, self.last_state] += ALPHA*(self.last_reward+self.GAMMA*max_Q-self.Q[self.last_action, self.last_state])
        self.last_action = new_action
        self.last_state  = curr_state

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
        # print 'dists',ghost_dists
        print ghost_states[1].getFeatures()
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

