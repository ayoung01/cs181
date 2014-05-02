from distanceCalculator import Distancer
from game import Actions
from game import Directions
from observedState import ObservedState
import numpy as np


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
        # self.Q.fill(INIT)
        # number of times action a has been taken from state s
        self.k = np.zeros((5,7,4,4,7,4,7,4,2))
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

        '''
        A: action space: <up, down, left, right, STOP>
        good_dist: smallest number of steps to tracked ghost <1,2,3,4,5,6-9,10+>
        good_dir: direction to reach tracked ghost <N, E, S, W>
        good_class: predicted class of tracked ghost <0,1,2,3>

        bad_dist: smallest number of steps to bad ghost <1,2,3,4,5,6-9,10+>
        bad_dir: direction to reach bad ghost <N, E, S, W>

        cap_dist: smallest number of steps to best capsule in world <1,2,3,4,5,6-9,10+>
        cap_dir: direction to reach best capsule in world <N, E, S, W>
        cap_type: predicted capsule type
        '''
        goodCapsules = observedState.getGoodCapsuleExamples()

        pacmanPosition = observedState.getPacmanPosition()
        ghost_states = observedState.getGhostStates() # states have getPosition() and getFeatures() methods
        legalActs = [a for a in observedState.getLegalPacmanActions()]
        ghost_dists = np.array([self.distancer.getDistance(pacmanPosition,gs.getPosition()) 
                              for gs in ghost_states])
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
        print 'dists',ghost_dists
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