from distanceCalculator import Distancer
from game import Actions
from game import Directions
from observedState import ObservedState
import numpy as np
import random
from sklearn.externals import joblib

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
        self.last_state  = None
        self.last_action = None
        self.last_score = 0
        self.bad_ghost = None
        self.step = 1
        self.Q = np.zeros((7,4,7,4,2,5))
        self.k = np.zeros((7,4,7,4,2,5)) # num times action a has been taken from state s
        self.ALPHA_POW = 1
        self.GAMMA = 0.9
        # self.Q = np.zeros((7,4,7,4,7,4,2,2,5))
        # self.k = np.zeros((7,4,7,4,7,4,2,2,5)) # num times action a has been taken from state s
        self.ghost_predictor = joblib.load('ghost_predictor.pkl')
    
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
        good_dist: smallest number of steps to tracked ghost <[0]:1,[1]:2,[2]:3,[3]:4,[4]:5,[5]:6-9,[6]:10+>
        good_dir: direction to reach tracked ghost <'0:N','1:E','2:S','3:W'>
        good_class: predicted class of tracked ghost <0,1,2,3>

        bad_dist: smallest number of steps to bad ghost <[0]:1,[1]:2,[2]:3,[3]:4,[4]:5,[5]:6-9,[6]:10+>
        bad_dir: direction to reach bad ghost <'0:N','1:E','2:S','3:W'>

        cap_dist: smallest number of steps to best capsule in world <[0]:1,[1]:2,[2]:3,[3]:4,[4]:5,[5]:6-9,[6]:10+>
        cap_dir: direction to reach best capsule in world <'0:N','1:E','2:S','3:W'>
        cap_type: predicted capsule type <0,1>

        scared_ghost_present: <0,1>
        action space: <'0:N','1:E','2:S','3:W','4:STOP'>

        '''
        print 'Step number: ', self.step

        # process current state variables
        pacmanPosition = observedState.getPacmanPosition()
        ghost_states = observedState.getGhostStates() # states have getPosition() and getFeatures() methods
        legalActs = [a for a in observedState.getLegalPacmanActions()]
        ghost_dists = np.array([self.distancer.getDistance(pacmanPosition,gs.getPosition()) 
                              for gs in ghost_states])
        ghost_quadrants = [observedState.getGhostQuadrant(gs) for gs in ghost_states]
        ghost_features = [gs.getFeatures()[0] for gs in ghost_states]
        directions = [Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST,Directions.STOP]


        # if we are just starting the game
        if self.last_action==None:
            # we identify the bad ghost by its first feature
            self.bad_ghost = ghost_states[ghost_quadrants.index(4)].getFeatures()[0]

        # check for the spawn of a new bad ghost
        if not self.bad_ghost in ghost_features:
            for gs in ghost_states:
                if not gs.getFeatures[0] in ghost_features:
                    self.bad_ghost = gs.getFeatures()[0]

                    if ghost_states[ghost_quadrants.index(4)].getFeatures()[0] != self.bad_ghost:
                        raise Exception("Bad ghost identified not in quadrant 4")
        
        capsule_data = observedState.getCapsuleData()
        curr_score = observedState.getScore()
        scared_ghost_present = int(observedState.scaredGhostPresent())

        def discretizeDistance(d):
            if d==0:
                print "We shouldn't be caring about zero distance"
                return 0
            if d <=5:
                return d-1
            if d <10:
                return 5
            return 6

        # return discretized distance from the Pacman to object in position pos
        def getDistance(pos):
            return discretizeDistance(self.distancer.getDistance(pacmanPosition,pos))

        # returns 0, 1, 2, or 3 corresponding to North, East, South, West
        def getDirection(pacman_pos, ghost_pos):
            x,y = ghost_pos[0]-pacman_pos[0], ghost_pos[1]-pacman_pos[1]
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
            gs_class_list = []
            gs_distance_list = []
            gs_direction_list = []
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

            return gs_distance_list[best_gs_i], gs_direction_list[best_gs_i]

        def getBadGhost(ghost_states):
            gs = ghost_states[ghost_features.index(self.bad_ghost)]
            bad_dist = getDistance(gs.getPosition())
            bad_dir = getDirection(pacmanPosition,gs.getPosition())
            # print 'Bad ghost is at: ', gs.getPosition()
            # print 'Direction to bad ghost: ', bad_dir
            return bad_dist, bad_dir

        def getBestCapsule(capsule_data):
            # for now just get closest capsule
            # process capsule locations and features to return distance, direction, type of best capsule
            return cap_dist, cap_dir

        good_dist,good_dir = getGoodGhost(ghost_states)
        bad_dist,bad_dir = getBadGhost(ghost_states)
        # cap_dist,cap_dir,cap_type = getBestCapsule(capsule_data)

        curr_state = good_dist,good_dir,ad_dist,bad_dir,scared_ghost_present#,cap_dist,cap_dir,cap_type
        print "current state: ",curr_state

        # returns a random legal action
        def default_action():
            action = random.choice([0,1,2,3,4])
            while not directions[action] in legalActs:
                action = random.choice([0,1,2,3,4])
            return action

        last_reward = curr_score - self.last_score
        
        new_action = default_action()
        if not self.last_action == None: # if we're not at the very beginning of the step
            # print self.Q.shape
            max_Q = np.max(self.Q[curr_state])

            # if we've seen this state before, take greedy action:
            if not sum(self.Q[curr_state])==0:
                Q_N = self.Q[curr_state][0]
                Q_E = self.Q[curr_state][1]
                Q_S = self.Q[curr_state][2]
                Q_W = self.Q[curr_state][3]
                Q_STOP = self.Q[curr_state][4]

                new_action = np.argmax([Q_N,Q_E,Q_S,Q_W,Q_STOP])

            self.k[curr_state][new_action] += 1
            ALPHA = 1/pow(self.k[curr_state][new_action], self.ALPHA_POW)
            self.Q[self.last_state][self.last_action] += ALPHA*(last_reward+self.GAMMA*max_Q-self.Q[self.last_state][self.last_action])
        self.last_action = new_action
        self.last_state  = curr_state
        self.last_score = curr_score
        print str(round(float(np.count_nonzero(self.Q))*100/self.Q.size,3)) + "%"
        self.step+=1
        print 'new action: ', directions[new_action]
        if not directions[new_action] in legalActs:
            print 'Illegal action!'
            new_action = default_action()
        return directions[new_action]


# # Below is the class students need to rename and modify
# class ExampleTeamAgent(BaseStudentAgent):
#     """
#     An example TeamAgent. After renaming this agent so it is called <YourTeamName>Agent,
#     (and also renaming it in registerInitialState() below), modify the behavior
#     of this class so it does well in the pacman game!
#     """
    
#     def __init__(self, *args, **kwargs):
#         """
#         arguments given with the -a command line option will be passed here
#         """
#         pass # you probably won't need this, but just in case
    
#     def registerInitialState(self, gameState):
#         """
#         Do any necessary initialization
#         """
#         # Here, you must replace "ExampleTeamAgent" with "<YourTeamName>Agent"
#         super(ExampleTeamAgent, self).registerInitialState(gameState)
        
#         # Here, you may do any necessary initialization, e.g., import some
#         # parameters you've learned, as in the following commented out lines
#         # learned_params = cPickle.load("myparams.pkl")
#         # learned_params = np.load("myparams.npy")        
    
#     def chooseAction(self, observedState):
#         """
#         Here, choose pacman's next action based on the current state of the game.
#         This is where all the action happens.
        
#         This silly pacman agent will move away from the ghost that it is closest
#         to. This is not a very good strategy, and completely ignores the features of
#         the ghosts and the capsules; it is just designed to give you an example.
#         """
#         goodCapsules = observedState.getGoodCapsuleExamples()
#         pacmanPosition = observedState.getPacmanPosition()
#         ghost_states = observedState.getGhostStates() # states have getPosition() and getFeatures() methods
#         legalActs = [a for a in observedState.getLegalPacmanActions()]
#         ghost_dists = np.array([self.distancer.getDistance(pacmanPosition,gs.getPosition()) 
#                               for gs in ghost_states])
#         ghost_quadrants = [observedState.getGhostQuadrant(gs) for gs in ghost_states]

#         # find the closest ghost by sorting the distances
#         closest_idx = sorted(zip(range(len(ghost_states)),ghost_dists), key=lambda t: t[1])[0][0]
#         # take the action that minimizes distance to the current closest ghost
#         best_action = Directions.STOP
#         best_dist = -np.inf
#         for la in legalActs:
#             if la == Directions.STOP:
#                 continue
#             successor_pos = Actions.getSuccessor(pacmanPosition,la)
#             new_dist = self.distancer.getDistance(successor_pos,ghost_states[closest_idx].getPosition())
#             if new_dist > best_dist:
#                 best_action = la
#                 best_dist = new_dist
#         return best_action

