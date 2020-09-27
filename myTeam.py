# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

class ModRange:
    """A generator class for getting ranges that wrap around modulo boundaries"""
    def __init__(self, start, end, mod):
        self.current = start % mod
        self.end = end
        self.mod = mod

    def next(self):
        """Returns the next value from the range or None if range has ended"""
        value = None
        if self.current != self.end:
            value = self.current
            self.current = (self.current + 1) % self.mod
        return value

class Vectors:
    _VECTORS_ = {
        Directions.NORTH: ( 0,  1),
        Directions.SOUTH: ( 0, -1),
        Directions.EAST:  ( 1,  0),
        Directions.WEST:  (-1,  0),
        Directions.STOP:  ( 0,  0)
    }

    def newPosition(x, y, action):
        vector = Vectors._VECTORS_[action]
        dx, dy = vector
        newX = x + dx
        newY = y + dy
        return (newX, newY)

    def findNeigbours(x, y, walls, allowStop=False):
        neighbours = []
        for action in Vectors._VECTORS_:
            if action != Directions.STOP or allowStop:
                newX, newY = Vectors.newPosition(x, y, action)
                if not walls[newX][newY]:
                    neighbours.append(action)
        return neighbours

    def rePos(x, y, walls, allowStop=False):
        actions = Vectors.findNeigbours(x, y, walls, allowStop)
        positions = []
        for a in actions:
            positions.append(Vectors.newPosition(x, y, a))
        return positions

Vectors.newPosition = staticmethod(Vectors.newPosition)
Vectors.findNeigbours = staticmethod(Vectors.findNeigbours)
Vectors.rePos = staticmethod(Vectors.rePos)

class BoardEdge:
    """
    Represents an edge of the graph of the board
    """
    def __init__(self, action, start, end=None):
        self.isNode = False
        self.ends = [start, end]
        self.positions = []
        self.actions = [action]

    def weight(self):
        return len(self.positions) + 1

    def end(self, node):
        if node != self.ends[0]:
            return self.ends[0]
        else:
            return self.ends[1]

    def distances(self, position):
        index = self.positions.index(position)
        return ((self.ends[0], index + 1), (self.ends[1], len(self.positions) - index))

    def foodCount(self, foodGrid):
        count = 0
        for pos in self.positions:
            if foodGrid[pos[0]][pos[1]]:
                count += 1
        return count

    def hasFood(self, foodGrid):
        if self.foodCount(foodGrid) > 0:
            return True
        else:
            return False

class BoardNode:
    """
    Represents a junction or terminal point of the board
    """
    def __init__(self, position, exits, isRed):
        self.isNode = True
        self.onBorder = False
        self.position = position
        self.isRed = isRed
        self.exits = {}
        for action in exits:
            self.exits[action] = None

    def createEdges(self, nodes, positions, walls):
        for action in self.exits:
            # Check if edge already exists
            if self.exits[action] == None:
                edge = None
                x, y = self.position
                newPos = Vectors.newPosition(x, y, action)
                if newPos in nodes:
                    # Zero length edge
                    edge = BoardEdge(action, nodes[self.position], nodes[newPos])
                else:
                    # Build out the edge
                    edge = BoardEdge(action, nodes[self.position])
                    while newPos not in nodes:
                        edge.positions.append(newPos)
                        positions[newPos] = edge
                        x, y = newPos
                        neighbours = Vectors.findNeigbours(x, y, walls)
                        if Directions.REVERSE[edge.actions[-1]] in neighbours:
                            neighbours.remove(Directions.REVERSE[edge.actions[-1]])
                        if len(neighbours) > 0:
                            edge.actions.append(neighbours[0])
                        newPos = Vectors.newPosition(x, y, edge.actions[-1])
                    edge.ends[1] = nodes[newPos]
                # Add the edges to the nodes
                self.exits[action] = edge
                nodes[newPos].exits[Directions.REVERSE[edge.actions[-1]]] = edge

    def hasFood(self, foodGrid):
        x, y = self.position
        return foodGrid[x][y]

class BoardGraph:
    """
    A graph representation of the pacman board
    """
    def __init__(self, walls):
        self.positions = {}
        self.nodes = {}
        self.border = {True: {}, False: {}}
        borderEast = walls.width // 2
        # Create nodes for all unwalled positions that do not have exactly two
        #   unwalled neighbors
        for x in range(1, walls.width - 1):
            for y in range(1, walls.height -1):
                if not walls[x][y]:
                    neighbours = Vectors.findNeigbours(x, y, walls)
                    if len(neighbours) != 2:
                        position = (x, y)
                        node = BoardNode(position, neighbours,
                            True if x < borderEast else False)
                        self.positions[position] = node
                        self.nodes[position] = node
        # Create nodes for all positions that mark the border if they are not
        #   already nodes
        borderWest = borderEast - 1
        for y in range(1, walls.height -1):
            if not walls[borderEast][y] and not walls[borderWest][y]:
                posEast = (borderEast, y)
                if posEast not in self.nodes:
                    neighbours = Vectors.findNeigbours(borderEast, y, walls)
                    node = BoardNode(posEast, neighbours, False)
                    self.positions[posEast] = node
                    self.nodes[posEast] = node
                    self.border[False][posEast] = node
                else:
                    self.border[False][posEast] = self.nodes[posEast]
                posWest = (borderWest, y)
                if posWest not in self.nodes:
                    neighbours = Vectors.findNeigbours(borderWest, y, walls)
                    node = BoardNode(posWest, neighbours, True)
                    self.positions[posWest] = node
                    self.nodes[posWest] = node
                    self.border[True][posWest] = node
                else:
                    self.border[False][posWest] = self.nodes[posWest]
                self.nodes[posEast].onBorder = True
                self.nodes[posWest].onBorder = True
        # Create edges between nodes
        for node in self.nodes:
            self.nodes[node].createEdges(self.nodes, self.positions, walls)

class Hivemind:
    """
    The Hivemind stores a team of agents collective understanding of the board
    and any belief states held about the position of the opposing team.
    """
    def __init__( self, teamIndexes, isRed ):
        self.teamIndexes = teamIndexes
        self.enemyIndexes = None
        self.isRed = isRed
        self.board = None
        self.history = []
        self.policies = None

    def registerInitialState(self, agentIndex, gameState):
        if len(self.history) == 0:
            self.board = BoardGraph(gameState.getWalls())
            beliefs = {}
            for agentIndex in range(0, gameState.getNumAgents()):
                belief = util.Counter()
                belief[gameState.getInitialAgentPosition(agentIndex)] = 1.0
                beliefs[agentIndex] = belief
            self.history.append((gameState, beliefs))
            foodGrid = None
            if self.isRed:
                self.enemyIndexes = gameState.getBlueTeamIndices()
                foodGrid = gameState.getBlueFood()
            else:
                self.enemyIndexes = gameState.getRedTeamIndices()
                foodGrid = gameState.getRedFood()
            self.policies = ValueIterations(self.board, foodGrid, beliefs, self.enemyIndexes, self.isRed)

    def registerNewState(self, agentIndex, gameState):
        beliefs = {}
        # Update belief about position of last agent on team to act
        lastAgent = self.teamIndexes[self.teamIndexes.index(agentIndex) -1 % len(self.teamIndexes)]
        beliefs[lastAgent] = self.updateBelief(lastAgent, agentIndex, gameState, hasMoved=True)
        # Update beliefs about the position of enemy agents that have moved since
        # the last agent on the team moved
        mRange = ModRange(lastAgent + 1, agentIndex, gameState.getNumAgents())
        while True:
            agent = mRange.next()
            if agent == None:
                break
            beliefs[agent] = self.updateBelief(agent, agentIndex, gameState, hasMoved=True)
        # Update beliefs about the position of agents who have not moved
        mRange = ModRange(agentIndex, lastAgent, gameState.getNumAgents())
        while True:
            agent = mRange.next()
            if agent == None:
                break
            beliefs[agent] = self.updateBelief(agent, agentIndex, gameState)
        self.policies.updateEnemyPosVal(beliefs)
        # Update food values
        lastObservation = self.history[-1]
        lastState = lastObservation[0]
        lastFoodCount = 0
        currFoodCount = 0
        foodGrid = None
        if self.isRed:
            foodGrid = gameState.getBlueFood()
            lastFoodCount = len(lastState.getBlueFood().asList())
            currFoodCount = len(gameState.getBlueFood().asList())
        else:
            foodGrid = gameState.getRedFood()
            lastFoodCount = len(lastState.getRedFood().asList())
            currFoodCount = len(gameState.getRedFood().asList())
        if currFoodCount != lastFoodCount:
            self.policies.updateFoodValues(foodGrid)
        #Update history
        self.history.append((gameState, beliefs))

    def updateBelief(self, agentIndex, currentAgent, gameState, hasMoved=False):
        newBelief = util.Counter()
        agentPosition = gameState.getAgentPosition(agentIndex)
        if agentPosition != None:
            # We can see the agent so just store its position
            newBelief[agentPosition] = 1.0
        else:
            # We can't see the agent time to investigate
            oldState = self.history[-1]
            oldPosition = oldState[0].getAgentPosition(agentIndex)
            spottersEaten = []
            if oldPosition != None:
                # We could see the agent last time
                for index in self.teamIndexes:
                    position = oldState[0].getAgentPosition(index)
                    distance = util.manhattanDistance(oldPosition, position)
                    if distance < 5:
                        movement = util.manhattanDistance(gameState.getAgentPosition(index), position)
                        if movement > 1 and distance < 3:
                            spottersEaten.append(True)
                        else:
                            spottersEaten.append(False)
            if not all(spottersEaten):
                # One of our agents should still be able to see them so the must be at spawn
                newBelief[gameState.getInitialAgentPosition(agentIndex)] = 1.0
            else:
                # We couldn't see the agent the last time or we got sent back to spawn
                oldBelief = oldState[1][agentIndex]
                noiseReading = gameState.getAgentDistances()[agentIndex]
                for belief in oldBelief:
                    positions = []
                    if hasMoved:
                        x, y = belief
                        actions = Vectors.findNeigbours(x, y,
                            gameState.getWalls(), allowStop=True)
                        for action in actions:
                            pos = Vectors.newPosition(x, y, action)
                            positions.append((pos, util.manhattanDistance(pos,
                                gameState.getAgentPosition(currentAgent))))
                    else:
                        positions.append((belief, util.manhattanDistance(belief,
                            gameState.getAgentPosition(currentAgent))))
                    for i in range(len(positions) - 1, -1, -1):
                        if (positions[i][1] < noiseReading -6 or
                            positions[i][1] > noiseReading + 6):
                            del positions[i]
                    divisor = len(positions)
                    for pos in positions:
                        newBelief[pos[0]] += oldBelief[belief] / divisor
        newBelief.normalize()
        return newBelief





#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'HivemindAgent', second = 'HivemindAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  hivemind = Hivemind([firstIndex, secondIndex], isRed)
  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex, hivemind), eval(second)(secondIndex, hivemind)]

##########
# Agents #
##########

class HivemindAgent(CaptureAgent):

  def __init__( self, index, hivemind , timeForComputing = .1):
    """
    Lists several variables you can query:
    self.index = index for this agent
    self.red = true if you're on the red team, false otherwise
    self.distancer = distance calculator (contest code provides this)
    self.observationHistory = list of GameState objects that correspond
        to the sequential order of states that have occurred so far this game
    self.timeForComputing = an amount of time to give each turn for computing maze distances
        (part of the provided distance calculator)
    """
    # Agent index for querying state
    self.index = index

    # Whether or not you're on the red team
    self.red = None

    # Maze distance calculator
    self.distancer = None

    # A history of observations
    self.observationHistory = []

    # Access to the graphics
    self.display = None

    # Hivemind for decision purposes
    self.hivemind = hivemind
    self.lastNode = None

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)
    self.hivemind.registerInitialState(self.index, gameState)

  def chooseAction(self, gameState):
    self.hivemind.registerNewState(self.index, gameState)
    pos = gameState.getAgentPosition(self.index)
    foodCarry = gameState.getAgentState(self.index).numCarrying
    returnValues = self.hivemind.policies.returnHome
    board = self.hivemind.board
    foodValues = self.hivemind.policies.foodValues
    boardFeature = board.positions[pos]
    if boardFeature.isNode and foodCarry < 2:
        self.lastNode = boardFeature
        actions = ['Stop']
        values = [foodValues[pos]]
        for exit in boardFeature.exits:
            newPos = boardFeature.exits[exit].end(boardFeature).position
            actions.append(exit)
            values.append(foodValues[newPos])
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        action = random.choice(bestActions)
    elif boardFeature.isNode:
        self.lastNode = boardFeature
        actions = ['Stop']
        values = [returnValues[pos]]
        for exit in boardFeature.exits:
            newPos = boardFeature.exits[exit].end(boardFeature).position
            actions.append(exit)
            values.append(returnValues[newPos])
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        action = random.choice(bestActions)
    else:
        if self.lastNode is None:
            startValue = foodValues[boardFeature.ends[0].position]
            endValue = foodValues[boardFeature.ends[1].position]
            if startValue > endValue:
                self.lastNode = boardFeature.ends[1]
            else:
                self.lastNode = boardFeature.ends[0]
        posIndex = boardFeature.positions.index(pos)
        if self.lastNode is boardFeature.ends[1]:
            action = Directions.REVERSE[boardFeature.actions[posIndex]]
        else:
            action = boardFeature.actions[posIndex + 1]
    return action

class ValueIterations:
    def __init__( self, board, foodGrid, beliefs, enemyIndexes, isRed ):
        self.enemyIndexes = enemyIndexes
        self.board = board
        self.enemyPosValues = {}
        self.updateEnemyPosVal(beliefs)
        self.returnHome = {}
        self.updateReturnHome(isRed)
        self.foodValues = {}
        self.updateFoodValues(foodGrid)

    def updateFoodValues(self, foodGrid, iteration=50, discount=0.9):
        # Set initial values
        foodValue = 10
        values = {}
        for pos in self.board.nodes:
            node = self.board.nodes[pos]
            if foodGrid[pos[0]][pos[1]]:
                value = foodValue
            else:
                value = 0
            values[pos] = value
        # Iteration loop
        for i in range(iteration):
            newValues = {}
            for pos in self.board.nodes:
                node = self.board.nodes[pos]
                value = values[pos]
                valueList = [discount * value + value]
                for exit in node.exits:
                    edge = node.exits[exit]
                    endValue = values[edge.end(node).position]
                    edgeValue = edge.foodCount(foodGrid) * foodValue
                    weightedDiscount = discount ** edge.weight()
                    totalValue = weightedDiscount * (endValue + edgeValue) + value
                    valueList.append(totalValue)
                newValues[pos] = max(valueList)
            values = newValues
        self.foodValues = values

    def calcEnemyPosValue(self, beliefState, iteration=50, discount=0.9):
        agentPenalty = -100
        enemyValues = {}
        for p in self.board.positions:
            enemyValues[p] = agentPenalty*beliefState[p]

        for i in range(iteration):
            newValues = {}
            for p in self.board.positions:
                boardFeature = self.board.positions[p]
                values = [enemyValues[p]]
                if boardFeature.isNode:
                    for exit in boardFeature.exits:
                        edge = boardFeature.exits[exit]
                        newPos = None
                        if edge.weight() == 1:
                            newPos = edge.end(boardFeature).position
                        else:
                            if boardFeature is edge.ends[0]:
                                newPos = edge.positions[0]
                            else:
                                newPos = edge.positions[-1]
                        values.append(enemyValues[newPos])
                else:
                    index = boardFeature.positions.index(p)
                    length = len(boardFeature.positions)
                    if index - 1 < 0:
                        values.append(enemyValues[boardFeature.ends[0].position])
                    else:
                        values.append(enemyValues[boardFeature.positions[index - 1]])
                    if index + 1 < length:
                        values.append(enemyValues[boardFeature.positions[index + 1]])
                    else:
                        values.append(enemyValues[boardFeature.ends[1].position])
                newValues[p] = discount*max(values) + enemyValues[p]
            enemyValues = newValues
        return enemyValues

    def updateEnemyPosVal(self, beliefs):
        for agent in self.enemyIndexes:
            self.enemyPosValues[agent] = self.calcEnemyPosValue(beliefs[agent])

    def updateReturnHome(self, isRed, iteration=50, discount=0.9):
        values = {}
        for pos in self.board.nodes:
            node = self.board.nodes[pos]
            if isRed == node.isRed and node.onBorder:
                value = 10
            else:
                value = 0
            values[pos] = value
        # Iteration loop
        for i in range(iteration):
            newValues = {}
            for pos in self.board.nodes:
                node = self.board.nodes[pos]
                value = values[pos]
                valueList = [discount * value + value]
                for exit in node.exits:
                    edge = node.exits[exit]
                    endValue = values[edge.end(node).position]
                    weightedDiscount = discount ** edge.weight()
                    totalValue = weightedDiscount * endValue + value
                    valueList.append(totalValue)
                newValues[pos] = max(valueList)
            values = newValues
        self.returnHome = values
