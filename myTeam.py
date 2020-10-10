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
from game import Directions, Agent
import game
import distanceCalculator

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
        if node == self.ends[1]:
            return self.ends[0]
        elif node == self.ends[0]:
            return self.ends[1]
        else:
            return None

    def calcAgentProb(self, belief):
        prob = 0
        for pos in self.positions:
            prob += belief[pos]
        return prob

    def distances(self, position):
        index = self.positions.index(position)
        return ((self.ends[0], index + 1), (self.ends[1], len(self.positions) - index))

    def neighbouringFood(self, foodGrid):
        food = self.hasFood(foodGrid)
        for end in self.ends:
            if end.hasFood(foodGrid):
                food = True
        return food

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

    def isDeadEnd(self):
        deadEnd = False
        for end in self.ends:
            if end.isDeadEnd():
                deadEnd = True
        return deadEnd

    def oneAway(self, position):
        positions = []
        index = self.positions.index(position)
        if index == 0:
            positions.append(self.ends[0].position)
        else:
            positions.append(self.positions[index - 1])
        if index == len(self.positions) - 1:
            positions.append(self.ends[1].position)
        else:
            positions.append(self.positions[index + 1])
        return positions

    def isRed(self):
        colours = [end.isRed() for end in self.ends]
        if all(colours):
            return True
        elif not any(colours):
            return False
        else:
            return None

class BoardNode:
    """
    Represents a junction or terminal point of the board
    """
    def __init__(self, position, exits, red):
        self.isNode = True
        self.onBorder = False
        self.position = position
        self.red = red
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

    def calcAgentProb(self, belief):
        return belief[self.position]

    def hasFood(self, foodGrid):
        x, y = self.position
        return foodGrid[x][y]

    def neighbouringFood(self, foodGrid):
        food = self.hasFood(foodGrid)
        for exit in self.exits:
            if self.exits[exit].hasFood(foodGrid):
                food = True
        return food

    def isDeadEnd(self):
        return False if len(self.exits) > 1 else True

    def oneAway(self, position):
        positions = []
        for exit in self.exits:
            edge = self.exits[exit]
            if edge.weight() == 1:
                positions.append(edge.end(self).position)
            elif self == edge.ends[0]:
                positions.append(edge.positions[0])
            elif self == edge.ends[1]:
                positions.append(edge.positions[-1])
        return positions

    def isRed(self):
        return self.red

class BoardGraph:
    """
    A graph representation of the pacman board
    """
    def __init__(self, walls):
        self.positions = {}
        self.nodes = {}
        self.border = {}
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
                    self.border[posEast] = node
                else:
                    self.border[posEast] = self.nodes[posEast]
                posWest = (borderWest, y)
                if posWest not in self.nodes:
                    neighbours = Vectors.findNeigbours(borderWest, y, walls)
                    node = BoardNode(posWest, neighbours, True)
                    self.positions[posWest] = node
                    self.nodes[posWest] = node
                    self.border[posWest] = node
                else:
                    self.border[posWest] = self.nodes[posWest]
                self.nodes[posEast].onBorder = True
                self.nodes[posWest].onBorder = True
        # Create edges between nodes
        for node in self.nodes:
            self.nodes[node].createEdges(self.nodes, self.positions, walls)

class ValueIterations:
    def __init__( self, foodGrid, beliefs, hivemind ):
        self.hivemind = hivemind
        self.enemyPosValues = {}
        self.updateEnemyPosVal(beliefs)
        self.returnHome = {}
        self.updateReturnHome()
        self.foodValues = {}
        self.updateFoodValues(foodGrid)
        self.huntValue = {}
        self.updateHuntValue(beliefs)

    def updateFoodValues(self, foodGrid, iteration=50, discount=0.9):
        # Set initial values
        foodValue = 10
        values = {}
        for pos in self.hivemind.board.nodes:
            node = self.hivemind.board.nodes[pos]
            if foodGrid[pos[0]][pos[1]]:
                value = foodValue
            else:
                value = 0
            values[pos] = value
        # Iteration loop
        for i in range(iteration):
            newValues = {}
            for pos in self.hivemind.board.nodes:
                node = self.hivemind.board.nodes[pos]
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
        agentPenalty = 10
        enemyValues = {}
        for p in self.hivemind.board.positions:
            boardFeature = self.hivemind.board.positions[p]
            if not boardFeature.isNode:
                boardFeature = boardFeature.ends[0]
            if boardFeature.isRed() == self.hivemind.isRed:
                enemyValues[p] = agentPenalty*beliefState[p]
            else:
                forecast = self.hivemind.forecastBelief(beliefState)
                enemyValues[p] = -agentPenalty*forecast[p]

        for i in range(iteration):
            newValues = {}
            for p in self.hivemind.board.positions:
                boardFeature = self.hivemind.board.positions[p]
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
        for agent in self.hivemind.enemyIndexes:
            self.enemyPosValues[agent] = self.calcEnemyPosValue(beliefs[agent])

    def updateReturnHome(self, iteration=50, discount=0.9):
        values = {}
        for pos in self.hivemind.board.nodes:
            node = self.hivemind.board.nodes[pos]
            if self.hivemind.isRed == node.isRed() and node.onBorder:
                value = 10
            else:
                value = 0
            values[pos] = value
        # Iteration loop
        for i in range(iteration):
            newValues = {}
            for pos in self.hivemind.board.nodes:
                node = self.hivemind.board.nodes[pos]
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

    def updateHuntValue(self, beliefs, iteration=50, discount=0.9):
        bounty = 10
        enemyValues = util.Counter()
        for agent in self.hivemind.enemyIndexes:
            for p in self.hivemind.board.positions:
                boardFeature = self.hivemind.board.positions[p]
                if not boardFeature.isNode:
                    boardFeature = boardFeature.ends[0]
                if boardFeature.isRed() == self.hivemind.isRed:
                    enemyValues[p] += bounty*beliefs[agent][p]
                else:
                    enemyValues[p] += 0

        for i in range(iteration):
            newValues = {}
            for p in self.hivemind.board.positions:
                boardFeature = self.hivemind.board.positions[p]
                values = [enemyValues[p]]
                newValue = 0
                if boardFeature.isNode and (boardFeature.isRed() == self.hivemind.isRed or boardFeature.onBorder):
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
                    newValue = discount*max(values) + enemyValues[p]
                elif not(boardFeature.isNode) and (boardFeature.ends[0].isRed() == self.hivemind.isRed):
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
                    newValue = discount*max(values) + enemyValues[p]
                newValues[p] = newValue
            enemyValues = newValues
        self.huntValue = enemyValues

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
        self.distancer = None

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
            self.policies = ValueIterations(foodGrid, beliefs, self)
            self.distancer = distanceCalculator.Distancer(gameState.data.layout)
            self.distancer.getMazeDistances()
        elif len(self.history) > 1:
            self.history = []
            beliefs = {}
            for agentIndex in range(0, gameState.getNumAgents()):
                belief = util.Counter()
                belief[gameState.getInitialAgentPosition(agentIndex)] = 1.0
                beliefs[agentIndex] = belief
            self.history.append((gameState, beliefs))

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
        self.policies.updateHuntValue(beliefs)
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
                        distances = [util.manhattanDistance(gameState.getAgentPosition(index),
                            positions[i][0]) for index in self.teamIndexes]
                        if (positions[i][1] < noiseReading -6 or
                            positions[i][1] > noiseReading + 6 or
                            distances[0] < 6 or distances[1] < 6):
                            del positions[i]
                    divisor = len(positions)
                    for pos in positions:
                        newBelief[pos[0]] += oldBelief[belief] / divisor
        newBelief.normalize()
        return newBelief

    def forecastBelief(self, oldBelief):
        newBelief = util.Counter()
        for belief in oldBelief:
            positions = []
            x, y = belief
            actions = Vectors.findNeigbours(x, y,
            self.history[0][0].getWalls(), allowStop=True)
            for action in actions:
                pos = Vectors.newPosition(x, y, action)
                positions.append(pos)
            divisor = len(positions)
            for pos in positions:
                newBelief[pos[0]] += oldBelief[belief] / divisor
        newBelief.normalize()
        return newBelief

    def getPreviousGameState(self, index=1):
        return self.history[-index][0]

    def getEnemyFood(self, gameState):
        foodGrid = None
        if self.isRed:
            foodGrid = gameState.getBlueFood()
        else:
            foodGrid = gameState.getRedFood()
        return foodGrid

    def getFeatures(self, state, action, index, iterable):
        """
        Takes the future position to get features for and a iterable of the features you want.
        """
        distScaleFactor = float(state.getWalls().width * state.getWalls().height)
        pos = state.getAgentPosition(index)
        position = Vectors.newPosition(pos[0], pos[1], action)
        features = util.Counter()
        # Boolean Features
        if "Bias" in iterable:
            features["Bias"] = 1.0
        if "On Edge" in iterable:
            features["On Edge"] = self.onEdgeFeature(position)
        if "Dead End" in iterable:
            features["Dead End"] = self.inDeadEndFeature(position)
        if "Home Side" in iterable:
            features["Home Side"] = self.homeSideFeature(position)
        if "Scared" in iterable:
            features["Scared"] = self.scaredFeature(index)
        if "Grab Food" in iterable:
            features["Grab Food"] = self.eatsFoodFeature(position, state)
        if "Capsule" in iterable:
            features["Capsule"] = self.eatsCapsuleFeature(position)
        # Distance Features
        if "Border" in iterable:
            features["Border"] = self.borderDistanceFeature(position) / distScaleFactor
        if "Food Dist" in iterable:
            features["Food Dist"] = self.foodDistanceFeature(position, state) / distScaleFactor
        if "Trespass" in iterable:
            pacmen, dists = [], []
            for enemy in self.enemyIndexes:
                pacmen.append(state.getAgentState(enemy).isPacman)
                dists.append(self.enemyDistanceFeature(position, enemy))
            trespassers = [d for p, d in zip(pacmen, dists) if p == True]
            trespass = min(trespassers if len(trespassers) != 0 else dists)
            features["Trespass"] = trespass / distScaleFactor
        if "Nearest Enemy Dist" in iterable:
            distances = []
            for enemy in self.enemyIndexes:
                distances.append(self.enemyDistanceFeature(position, enemy))
            features["Nearest Enemy Dist"] = min(distances) / distScaleFactor
        if "Enemy 0 Dist" in iterable:
            features["Enemy 0 Dist"] = self.enemyDistanceFeature(position, self.enemyIndexes[0]) / distScaleFactor
        if "Enemy 1 Dist" in iterable:
            features["Enemy 1 Dist"] = self.enemyDistanceFeature(position, self.enemyIndexes[1]) / distScaleFactor
        # Misc Features
        if "Score" in iterable:
            features["Score"] = self.scoreFeature(index, position)
        if "Turns" in iterable:
            features["Turns"] = self.turnsRemainingFeature()
        if "Carrying" in iterable:
            features["Carrying"] = self.foodCarriedFeature(index, position, state)
        if "Returned" in iterable:
            features["Returned"] = self.foodReturnedFeature(index, position, state)
        if "Near Food" in iterable:
            features["Near Food"] = self.nearbyFoodFeature(position, state)
        if "Near Enemy" in iterable:
            features["Near Enemy"] = self.enemiesOneAway(index, position, state)
        if "Kill" in iterable:
            features["Kill"] = self.kill(index, position, state)
        return features

    """
    Feature Extractors
    """
    def onEdgeFeature(self, position):
        return 1.0 if not self.board.positions[position].isNode else 0.0

    def inDeadEndFeature(self, position):
        return 1.0 if self.board.positions[position].isDeadEnd() else 0.0

    def homeSideFeature(self, position):
        return 1.0 if self.board.positions[position].isRed() == self.isRed else -1.0

    def scaredFeature(self, index):
        timer = self.history[-1][0].getAgentState(index).scaredTimer
        return 1.0 if timer > 1 else 0.0

    def eatsFoodFeature(self, position, state):
        x, y = position
        if self.enemiesOneAway(position) == 0 and self.getEnemyFood(state)[x][y]:
            return 1.0
        else:
            return 0.0

    def eatsCapsuleFeature(self, position):
        capsules = None
        if self.isRed:
            capsules = self.history[-1][0].getBlueCapsules()
        else:
            capsules = self.history[-1][0].getRedCapsules()
        return 1.0 if position in capsules else 0.0

    def borderDistanceFeature(self, position):
        # Initialise search
        fringe = util.PriorityQueue()
        visited = {}
        mid = self.history[0][0].getWalls().width / 2 + 0.5
        boardFeature = self.board.positions[position]
        if boardFeature.isNode:
            hCost = int(abs(mid - boardFeature.position[0]))
            fringe.push((boardFeature, 0), hCost)
        else:
            for node in boardFeature.distances(position):
                hCost = int(abs(mid - node[0].position[0]))
                fringe.push(node, node[1] + hCost)
        # While search hasn't failed
        while not fringe.isEmpty():
            node = fringe.pop()
            # Goal test
            if node[0].onBorder:
                return node[1]
            # Successor generation
            if node[0] not in visited:
                visited[node[0]] = node[1]
                edges = [node[0].exits[exit] for exit in node[0].exits]
                costs = [node[1] + edge.weight() for edge in edges]
                nodes = [edge.end(node[0]) for edge in edges]
                successors = zip(nodes, costs)
                for successor in successors:
                    hCost = int(abs(mid - successor[0].position[0]))
                    fringe.update(successor, successor[1] + hCost)

    def foodDistanceFeature(self, position, state):
        foodList = self.getEnemyFood(state).asList()
        width, height = state.getWalls().width, state.getWalls().height
        distances = []
        for pos in foodList:
            distances.append(self.distancer.getDistance(position, pos) / float(width * height))
            distances.append(self.distancer.getDistance(position, pos))
        return min(distances) if len(distances) > 0 else 0.0

    def enemyDistanceFeature(self, position, enemyIndex):
        belief = self.history[-1][1][enemyIndex]
        distance = 0
        for pos in belief:
            distance += self.distancer.getDistance(position, pos) * belief[pos]
        return distance

    def scoreFeature(self, index, position):
        boardFeature = self.board.positions[position]
        gameState = self.history[-1][0]
        score = gameState.data.score
        if boardFeature.isNode and boardFeature.onBorder and (boardFeature.isRed() == self.isRed):
            score += gameState.getAgentState(index).numCarrying
        if not self.isRed:
            score *= -1
        initialFood = self.history[0][0].getRedFood().count()
        return score / initialFood

    def turnsRemainingFeature(self):
        return 300 - (len(self.history) / 2)

    def foodCarriedFeature(self, index, position, state):
        boardFeature = self.board.positions[position]
        if boardFeature.isNode and boardFeature.onBorder and (boardFeature.isRed() == self.isRed):
            return 0
        carried = self.history[-1][0].getAgentState(index).numCarrying
        return float(carried + self.eatsFoodFeature(position, state))

    def foodReturnedFeature(self, index, position, state):
        returned = state.getAgentState(index).numReturned
        boardFeature = self.board.positions[position]
        if boardFeature.isNode and boardFeature.onBorder and (boardFeature.isRed() == self.isRed):
            returned += state.getAgentState(index).numCarrying
        return float(returned)

    def nearbyFoodFeature(self, position, state):
        return 1.0 if self.board.positions[position].neighbouringFood(self.getEnemyFood(state)) else 0.0

    def enemiesOneAway(self, index, position, state):
        numEnemies = 0.0
        agentState = state.getAgentState(index)
        positions = self.board.positions[position].oneAway(position)
        for enemy in self.enemyIndexes:
            enemyState = state.getAgentState(enemy)
            for pos in positions:
                if enemyState.getPosition() == pos:
                    if ((enemyState.isPacman and agentState.scaredTimer < 1) or
                            (self.board.positions[pos].isRed() != self.isRed
                            and enemyState.scaredTimer > 0)):
                        numEnemies += 1.0
                    else:
                        numEnemies -= 1.0
        return numEnemies

    def kill(self, index, position, state):
        killValue = 0.0
        agentState = state.getAgentState(index)
        for enemy in self.enemyIndexes:
            enemyState = state.getAgentState(enemy)
            if enemyState.getPosition() == position:
                if ((enemyState.isPacman and agentState.scaredTimer < 1) or
                        (self.board.positions[position].isRed() != self.isRed
                        and enemyState.scaredTimer > 0)):
                    killValue += 1.0
                else:
                    killValue -= 1.0
        return killValue

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'GreedyHivemindAgent', second = 'HunterAgent', **kwargs):
  print(f"create Team {kwargs}")
  hivemind = Hivemind([firstIndex, secondIndex], isRed)
  return [eval(first)(firstIndex, hivemind, **kwargs),
        eval(second)(secondIndex, hivemind, **kwargs)]

##########
# Agents #
##########

class GreedyHivemindAgent(CaptureAgent):

  def __init__( self, index, hivemind , timeForComputing = .1, **kwargs):
    self.index = index
    self.red = None
    self.distancer = None
    self.observationHistory = []
    self.display = None
    self.hivemind = hivemind
    self.lastNode = None

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.hivemind.registerInitialState(self.index, gameState)

  def findBestActions(self, gameState, policy):
    board = self.hivemind.board
    pos = gameState.getAgentPosition(self.index)
    boardFeature = board.positions[pos]
    bestActions = []
    maxValue = 0
    if boardFeature.isNode:
        self.lastNode = boardFeature
        actions = ['Stop']
        values = [policy[pos]]
        for exit in boardFeature.exits:
            newPos = boardFeature.exits[exit].end(boardFeature).position
            actions.append(exit)
            values.append(policy[newPos])
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    else:
        if self.lastNode is None:
            startValue = policy[boardFeature.ends[0].position]
            endValue = policy[boardFeature.ends[1].position]
            if startValue > endValue:
                self.lastNode = boardFeature.ends[1]
            else:
                self.lastNode = boardFeature.ends[0]
        posIndex = boardFeature.positions.index(pos)
        if self.lastNode is boardFeature.ends[1]:
            bestActions = [Directions.REVERSE[boardFeature.actions[posIndex]]]
            maxValue = policy[boardFeature.ends[0].position]
        else:
            bestActions = [boardFeature.actions[posIndex + 1]]
            maxValue = policy[boardFeature.ends[1].position]
    return (maxValue, bestActions)

  def chooseAction(self, gameState):
    self.hivemind.registerNewState(self.index, gameState)
    value = 0
    actions = ['Stop']
    pos = gameState.getAgentPosition(self.index)
    boardFeature = self.hivemind.board.positions[pos]
    if not boardFeature.isNode:
        boardFeature = boardFeature.ends[0]
    dist = []
    for index in self.hivemind.enemyIndexes:
        belief = self.hivemind.history[-1][1][index].argMax()
        dist.append(self.getMazeDistance(pos, belief))
    closest = min(dist)
    closestList = [a for a, v in zip(self.hivemind.enemyIndexes, dist) if v == closest]
    if boardFeature.isRed() != self.hivemind.isRed and closest < 6:
        enemyPolicy = self.hivemind.policies.enemyPosValues[closestList[0]]
        value, actions = self.findBestActions(gameState, enemyPolicy)
    else:
        nearbyFood = self.hivemind.board.positions[pos].neighbouringFood(
                self.hivemind.getEnemyFood(self.hivemind.getPreviousGameState()))
        if gameState.getAgentState(self.index).numCarrying > 2 and not nearbyFood:
            returnPolicy = self.hivemind.policies.returnHome
            value, actions = self.findBestActions(gameState, returnPolicy)
        else:
            # huntPolicy = self.hivemind.policies.huntValue
            # value, actions = self.findBestActions(gameState, huntPolicy)
            # if value == 0:
                foodPolicy = self.hivemind.policies.foodValues
                value, actions = self.findBestActions(gameState, foodPolicy)
    return random.choice(actions)

class DefensiveHivemindAgent(CaptureAgent):
  def __init__( self, index, hivemind , timeForComputing = .1):
    self.index = index
    self.red = None
    self.distancer = None
    self.observationHistory = []
    self.display = None
    self.hivemind = hivemind
    self.lastNode = None

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.hivemind.registerInitialState(self.index, gameState)

  def findBestActions(self, gameState, policy, reciprocal=False):
    board = self.hivemind.board
    pos = gameState.getAgentPosition(self.index)
    boardFeature = board.positions[pos]
    bestActions = []
    maxValue = 0
    if boardFeature.isNode:
        self.lastNode = boardFeature
        actions = ['Stop']
        value = policy[pos]
        if reciprocal and (policy[pos] > 0):
            value = 1 / value
        values = [value]
        for exit in boardFeature.exits:
            newPos = boardFeature.exits[exit].end(boardFeature).position
            actions.append(exit)
            value = policy[newPos]
            if reciprocal and (policy[newPos] > 0):
                value = 1 / value
            values.append(value)
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    else:
        if self.lastNode is None:
            startValue = policy[boardFeature.ends[0].position]
            endValue = policy[boardFeature.ends[1].position]
            if startValue > endValue:
                self.lastNode = boardFeature.ends[1]
            else:
                self.lastNode = boardFeature.ends[0]
        posIndex = boardFeature.positions.index(pos)
        if self.lastNode is boardFeature.ends[1]:
            bestActions = [Directions.REVERSE[boardFeature.actions[posIndex]]]
            maxValue = policy[boardFeature.ends[0].position]
            if reciprocal and (maxValue > 0):
                maxValue = 1 / maxValue
        else:
            bestActions = [boardFeature.actions[posIndex + 1]]
            maxValue = policy[boardFeature.ends[1].position]
            if reciprocal and maxValue > 0:
                maxValue = 1 / maxValue
    return (maxValue, bestActions)

  def chooseAction(self, gameState):
    self.hivemind.registerNewState(self.index, gameState)
    pos = gameState.getAgentPosition(self.index)
    nearbyFood = self.hivemind.board.positions[pos].neighbouringFood(self.hivemind.getEnemyFood(self.hivemind.getPreviousGameState()))
    value = 0
    actions = ['Stop']
    if gameState.getAgentState(self.index).numCarrying > 0 and not nearbyFood:
        returnPolicy = self.hivemind.policies.returnHome
        value, actions = self.findBestActions(gameState, returnPolicy)
    else:
        huntPolicy = self.hivemind.policies.huntValue
        value, actions = self.findBestActions(gameState, huntPolicy)
        if gameState.getAgentState(self.index).scaredTimer > 0 or value == 0:
            foodPolicy = self.hivemind.policies.foodValues
            value, actions = self.findBestActions(gameState, foodPolicy)
    return random.choice(actions)

class ApproximateQAgent(Agent):
    def __init__(self, index, hivemind, epsilon=0.05, gamma=0.8, alpha=0.2, **kwargs):
        # numTraining=100, epsilon=0.5, alpha=0.5, gamma=1
        self.index = index
        self.red = None
        self.observationHistory = []
        self.display = None
        self.hivemind = hivemind
        self.weights = util.Counter()
        self.explorationChance = float(epsilon)
        self.learningRate = float(alpha)
        self.discount = float(gamma)
        # Training reporting variables
        if 'numTraining' in kwargs:
            self.numTraining = int(kwargs['numTraining'])
        else:
            self.numTraining = 0
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0

    # Function for customizing Hivemind Q-Learning Agent
    def setWeights(self, weights):
        self.weights = weights

    def rewardFunction(self, gameState):
        carryDiff = float(gameState.getAgentState(self.index).numCarrying -
            self.lastState.getAgentState(self.index).numCarrying)
        returnDiff = float(gameState.getAgentState(self.index).numReturned -
            self.lastState.getAgentState(self.index).numReturned)
        reward = (returnDiff + carryDiff / 2.0) * 10.0
        return reward if reward != 0.0 else -0.05

    # ApproximateQAgent functions copied from p3-reinforcement-s3689650
    def registerInitialState(self, state):
        self.startEpisode()
        self.red = state.isOnRedTeam(self.index)
        import __main__
        if '_display' in dir(__main__):
          self.display = __main__._display
        self.hivemind.registerInitialState(self.index, state)
        self.observationHistory.append(state)

    def observationFunction(self, gameState):
        state = gameState.makeObservation(self.index)
        self.hivemind.registerNewState(self.index, state)
        self.observationHistory.append(state)
        if not self.lastState is None:
            reward = self.rewardFunction(gameState)
            self.episodeRewards += reward
            self.update(self.lastState, self.lastAction, state, reward)
        return state

    def getQValue(self, state, action):
        return self.weights * self.hivemind.getFeatures(state, action, self.index, self.weights)

    def computeValueFromQValues(self, state):
        value = float("-inf")
        actions = state.getLegalActions(self.index)
        if len(actions) > 0:
            value = max([self.getQValue(state, action) for action in actions])
        return value if value != float("-inf") else 0.0

    def computeActionFromQValues(self, state):
        bestActions = []
        bestValue = self.computeValueFromQValues(state)
        actions = state.getLegalActions(self.index)
        for action in actions:
            if self.getQValue(state, action) == bestValue:
                bestActions.append(action)
        bestAction = None
        if len(bestActions) > 0:
            bestAction = random.choice(bestActions)
        elif len(actions) > 0:
            bestAction = random.choice(actions)
        return bestAction

    def update(self, state, action, nextState, reward):
        oldValue = self.getQValue(state, action)
        nextValue = self.computeValueFromQValues(nextState)
        difference = (reward + self.discount * nextValue) - oldValue
        features = self.hivemind.getFeatures(state, action, self.index, self.weights)
        for feat in features:
            self.weights[feat] += self.learningRate * difference * features[feat]

    def getAction(self, state):
        legalActions = state.getLegalActions(self.index)
        action = None
        if len(legalActions) > 0:
            if util.flipCoin(self.explorationChance):
                action = random.choice(legalActions)
            else:
                action = self.computeActionFromQValues(state)
        self.lastState = state
        self.lastAction = action
        return action

    ### Episode Related Functions copied from p3-reinforcement
    def startEpisode(self):
        """
          Called by environment when new episode is starting
        """
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0

    def stopEpisode(self):
        """
          Called by environment when episode is done
        """
        if self.episodesSoFar < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.explorationChance = 0.0    # no exploration
            self.learningRate = 0.0      # no learning

    def isInTraining(self):
        return self.episodesSoFar < self.numTraining

    def isInTesting(self):
        return not self.isInTraining()

    def final(self, state):
        "Called at the end of each game."
        self.observationHistory = []
        # call the super-class final method
        self.hivemind.registerNewState(self.index, state)
        terminalScore = state.getScore() if self.hivemind.isRed else - state.getScore()
        deltaReward = self.rewardFunction(state)
        deltaReward -= state.getAgentState(self.index).numCarrying * 5.0
        self.episodeRewards += deltaReward
        self.update(self.lastState, self.lastAction, state, deltaReward)
        self.stopEpisode()

        # Make sure we have this var
        if not 'episodeStartTime' in self.__dict__:
            self.episodeStartTime = time.time()
        if not 'lastWindowAccumRewards' in self.__dict__:
            self.lastWindowAccumRewards = 0.0
        self.lastWindowAccumRewards += self.episodeRewards


        NUM_EPS_UPDATE = 5
        if self.episodesSoFar % NUM_EPS_UPDATE == 0:
            print('Reinforcement Learning Status:')
            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
            if self.episodesSoFar <= self.numTraining:
                trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
                print('\tCompleted %d out of %d training episodes' % (
                        self.episodesSoFar,self.numTraining))
                print('\tAverage Rewards over all training: %.2f' % (
                        trainAvg))
            else:
                testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
                print('\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining))
                print('\tAverage Rewards over testing: %.2f' % testAvg)
            print('\tAverage Rewards for last %d episodes: %.2f'  % (
                    NUM_EPS_UPDATE,windowAvg))
            print('\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime))
            self.lastWindowAccumRewards = 0.0
            self.episodeStartTime = time.time()

        if self.episodesSoFar == self.numTraining:
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg,'-' * len(msg)))

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print(self.weights)

    ### Debug functions copied from CaptureAgent
    def debugDraw(self, cells, color, clear=False):
      if self.display:
        from captureGraphicsDisplay import PacmanGraphics
        if isinstance(self.display, PacmanGraphics):
          if not type(cells) is list:
            cells = [cells]
          self.display.debugDraw(cells, color, clear)

    def debugClear(self):
        if self.display:
            from captureGraphicsDisplay import PacmanGraphics
            if isinstance(self.display, PacmanGraphics):
                self.display.clearDebug()

    def displayDistributionsOverPositions(self, distributions):
        dists = []
        for dist in distributions:
            if dist != None:
                if not isinstance(dist, util.Counter): raise Exception("Wrong type of distribution")
                dists.append(dist)
            else:
                dists.append(util.Counter())
        if self.display != None and 'updateDistributions' in dir(self.display):
            self.display.updateDistributions(dists)
        else:
            self._distributions = dists # These can be read by pacclient.py

class AllFeaturesAgent(ApproximateQAgent):
    def __init__(self, *args, **kwargs):
        ApproximateQAgent.__init__(self, *args, **kwargs)
        weights = util.Counter()
        weights["Bias"] = 1.0
        weights["On Edge"] = 1.0
        weights["Dead End"] = 1.0
        weights["Home Side"] = 1.0
        weights["Scared"] = 1.0
        weights["Grab Food"] = 1.0
        weights["Capsule"] = 1.0
        weights["Border"] = 1.0
        weights["Food Dist"] = 1.0
        weights["Score"] = 1.0
        weights["Turns"] = 1.0
        weights["Carrying"] = 1.0
        weights["Near Food"] = 1.0
        weights["Near Enemy"] = 1.0
        # weights["Nearest Enemy Dist"] = 1.0
        # weights["Enemy 0 Dist"] = 1.0
        # weights["Enemy 1 Dist"] = 1.0
        weights["Returned"] = 1.0
        self.setWeights(weights)

class HunterAgent(ApproximateQAgent):
    def __init__(self, *args, gamma=0.99, **kwargs):
        ApproximateQAgent.__init__(self, *args, **kwargs)
        weights = util.Counter()
        weights["Trespass"] = -3.2180937068880735
        weights["Near Enemy"] = 49.612649858234974
        weights["Kill"] = 100.13878098751749
        self.setWeights(weights)

    def rewardFunction(self, gameState):
        scoreChange = gameState.getScore() - self.lastState.getScore()
        if not self.hivemind.isRed:
            scoreChange *= -1.0
        reward = min(0.0, scoreChange) * 100.0
        x, y = self.lastState.getAgentPosition(self.index)
        lastPos = Vectors.newPosition(x, y,
                self.lastAction)
        lastEnemyPos = []
        for enemy in self.hivemind.enemyIndexes:
            lastEnemyPos.append(self.lastState.getAgentPosition(enemy))
        if gameState.getAgentPosition(self.index) != lastPos:
            reward -= 100.0
        elif lastPos in lastEnemyPos:
            reward += 100.0
        return reward if reward != 0.0 else 1.0
