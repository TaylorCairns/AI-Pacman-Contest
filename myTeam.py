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

    def trapped(self, position, enemyIndexes, beliefDistribution):
        index = self.positions.index(position)
        if self.ends[0].isDeadEnd():
            for enemy in enemyIndexes:
                prob = self.ends[1].calcAgentProb(beliefDistribution[enemy])
                for i in range(index, len(self.positions)):
                    prob += beliefDistribution[enemy][self.positions[i]]
                if prob == 1.0:
                    return True
        elif self.ends[1].isDeadEnd():
            for enemy in enemyIndexes:
                prob = self.ends[0].calcAgentProb(beliefDistribution[enemy])
                for i in range(index + 1):
                    prob += beliefDistribution[enemy][self.positions[i]]
                if prob == 1.0:
                    return True
        return False

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

    def trapped(self, position, enemyIndexes, beliefDistribution):
        if len(self.exits) == 1:
            edge = iter(self.exits.values()).__next__()
            for enemy in enemyIndexes:
                prob = self.calcAgentProb(beliefDistribution[enemy])
                prob += edge.calcAgentProb(beliefDistribution[enemy])
                prob += edge.end(self).calcAgentProb(beliefDistribution[enemy])
                if prob == 1.0:
                    return True
        elif len(self.exits) == 2:
            blocked = {}
            for exit in self.exits:
                blocked[exit] = False
                edge = self.exits[exit]
                for enemy in enemyIndexes:
                    prob = self.calcAgentProb(beliefDistribution[enemy])
                    prob += edge.calcAgentProb(beliefDistribution[enemy])
                    prob += edge.end(self).calcAgentProb(beliefDistribution[enemy])
                    if prob == 1.0:
                        blocked[exit] = True
            if all(blocked.values()):
                return True
        return False
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

    def getBeliefDistributions(self):
        distributions = []
        for agentIndex in range(self.history[-1][0].getNumAgents()):
            distributions.append(self.history[-1][1][agentIndex].copy())
        return distributions

    def getFeatures(self, state, action, agent):
        """
        Takes the future position to get features for and a iterable of the features you want.
        """
        distScaleFactor = float(state.getWalls().width * state.getWalls().height)
        foodScaleFactor = self.history[0][0].getRedFood().count()
        enemiesScaleFactor = len(self.enemyIndexes)
        x, y = state.getAgentPosition(agent.index)
        position = Vectors.newPosition(x, y, action)
        features = util.Counter()
        # Boolean Features
        if "Bias" in agent.getWeights():
            features["Bias"] = 1.0
        if "On Edge" in agent.getWeights():
            features["On Edge"] = self.onEdgeFeature(position)
        if "Dead End" in agent.getWeights():
            features["Dead End"] = self.inDeadEndFeature(position)
        if "Home Side" in agent.getWeights():
            features["Home Side"] = self.homeSideFeature(position)
        if "Scared" in agent.getWeights():
            features["Scared"] = self.scaredFeature(agent.index)
        if "Grab Food" in agent.getWeights():
            features["Grab Food"] = self.eatsFoodFeature(agent.index,  position, state)
        if "Capsule" in agent.getWeights():
            features["Capsule"] = self.eatsCapsuleFeature(position)
        if "Delivery" in agent.getWeights():
            features["Delivery"] = self.foodDeliveredFeature(agent.index, position, state)
        if "Chased" in agent.getWeights():
            features["Chased"] = self.beingChased(agent.index, position, state)
        # Distance Features
        if "Border" in agent.getWeights():
            features["Border"] = self.borderDistanceFeature(position) / distScaleFactor
        if "Food Dist" in agent.getWeights():
            features["Food Dist"] = self.foodDistanceFeature(position, state) / distScaleFactor
        if "Trespass" in agent.getWeights():
            features["Trespass"] = self.nearestTrespasserFeature(self, position) / distScaleFactor
        if "Nearest Enemy Dist" in agent.getWeights():
            features["Nearest Enemy Dist"] = self.nearestEnemyFeature(position) / distScaleFactor
        if "Enemy 0 Dist" in agent.getWeights():
            features["Enemy 0 Dist"] = self.enemyDistanceFeature(position, self.enemyIndexes[0]) / distScaleFactor
        if "Enemy 1 Dist" in agent.getWeights():
            features["Enemy 1 Dist"] = self.enemyDistanceFeature(position, self.enemyIndexes[1]) / distScaleFactor
        # Misc Features
        if "Score" in agent.getWeights():
            features["Score"] = self.scoreFeature(agent.index, position) / foodScaleFactor
        if "Turns" in agent.getWeights():
            features["Turns"] = self.turnsRemainingFeature()
        if "Carrying" in agent.getWeights():
            features["Carrying"] = self.foodCarriedFeature(agent.index, position, state) / foodScaleFactor
        if "Return" in agent.getWeights():
            features["Return"] = self.foodReturnFeature(agent.index, position, state) / (foodScaleFactor * distScaleFactor)
        if "Near Food" in agent.getWeights():
            features["Near Food"] = self.nearbyFoodFeature(position, state) / foodScaleFactor
        if "Near Enemy" in agent.getWeights():
            features["Near Enemy"] = self.enemiesOneAway(agent.index, position, state) / enemiesScaleFactor
        if "Kill" in agent.getWeights():
            features["Kill"] = self.kill(agent.index, position, state) / enemiesScaleFactor
        return features

    """
    Feature Extractors
    """
    # Boolean Features
    def onEdgeFeature(self, position):
        return 1.0 if not self.board.positions[position].isNode else 0.0

    def inDeadEndFeature(self, position):
        return 1.0 if self.board.positions[position].isDeadEnd() else 0.0

    def homeSideFeature(self, position):
        return 1.0 if self.board.positions[position].isRed() == self.isRed else -1.0

    def scaredFeature(self, index):
        timer = self.history[-1][0].getAgentState(index).scaredTimer
        return 1.0 if timer > 1 else 0.0

    def eatsFoodFeature(self, index,  position, state):
        x, y = position
        if self.enemiesOneAway(index, position, state) == 0 and self.getEnemyFood(state)[x][y]:
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

    def foodDeliveredFeature(self, index, position, state):
        boardFeature = self.board.positions[position]
        if (boardFeature.isNode and boardFeature.onBorder and
                (boardFeature.isRed() == self.isRed) and
                state.getAgentState(index).numCarrying > 0):
            return 1.0
        return 0.0

    def beingChased(self, index, position, state):
        for enemy in self.enemyIndexes:
            enemyPos = state.getAgentPosition(enemy)
            if enemyPos != None:
                lastBelief = self.history[-3][1][enemy]
                lastDistance = 0
                for pos in lastBelief:
                    lastDistance += self.distancer.getDistance(position, pos) * lastBelief[pos]
                belief = self.history[-1][1][enemy]
                distance = 0
                for pos in belief:
                    distance += self.distancer.getDistance(position, pos) * belief[pos]
                if distance <= lastDistance and distance < 6:
                    return 1.0
        return 0.0
    # Distance Features
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
        distances = []
        for pos in foodList:
            distances.append(self.distancer.getDistance(position, pos))
        return min(distances) if len(distances) > 0 else 0.0

    def enemyDistanceFeature(self, position, enemyIndex):
        belief = self.history[-1][1][enemyIndex]
        distance = 0
        for pos in belief:
            distance += self.distancer.getDistance(position, pos) * belief[pos]

    def nearestEnemyFeature(self, position):
        distances = []
        for enemy in self.enemyIndexes:
            distances.append(self.enemyDistanceFeature(position, enemy))
        return min(distances)

    def nearestTrespasserFeature(self, position):
        pacmen, dists = [], []
        for enemy in self.enemyIndexes:
            pacmen.append(state.getAgentState(enemy).isPacman)
            dists.append(self.enemyDistanceFeature(position, enemy))
        trespassers = [d for p, d in zip(pacmen, dists) if p == True]
        return min(trespassers) if len(trespassers) != 0 else float('inf')
    return distance
    # Misc Features
    def scoreFeature(self, index, position):
        boardFeature = self.board.positions[position]
        gameState = self.history[-1][0]
        score = gameState.data.score
        if boardFeature.isNode and boardFeature.onBorder and (boardFeature.isRed() == self.isRed):
            score += gameState.getAgentState(index).numCarrying
        if not self.isRed:
            score *= -1
        return score

    def turnsRemainingFeature(self):
        return 300 - (len(self.history) / 2)

    def foodCarriedFeature(self, index, position, state):
        boardFeature = self.board.positions[position]
        if boardFeature.isNode and boardFeature.onBorder and (boardFeature.isRed() == self.isRed):
            return 0
        carried = self.history[-1][0].getAgentState(index).numCarrying
        return float(carried + self.eatsFoodFeature(index,  position, state))

    def foodReturnFeature(self, index, position, state):
        carried = self.foodCarriedFeature(index, position, state)
        borderDist = self.borderDistanceFeature(position)
        return carried * borderDist

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
                            (enemyState.scaredTimer > 0 and not enemyState.isPacman)):
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
                        (enemyState.scaredTimer > 0 and not enemyState.isPacman)):
                    killValue += 1.0
                else:
                    killValue -= 1.0
        return killValue

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'AttackAgent', second = 'HunterAgent', **kwargs):
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
    nearbyFood = self.hivemind.board.positions[pos].neighbouringFood(
            self.hivemind.getEnemyFood(self.hivemind.getPreviousGameState()))
    value = 0
    actions = ['Stop']
    if gameState.getAgentState(self.index).numCarrying > 0 and not nearbyFood:
        returnPolicy = self.hivemind.policies.returnHome
        value, actions = self.findBestActions(gameState, returnPolicy)
    else:
        huntPolicy = self.hivemind.policies.huntValue
        value, actions = self.findBestActions(gameState, huntPolicy)
        # if gameState.getAgentState(self.index).scaredTimer > 0 or value == 0:
        #     foodPolicy = self.hivemind.policies.foodValues
        #     value, actions = self.findBestActions(gameState, foodPolicy)
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
        self.mode = None
        # Training reporting variables
        if 'numTraining' in kwargs:
            self.numTraining = int(kwargs['numTraining'])
        else:
            self.numTraining = 0
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0

    # Function for customizing Hivemind Q-Learning Agents
    def rewardFunction(self, gameState, isFinal=False):
        util.raiseNotDefined

    def getWeights(self):
        return self.weights

    def printWeights(self):
        print(self.weights)

    def setMode(self, gameState):
        self.mode = None

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
            self.setMode(gameState)
        return state

    def getQValue(self, state, action):
        return self.getWeights() * self.hivemind.getFeatures(state, action, self)

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
        features = self.hivemind.getFeatures(state, action, self)
        for feat in features:
            self.getWeights()[feat] += self.learningRate * difference * features[feat]

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
        deltaReward = self.rewardFunction(state, True)
        self.episodeRewards += deltaReward
        self.update(self.lastState, self.lastAction, state, deltaReward)
        self.stopEpisode()

        # Make sure we have this var
        if not 'episodeStartTime' in self.__dict__:
            self.episodeStartTime = time.time()
        if not 'lastWindowAccumRewards' in self.__dict__:
            self.lastWindowAccumRewards = 0.0
        self.lastWindowAccumRewards += self.episodeRewards


        NUM_EPS_UPDATE = 1
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
            self.printWeights()

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
        weights["Delivery"] = 1.0
        weights["Border"] = 1.0
        weights["Food Dist"] = 1.0
        weights["Trespass"] = 1.0
        weights["Nearest Enemy Dist"] = 1.0
        weights["Enemy 0 Dist"] = 1.0
        weights["Enemy 1 Dist"] = 1.0
        weights["Score"] = 1.0
        weights["Turns"] = 1.0
        weights["Carrying"] = 1.0
        weights["Return"] = 1.0
        weights["Near Food"] = 1.0
        weights["Near Enemy"] = 1.0
        weights["Kill"] = 1.0
        weights["Chased"] = 1.0
        self.weights = weights

class HunterAgent(ApproximateQAgent):
    def __init__(self, *args, gamma=0.99, **kwargs):
        ApproximateQAgent.__init__(self, *args, **kwargs)
        self.weights["Trespass"] = -43.52709827983609
        self.weights["Near Enemy"] = 113.58509702452676
        self.weights["Kill"] = 195.97367809099194

    def rewardFunction(self, gameState, isFinal=False):
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

class AttackAgent(ApproximateQAgent):
    def __init__(self, *args, gamma=0.99, **kwargs):
        ApproximateQAgent.__init__(self, *args, **kwargs)
        self.weights["Near Enemy"] = 42.88718091048829
        self.weights["Kill"] = 38.7956831767021
        self.weights["Grab Food"] = 5.71864804146864
        self.weights["Delivery"] = 25.73787094094885
        self.weights["Food Dist"] = -3.263353405110272
        self.weights["Trespass"] = -0.6578892848291951

    def rewardFunction(self, gameState, isFinal=False):
        carryDiff = float(gameState.getAgentState(self.index).numCarrying -
                self.lastState.getAgentState(self.index).numCarrying)
        returnDiff = float(gameState.getAgentState(self.index).numReturned -
                self.lastState.getAgentState(self.index).numReturned)
        reward = (returnDiff + carryDiff / 2.0) * 10.0
        x, y = self.lastState.getAgentPosition(self.index)
        lastPos = Vectors.newPosition(x, y,
                self.lastAction)
        lastEnemyPos = []
        for enemy in self.hivemind.enemyIndexes:
            lastEnemyPos.append(self.lastState.getAgentPosition(enemy))
        if gameState.getAgentPosition(self.index) != lastPos:
            reward -= 5.0
        elif lastPos in lastEnemyPos:
            reward += 5.0
        if isFinal:
            reward -= gameState.getAgentState(self.index).numCarrying * 5.0
        return reward if reward != 0.0 else -0.05

class ReactiveAgent(ApproximateQAgent):
    def __init__(self, *args, gamma=0.99, **kwargs):
        ApproximateQAgent.__init__(self, *args, **kwargs)
        # patrolMode - patrols border
        self.mode = "Patrol"
        self.patrol = None
        self.weights["Bias"] = 0.0
        # recklessFood - greedy food grab
        # cautiousFood - grab food safely - rewards staying near border/ avoid dead ends
        self.food = util.Counter()
        self.food["Near Enemy"] = 1.0
        self.food["Kill"] = 1.0
        self.food["Grab Food"] = 1.0
        self.food["Delivery"] = 1.0
        self.food["Food Dist"] = -1.0
        self.food["Nearest Enemy"] = 1.0
        # huntMode - hunts enemy pacman
        self.hunt = util.Counter()
        self.hunt["Trespass"] = -1.0
        self.hunt["Near Enemy"] = 1.0
        self.hunt["Kill"] = 1.0
        # escapeMode - safely returns home
        self.escape = util.Counter()
        self.escape["Near Enemy"] = 1.0
        self.escape["Kill"] = 1.0
        self.escape["Border"] = 1.0
        self.escape["Home Side"] = 1.0
        self.escape["Dead End"] = -1.0
        # suicideMode - kills self asap
        self.suicide = util.Counter()
        self.suicide["Near Enemy"] = -1.0
        self.suicide["Kill"] = -1.0
        self.suicide["Nearest Enemy"] = -1.0

    """
        Mode triggers
        start - patrolMode
        patrol mode - enemy pacman - huntMode
        patrol mode - food < half enemy distance - recklessFood
        huntMode - enemy closer to spawn than scared timer - suicide
        huntMode - no enemy pacman - patrolMode
        suicideMode - at spawn -patrolMode
        recklessFood - ? - cautiousFood
        cautiousFood/recklessFood - enemy Pacman and not scared - huntMode
        cautiousFood - chased - escapeMode
        recklessFood/cautiousFood/escapeMode - trapped - suicideMode
        escapeMode - returned home - patrolMode
    """
    def setMode(self, gameState):
        agentPos = gameState.getAgentPosition(self.index)
        agentState = gameState.getAgentState(self.index)
        if self.mode == None:
            self.mode = "Patrol"
        elif self.mode == "Patrol":
            enemies = []
            for enemy in self.hivemind.enemyIndexes:
                enemies.append(gameState.getAgentState(enemy).isPacman)
            if any(enemies):
                self.mode = "Hunt"
            else:
                foodDist = self.hivemind.foodDistanceFeature(agentPos, gameState)
                enemyDist = self.hivemind.nearestEnemyFeature(agentPos)
                if foodDist*2 < enemyDist:
                    self.mode = "Food"
        elif self.mode == "Hunt":
            enemies = []
            for enemy in self.hivemind.enemyIndexes:
                enemies.append(gameState.getAgentState(enemy).isPacman)
            if not any(enemies):
                self.mode = "Patrol"
            elif agentState.scaredTimer > 1:
                for enemy in self.hivemind.enemyIndexes:
                    enemyPos = self.hivemind.history[-1][1][enemy].argMax()
                    dist = self.hivemind.distancer.getDistance(enemyPos, agentState.start.pos)
                    if dist < agentState.scaredTimer:
                        self.mode = "Suicide"
                        break
        elif self.mode == "Suicide":
            if agentPos == agentState.start.pos:
                self.mode = "Patrol"
        elif self.mode == "Food":
            boardFeature = self.hivemind.board.positions[agentPos]
            if boardFeature.trapped(agentPos, self.hivemind.enemyIndexes,
                    self.hivemind.getBeliefDistributions()):
                self.mode = "Suicide"
            elif self.hivemind.beingChased(self.index, agentPos, gameState) == 1.0:
                self.mode = "Escape"
            else:
                for enemy in self.hivemind.enemyIndexes:
                    if gameState.getAgentState(enemy).isPacman:
                        self.mode = "Hunt"
                        break
        elif self.mode == "Escape":
            if self.hivemind.homeSideFeature(agentPos) == 1.0:
                self.mode = "Patrol"

    def getWeights(self):
        if self.mode == None:
            return self.weights
        elif self.mode == "Patrol":
            return self.weights
        elif self.mode == "Hunt":
            return self.hunt
        elif self.mode == "Suicide":
            return self.suicide
        elif self.mode == "Food":
            return self.food
        elif self.mode == "Escape":
            return self.escape

    def printWeights(self):
        print(f"Hunt: {self.hunt}")
        print(f"Food: {self.food}")
        print(f"Escape: {self.escape}")
        print(f"Suicide: {self.suicide}")

    def rewardFunction(self, gameState, isFinal=False):
        reward = 0.0
        pos = gameState.getAgentPosition(self.index)
        x, y = self.lastState.getAgentPosition(self.index)
        lastPos = Vectors.newPosition(x, y, self.lastAction)
        lastEnemyPos = []
        for enemy in self.hivemind.enemyIndexes:
            lastEnemyPos.append(self.lastState.getAgentPosition(enemy))
        if pos != lastPos:
            # Died - Penalty unless Suicide
            if self.mode != "Suicide":
                reward -= 100.0
            else:
                reward += 100.0
        elif lastPos in lastEnemyPos:
            # Killed enemy - Bonus if Hunt/Patrol
            if self.mode == "Hunt" or self.mode == "Patrol":
                reward += 100.0
            elif self.mode == "Suicide":
                reward -= 100.0
        # Score Loss - Penalty if Hunt or Patrol
        scoreChange = gameState.getScore() - self.lastState.getScore()
        if not self.hivemind.isRed:
            scoreChange *= -1.0
        if self.mode == "Hunt" or self.mode == "Patrol":
            reward += min(0.0, scoreChange) * 100.0
        if self.mode == "Food" or self.mode == "Escape":
            # Carry Gain - Bonus if AnyFood
            # Carry Loss - Penalty if AnyFood or Escape
            carryDiff = float(gameState.getAgentState(self.index).numCarrying -
                    self.lastState.getAgentState(self.index).numCarrying)
            if (carryDiff > 0.0 and self.mode == "Food") or carryDiff < 0.0:
                reward += carryDiff * 5.0
            # Returned Gain - Bonus if CautiousFood or Escape
            returnDiff = float(gameState.getAgentState(self.index).numReturned -
                    self.lastState.getAgentState(self.index).numReturned)
            if returnDiff > 0.0:
                reward += returnDiff * 10.0
            # Grab Capsule - Bonus if CautiousFood or Escape
            if self.hivemind.eatsCapsuleFeature(pos) == 1.0:
                reward += 10.0
        # Trapped - Penalty unless Suicide or Hunt
        boardFeature = self.hivemind.board.positions[pos]
        if self.mode != "Hunt" and self.mode != "Suicide" and boardFeature.trapped(pos,
                self.hivemind.enemyIndexes, self.hivemind.getBeliefDistributions()):
            reward -= 50.0
        return reward

    def getAction(self, state):
        if self.mode == None:
            self.setMode(state)
        action = None
        if self.mode == "Patrol":
            pos = state.getAgentPosition(self.index)
            if self.patrol == pos or self.patrol == None:
                targets = self.hivemind.board.border.keys()
                x = state.getWalls().width // 2
                if self.hivemind.isRed:
                    x -= 1
                targets = [target for target in targets if target[0] == x]
                if self.patrol != None:
                    y = state.getWalls().height // 2
                    temp = []
                    if pos[1] >= y:
                        temp = [target for target in targets if target[1] < y]
                    else:
                        temp = [target for target in targets if target[1] >= y]
                    if len(temp) > 0:
                        targets = temp
                self.patrol = random.choice(targets)
            actions = Vectors.findNeigbours(pos[0], pos[1], state.getWalls())
            values = []
            for action in actions:
                newPos = Vectors.newPosition(pos[0], pos[1], action)
                if (self.hivemind.enemiesOneAway(self.index, pos, state) < 0.0 or
                        self.hivemind.kill(self.index, pos, state) < 0.0):
                    values.append(float("inf"))
                else:
                    values.append(self.hivemind.distancer.getDistance(self.patrol, newPos))
            minValue = min(values)
            bestActions = [a for a, v in zip(actions, values) if v == minValue]
            action = random.choice(bestActions) if len(bestActions) > 0 else 'Stop'
        else:
            legalActions = state.getLegalActions(self.index)
            if len(legalActions) > 0:
                if util.flipCoin(self.explorationChance):
                    action = random.choice(legalActions)
                else:
                    action = self.computeActionFromQValues(state)
        self.lastState = state
        self.lastAction = action
        return action
