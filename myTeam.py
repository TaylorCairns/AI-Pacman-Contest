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
        for end in ends:
            if end.isDeadEnd():
                deadEnd = True
        return deadEnd

    def oneAway(self, position):
        positions = []
        index = self.positions.index(position)
        if index == 0:
            positions.append(self.end[0].position)
        else:
            positions.append(self.positions[index - 1])
        if index == len(self.positions) - 1:
            positions.append(self.end[1].position)
        else:
            positions.append(self.positions[index + 1])
        return positions

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
        return False if len(exits) > 1 else True

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
            if boardFeature.isRed == self.hivemind.isRed:
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
            if self.hivemind.isRed == node.isRed and node.onBorder:
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
                if boardFeature.isRed == self.hivemind.isRed:
                    enemyValues[p] += bounty*beliefs[agent][p]
                else:
                    enemyValues[p] += 0

        for i in range(iteration):
            newValues = {}
            for p in self.hivemind.board.positions:
                boardFeature = self.hivemind.board.positions[p]
                values = [enemyValues[p]]
                newValue = 0
                if boardFeature.isNode and (boardFeature.isRed == self.hivemind.isRed or boardFeature.onBorder):
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
                elif not(boardFeature.isNode) and (boardFeature.ends[0].isRed == self.hivemind.isRed):
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

    def getEnemyFood(self):
        foodGrid = None
        if self.isRed:
            foodGrid = self.history[-1][0].getBlueFood()
        else:
            foodGrid = self.history[-1][0].getRedFood()
        return foodGrid

    def getFeatures(self, position, iterable):
        features = util.Counter()
        # Boolean Features
        if "On Edge" in iterable:
            features["On Edge"] = self.onEdgeFeature(position)
        if "Dead End" in iterable:
            features["Dead End"] = self.inDeadEndFeature(position)
        if "Surrounded" in iterable:
            features["Surrounded"] = self.surroundedFeature(position)
        if "Grab Food" in iterable:
            features["Grab Food"] = self.eatsFoodFeature(position)
        if "Capsule" in iterable:
            features["Capsule"] = self.eatsCapsuleFeature(position)
        # Distance Features
        if "Border" in iterable:
            features["Border"] = 1 / (self.borderDistanceFeature(position) + 1)
        if "Food Dist" in iterable:
            features["Food Dist"] = 1 / (self.foodDistanceFeature(position) + 1)
        if "Enemy Dist" in iterable:
            features["Enemy Dist"] = 1 / (self.enemyDistanceFeature(position) + 1)
        # Misc Features
        if "Score" in iterable:
            features["Score"] = self.scoreFeature(self.history[-1][0])
        if "Turns" in iterable:
            features["Turns"] = self.turnsRemainingFeature()
        if "Carrying" in iterable:
            features["Carrying"] = self.foodCarriedFeature()
        if "Near Food" in iterable:
            features["Near Food"] = self.nearbyFoodFeature(position)
        if "Near Enemy" in iterable:
            features["Near Enemy"] = self.oneAwayFeature(position)
        return features

    """
    Feature Extractors
    """
    def scoreFeature(self, gameState):
        score = gameState.data.score
        if not self.isRed:
            score *= -1
        initialFood = self.history[0][0].getRedFood().count()
        return score / initialFood

    def eatsFoodFeature(self, position):
        x, y = position
        return 1 if self.getEnemyFood()[x][y] else 0

    def eatsCapsuleFeature(self, position):
        capsules = None
        if self.isRed:
            capsules = self.history[-1][0].getBlueCapsules()
        else:
            capsules = self.history[-1][0].getRedCapsules()
        return 1 if position in capsules else 0

    def onEdgeFeature(self, position):
        return 1 if not self.board.positions[position].isNode else 0

    def inDeadEndFeature(self, position):
        return 1 if self.board.positions[position].isDeadEnd() else 0

    def foodCarriedFeature(self):
        index = len(self.history) % 2
        return self.history[-1][0].getAgentState(index).numCarrying

    def enemiesOneAway(self, position):
        sumProb = 0
        positions = self.board.positions[position].oneAway(position)
        for agent in self.enemyIndexes:
            sumProb += self.history[-1][1][agent][position]
            for pos in positions:
                sumProb += self.history[-1][1][agent][pos]
        return sumProb

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
            for node in boardFeature.distances(positions):
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

    def foodDistanceFeature(self, position):
        foodGrid = self.getEnemyFood()
        # Initialise search
        fringe = util.PriorityQueue()
        visited = {}
        boardFeature = self.board.positions[position]
        if boardFeature.isNode:
            fringe.push((boardFeature.position, 0), 0)
        else:
            for node in boardFeature.distances(positions):
                fringe.push(node[0].position, node[1])
        # While search hasn't failed
        while not fringe.isEmpty():
            next, cost = fringe.pop()
            boardFeature = self.board.positions[next]
            # Goal test
            if boardFeature.hasFood(foodGrid):
                return cost
            # Successor generation
            if next not in visited:
                visited[next] = cost
                edges = [boardFeature.exits[exit] for exit in boardFeature.exits]
                successors = []
                for edge in edges:
                    if edge.hasFood(foodGrid):
                        distance = 1
                        for pos in edge.positions:
                            if foodGrid[pos[0]][pos[1]]:
                                successors.append((pos, cost + distance))
                                break
                            distance += 1
                    else:
                        node = edge.end(boardFeature)
                        successors.append((node.position, cost + edge.weight()))
                for successor in successors:
                    fringe.update(successor, successor[1])

    def enemyDistanceFeature(self, position):
        # Initialise search
        fringe = util.PriorityQueue()
        visited = {}
        boardFeature = self.board.positions[position]
        if boardFeature.isNode:
            fringe.push((boardFeature.position, 0), 0)
        else:
            for node in boardFeature.distances(positions):
                fringe.push(node[0].position, node[1])
        # While search hasn't failed
        while not fringe.isEmpty():
            next, cost = fringe.pop()
            boardFeature = self.board.positions[next]
            # Goal test
            if not boardFeature.isNode:
                return cost
            else:
                prob = 0
                for agent in self.enemyIndexes:
                    prob += self.history[-1][1][agent][next]
                if prob > 0:
                    return cost
            # Successor generation
            if next not in visited:
                visited[next] = cost
                edges = [boardFeature.exits[exit] for exit in boardFeature.exits]
                successors = []
                for edge in edges:
                    for agent in self.enemyIndexes:
                        if edge.calcAgentProb(self.history[-1][1][agent][next]) > 0:
                            distance = 1
                            for pos in edge.positions:
                                if self.history[-1][1][agent][pos] > 0:
                                    successors.append((pos, cost + distance))
                                    break
                                distance += 1
                        else:
                            node = edge.end(boardFeature)
                            successors.append((node.position, cost + edge.weight()))
                for successor in successors:
                    fringe.update(successor, successor[1])

    def nearbyFoodFeature(self, position):
        return self.board.positions[position].neighbouringFood(self.getEnemyFood())

    def turnsRemainingFeature(self):
        return 300 - ((len(self.history) - 1) / 2)

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'GreedyHivemindAgent', second = 'DefensiveHivemindAgent'):
  hivemind = Hivemind([firstIndex, secondIndex], isRed)
  return [eval(first)(firstIndex, hivemind), eval(second)(secondIndex, hivemind)]

##########
# Agents #
##########

class GreedyHivemindAgent(CaptureAgent):

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
    if boardFeature.isRed != self.hivemind.isRed and closest < 6:
        enemyPolicy = self.hivemind.policies.enemyPosValues[closestList[0]]
        value, actions = self.findBestActions(gameState, enemyPolicy)
    else:
        nearbyFood = self.hivemind.board.positions[pos].neighbouringFood(self.hivemind.getEnemyFood())
        if gameState.getAgentState(self.index).numCarrying > 2 and not nearbyFood:
            returnPolicy = self.hivemind.policies.returnHome
            value, actions = self.findBestActions(gameState, returnPolicy)
        else:
            huntPolicy = self.hivemind.policies.huntValue
            value, actions = self.findBestActions(gameState, huntPolicy)
            if value == 0:
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
    nearbyFood = self.hivemind.board.positions[pos].neighbouringFood(self.hivemind.getEnemyFood())
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
