# multiAgents.py
# --------------
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


from util import manhattanDistance
import random
import util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (new_food) and Pacman position after moving (new_pos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = currentGameState.generatePacmanSuccessor(action)
        new_pos = successor_game_state.getPacmanPosition()
        new_food = successor_game_state.getFood()
        new_ghost_states = successor_game_state.getGhostStates()
        new_capsules = successor_game_state.getCapsules()

        score = successor_game_state.getScore()

        # Compute the nearest food distance
        food_list = new_food.asList()
        nearest_food_distance = float('inf')
        for food_pos in food_list:
            nearest_food_distance = min(
                nearest_food_distance, util.manhattanDistance(food_pos, new_pos))
        if not food_list:
            nearest_food_distance = 0

        # Compute the nearest ghost distance, incorporating scared ghosts
        nearest_ghost_distance = float('inf')
        for ghost_state in new_ghost_states:
            ghost_pos = ghost_state.getPosition()
            ghost_distance = util.manhattanDistance(ghost_pos, new_pos)
            if ghost_state.scaredTimer > 0:
                # Reward Pacman for chasing scared ghosts
                # More points for closer scared ghosts
                score += max(20 - ghost_distance, 0)
            else:
                nearest_ghost_distance = min(
                    nearest_ghost_distance, ghost_distance)

        # Add a penalty for food distance (closer food is better)
        score -= nearest_food_distance / 6

        # Add a penalty for being too close to non-scared ghosts
        if nearest_ghost_distance < 2:  # Avoid getting too close to ghosts
            score -= (2 ** (5 - nearest_ghost_distance))

        # Add a penalty for remaining food
        food_penalty = len(food_list) * 5  # Penalize more food left
        score -= food_penalty

        # Reward Pacman for eating food
        if currentGameState.getFood().count() > new_food.count():
            score += 10  # Reward for eating food

        # Encourage Pacman to go for capsules when ghosts are nearby
        nearest_capsule_distance = float('inf')
        if new_capsules:
            nearest_capsule_distance = min(
                [util.manhattanDistance(capsule, new_pos) for capsule in new_capsules])
            # Encourage getting capsules
            score += 10 / (nearest_capsule_distance + 1)

        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Get the legal actions for Pacman (agentIndex = 0 at the max layer)
        legal_actions = gameState.getLegalActions(0)

        max_score = float('-inf')
        scores = []

        for action in legal_actions:
            successor_state = gameState.generateSuccessor(0, action)
            score = self.min_value(successor_state, 1, 1)
            scores.append(score)
            max_score = max(max_score, score)

        # Choose actions that gives the maximum score
        max_indices = [index for index in range(
            len(scores)) if scores[index] == max_score]
        chosen_index = random.choice(max_indices)

        return legal_actions[chosen_index]

    def max_value(self, game_state: GameState, depth: int, layer: int) -> int:
        """
        Computes the maximum score for Pacman (agentIndex = 0) at a given depth level.

        Pacman (agentIndex = 0) is trying to maximize the score, so this function finds the
        best possible score Pacman can achieve, given the game's current state and possible
        future actions. It also limits the search by a specified depth to avoid excessive recursion.

        Args:
        - game_state: The current game state.
        - depth: The current depth of the search (used to limit the search).
        - layer: The agentIndex; 0 for Pacman, >=1 for ghosts.

        Returns:
        - max_score: The maximum score Pacman can achieve.
        """
        # Check if we've reached a terminal state (win/lose) or the depth limit
        if game_state.isWin() or game_state.isLose() or depth > self.depth:
            return self.evaluationFunction(game_state)

        # Get the legal actions for Pacman (agentIndex = 0 at the max layer)
        legal_actions = game_state.getLegalActions(layer)

        max_score = float('-inf')
        for action in legal_actions:
            successor_state = game_state.generateSuccessor(layer, action)
            max_score = max(max_score, self.min_value(
                successor_state, depth, layer + 1))

        return max_score

    def min_value(self, game_state: GameState, depth: int, layer: int) -> int:
        """
        Computes the minimum score for the ghosts (agentIndex >= 1) at a given depth level.

        The ghosts try to minimize Pacman's score. This function calculates the minimum score
        possible, given the actions of the ghosts at each depth. If the search reaches the last
        ghost, it returns to the max layer for Pacman.

        Args:
        - game_state: The current game state.
        - depth: The current depth of the search (used to limit the search).
        - layer: The agentIndex (>= 1) for ghosts.

        Returns:
        - min_score: The minimum score that the ghosts can enforce on Pacman.
        """

        # Check if we've reached a terminal state (win/lose)
        if game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state)

        # There are multiple layers (multiple ghosts) within each depth level
        legal_actions = game_state.getLegalActions(layer)

        min_score = float('inf')
        for action in legal_actions:
            successor_state = game_state.generateSuccessor(layer, action)
            if layer < game_state.getNumAgents() - 1:
                # If it's not the last ghost, continue to the next ghost's turn (min layer)
                min_score = min(min_score, self.min_value(
                    successor_state, depth, layer + 1))
            else:
                # If it's the last ghost, return to Pacman's turn (max layer)
                min_score = min(min_score, self.max_value(
                    successor_state, depth + 1, 0))

        return min_score


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    Alpha-beta pruning is an optimization of minimax that avoids exploring
    branches of the tree that cannot influence the final decision.
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using alpha-beta pruning.
        This method uses alpha (best option for maximizer) and beta (best option for minimizer)
        to prune unnecessary branches.

        Args:
        - gameState: The current game state.

        Returns:
        - The chosen action that maximizes Pacman's score.
        """
        # Get the legal actions for Pacman (agentIndex = 0)
        legal_actions = gameState.getLegalActions(0)

        # Calculate the score of each action
        max_score = float('-inf')
        scores = []
        alpha, beta = float('-inf'), float('inf')

        for action in legal_actions:
            successor_state = gameState.generateSuccessor(0, action)
            score = self.min_value(successor_state, 1, 1, alpha, beta)
            scores.append(score)
            max_score = max(max_score, score)
            alpha = max(alpha, max_score)

        # Choose actions that gives the maximum score
        max_indices = [index for index in range(
            len(scores)) if scores[index] == max_score]
        chosen_index = random.choice(max_indices)

        return legal_actions[chosen_index]

    def max_value(self, game_state: GameState, depth: int, layer: int,
                  alpha: int, beta: int) -> int:
        """
        Computes the maximum score for Pacman (agentIndex = 0) at
        a given depth level using alpha-beta pruning.

        Args:
        - game_state: The current game state.
        - depth: The current depth of the search.
        - layer: The agentIndex, 0 for Pacman.
        - alpha: The best score for the maximizer (Pacman) found so far.
        - beta: The best score for the minimizer (ghosts) found so far.

        Returns:
        - max_score: The maximum score Pacman can achieve from this state.
        """
        # Check if we've reached a terminal state (win/lose) or the depth limit
        if game_state.isWin() or game_state.isLose() or depth > self.depth:
            return self.evaluationFunction(game_state)

        # Get the legal actions for Pacman (agentIndex = 0 at the max layer)
        legal_actions = game_state.getLegalActions(layer)

        max_score = float('-inf')
        for action in legal_actions:
            successor_state = game_state.generateSuccessor(layer, action)
            max_score = max(max_score, self.min_value(
                successor_state, depth, layer + 1, alpha, beta))
            if max_score > beta:
                return max_score
            alpha = max(alpha, max_score)

        return max_score

    def min_value(self, game_state: GameState, depth: int, layer: int,
                  alpha: int, beta: int) -> int:
        """
        Computes the minimum score for the ghosts (agentIndex >= 1) at
        a given depth level using alpha-beta pruning.

        Args:
        - game_state: The current game state.
        - depth: The current depth of the search.
        - layer: The agentIndex (>= 1) for ghosts.
        - alpha: The best score for the maximizer (Pacman) found so far.
        - beta: The best score for the minimizer (ghosts) found so far.

        Returns:
        - min_score: The minimum score the ghosts can enforce on Pacman.
        """
        # Check if we've reached a terminal state (win/lose)
        if game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state)

        # There are multiple layers (multiple ghosts) within each depth level
        legal_actions = game_state.getLegalActions(layer)

        min_score = float('inf')
        for action in legal_actions:
            successor_state = game_state.generateSuccessor(layer, action)
            if layer < game_state.getNumAgents() - 1:
                # If it's not the last ghost, continue to the next ghost's turn (min layer)
                min_score = min(min_score, self.min_value(
                    successor_state, depth, layer + 1, alpha, beta))
            else:
                # If it's the last ghost, return to Pacman's turn (max layer)
                min_score = min(min_score, self.max_value(
                    successor_state, depth + 1, 0, alpha, beta))
            if min_score < alpha:
                return min_score
            beta = min(beta, min_score)

        return min_score


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)

      Expectimax is useful for modeling probabilistic behavior, particularly for agents
      that may not always make optimal decisions (like ghosts). In this implementation,
      ghosts are assumed to choose actions uniformly at random rather than strategically.
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction.
        Pacman (agentIndex = 0) uses expectimax to maximize its score, while ghosts
        (agentIndex >= 1) are assumed to act uniformly at random.

        Args:
        - gameState: The current game state.

        Returns:
        - The chosen action that maximizes Pacman's expected score.
        """
        # Get the legal actions for Pacman (agentIndex = 0)
        legal_actions = gameState.getLegalActions(0)

        # Calculate the score of each action
        max_score = float('-inf')
        scores = []

        for action in legal_actions:
            successor_state = gameState.generateSuccessor(0, action)
            # Get the expected score considering ghost behavior (uniformly random choice)
            score = self.expected_value(successor_state, 1, 1)
            scores.append(score)
            max_score = max(max_score, score)

        # Choose actions that gives the maximum score
        max_indices = [index for index in range(
            len(scores)) if scores[index] == max_score]
        chosen_index = random.choice(max_indices)

        return legal_actions[chosen_index]

    def expectimax_value(self, game_state: GameState, depth: int, layer: int) -> int:
        """
        Computes the maximum score Pacman can achieve at a given depth level.

        Pacman tries to maximize the score, similar to the Minimax approach. However,
        ghosts will act randomly (modeled in expected_value), rather than adversarially.

        Args:
        - game_state: The current game state.
        - depth: The current depth of the search.
        - layer: The agentIndex, 0 for Pacman, >= 1 for ghosts.

        Returns:
        - max_score: The maximum score Pacman can achieve from this state.
        """
        # Check if we've reached a terminal state (win/lose) or the depth limit
        if game_state.isWin() or game_state.isLose() or depth > self.depth:
            return self.evaluationFunction(game_state)

        # Get the legal actions for Pacman (agentIndex = 0 at the max layer)
        legal_actions = game_state.getLegalActions(layer)

        max_score = float('-inf')
        for action in legal_actions:
            successor_state = game_state.generateSuccessor(layer, action)
            # Calculate the expected score, considering random ghost moves
            max_score = max(max_score, self.expected_value(
                successor_state, depth, layer + 1))

        return max_score

    def expected_value(self, game_state: GameState, depth: int, layer: int) -> int:
        """
        Computes the expected value of a state, assuming ghosts choose their
        actions randomly.

        For ghost agents (agentIndex >= 1), we calculate the expected value of
        all possible actions, assuming each action is chosen uniformly at random.
        Pacman still tries to maximize its score.

        Args:
        - game_state: The current game state.
        - depth: The current depth of the search.
        - layer: The agentIndex (>= 1) for ghosts.

        Returns:
        - expected_value: The expected score Pacman can achieve based on
                          the random behavior of ghosts.
        """
        # Check if we've reached a terminal state (win/lose)
        if game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state)

        # There are multiple layers (multiple ghosts) within each depth level
        legal_actions = game_state.getLegalActions(layer)
        if not legal_actions:
            return 0

        total_score = 0
        for action in legal_actions:
            successor_state = game_state.generateSuccessor(layer, action)
            if layer < game_state.getNumAgents() - 1:
                # If it's not the last ghost, continue to the next ghost's turn (min layer)
                total_score += self.expected_value(
                    successor_state, depth, layer + 1)
            else:
                # If it's the last ghost, return to Pacman's turn (max layer)
                total_score += self.expectimax_value(
                    successor_state, depth + 1, 0)

        return total_score


def betterEvaluationFunction(currentGameState: GameState):
    """
    Improved evaluation function to better handle ghost avoidance and
    optimize for long-term survival, while still prioritizing food collection
    and capsule use.

    DESCRIPTION:
    - Increased penalty for being close to non-scared ghosts (especially if within range 2).
    - Increased weight for collecting capsules, especially when ghosts are near.
    - Introduced a penalty for dead-end positions to avoid trapping Pacman.
    - Maintains balance between prioritizing food collection and survival.
    """
    pacman_position = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghost_states = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    score = currentGameState.getScore()

    # --- Factor 1: Pacman's distance to the nearest food ---
    if food:
        nearest_food_distance = min(
            [manhattanDistance(pacman_position, food_pos) for food_pos in food])
        # Incentivize Pacman to get closer to food
        score += 10 / (nearest_food_distance + 1)
        score -= 4 * len(food)  # Fewer food pellets remaining is better

    # --- Factor 2: Pacman's distance to ghosts (with improved handling) ---
    for ghost in ghost_states:
        ghost_position = ghost.getPosition()
        distance_to_ghost = manhattanDistance(pacman_position, ghost_position)

        if ghost.scaredTimer > 0:
            # --- Factor 3: Scared Ghosts (closer is better) ---
            # Encourage Pacman to eat scared ghosts
            score += 200 / (distance_to_ghost + 1)
        else:
            # Dangerous ghosts: Penalize more heavily if Pacman is within close range
            if distance_to_ghost > 0:
                if distance_to_ghost <= 2:
                    score -= 150 / distance_to_ghost  # Strong penalty for being too close
                else:
                    score -= 10 / distance_to_ghost  # Standard penalty for being near ghosts

    # --- Factor 4: Handling capsules more aggressively ---
    if capsules:
        nearest_capsule_distance = min(
            [manhattanDistance(pacman_position, cap_pos) for cap_pos in capsules])
        # Encourage Pacman to grab capsules
        score += 25 / (nearest_capsule_distance + 1)
        score -= 100 * len(capsules)  # Fewer capsules left is better

    # --- Factor 5: Avoiding dead-end positions ---
    legal_moves = currentGameState.getLegalActions(0)
    if len(legal_moves) == 1:  # If Pacman is in a dead-end (only one possible move)
        score -= 200  # Heavily penalize dead-ends, so Pacman avoids getting trapped

    return score


# Abbreviation
better = betterEvaluationFunction
