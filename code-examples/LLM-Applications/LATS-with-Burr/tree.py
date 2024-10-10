import math
from collections import deque
from typing import Optional
from reflection import Reflection

class Node:
    def __init__(
        self,
        messages: list[dict],
        reflection: Reflection,
        parent: Optional["Node"] = None,
    ):
        self.messages = messages
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0
        self.reflection = reflection
        self.depth = parent.depth + 1 if parent is not None else 1
        self.is_solved = reflection.found_solution
        self.backpropagate(reflection.normalized_score, self.is_solved)

    @property
    def height(self) -> int:
        """Check for how far we've rolled out the tree."""
        if self.children:
            return 1 + max([child.height for child in self.children])
        return 1
    
    def select_best(self):
        """Starting from the root node a child node is selected at each tree level until a leaf node is reached."""        
        node = self
        while node.children:
            max_child = max(node.children, key=lambda child: child.upper_confidence_bound())
            node = max_child

        return node
    
    def select_best_solution(self):
        """Select the best terminal node with solution"""        
        all_solved_leaves = []
        nodes = deque()
        nodes.append(self)
        while nodes:
            node = nodes.popleft()
            if not node.children and node.is_solved:
                all_solved_leaves.append(node)
            for child in node.children:
                nodes.append(child)

        if all_solved_leaves:
            return max(all_solved_leaves, key=lambda node: node.upper_confidence_bound(0))
        return None

    def upper_confidence_bound(self, exploration_weight=1.0):
        """Return the UCT score. This helps balance exploration vs. exploitation of a branch."""
        if self.parent is None:
            raise ValueError("Cannot obtain UCT from root node")
        if self.visits == 0:
            return self.value
        # Encourages exploitation of high-value trajectories
        average_reward = self.value / self.visits
        # Encourages exploration of less-visited trajectories
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        return average_reward + exploration_weight * exploration_term

    def backpropagate(self, score: float, is_solved: bool):
        """Update the score of this node and its parents."""
        node = self
        while node:
            node.visits += 1
            node.value += score
            node.is_solved = node.is_solved or is_solved
            node = node.parent

    def get_messages(self):
        return self.messages + [self.reflection.as_message()]

    def get_trajectory(self) -> list[dict]:
        """Get messages representing this search branch."""
        messages = []
        node = self
        while node:
            messages = node.get_messages() + messages
            node = node.parent
        return messages 