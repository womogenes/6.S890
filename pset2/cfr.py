"""
Implement counterfactual regret minimization for two-player zero-sum games

The idea:
    Instantiate R_j, a regret minimizer at each decision node. Each one outputs
    a local strategy (probability distribution over actions) for that node's
    actions only.

    NextStrategy():
        - Ask each R_j for its next strategy.
        - Assemble these strategies into overall sequence-form strategy
    ObserveUtility(g^t):
        - We want each R_j to observe a utility.
        - Initialize new map V^t that takes [infosets] to real numbers
        - For each node:
            value is sum over local actions of
                local R_j prob of that action times
                (sum of gradient of this (node, action) pair and
                 V^t of child node)
        - For each decision node:
            - zero out "local gradient"
            - for each action:
                "local gradient" at this action is given g^t plus
                V^t of child
            - update local R_j
"""

import numpy as np
from functools import lru_cache

from pprint import pprint

from game import Game

class RegretMatching:
    def __init__(self, A):
        self.A = sorted(A)
        self.N = len(A)
        self.r = [np.zeros(self.N)]
        self.x = [np.ones(self.N) / self.N]
        self.T = 0

    def next_strategy(self):
        prev_r = self.r[-1]
        if np.any(prev_r != 0):
            strat = prev_r / np.linalg.norm(prev_r, ord=1)
        else:
            # Return uniform strategy by default
            strat = self.x[0]
        
        # Turn into a dict for the outside world
        return {a: float(strat[i]) for i, a in enumerate(self.A)}
        
    def observe_utility(self, g):
        # g is a dict, turn it into a gradient vector
        g = np.array([g[a] for a in self.A])

        prev_r = self.r[-1]
        assert g.shape == prev_r.shape, "Grad vector shape does not match regret shape"
        self.r.append(prev_r + g - np.dot(g, self.x[-1]) @ np.ones_like(prev_r))

class CFR:
    def __init__(self, game: Game, player: str):
        # Make one instance of regret matching per decision node
        self.tfdp = game.build_tfdp(player)

        self.J = {key: node for key, node in self.tfdp.items() if node["type"] == "decision"}
        self.K = {key: node for key, node in self.tfdp.items() if node["type"] == "obs"}

        self.R: dict[str, RegretMatching] = {}
        for j in self.J:
            # Initialize regret minimizer
            self.R[j] = RegretMatching(self.J[j]["actions"])

        # Assume always up-to-date
        self.b = {}
        self.x = []
        self.Sigma = game.all_seqs[player]
        self.game = game

    def next_strategy(self):
        self.b = {}

        # Step 1: get local strategies per decision node
        for j in self.J:
            self.b[j] = self.R[j].next_strategy()
        
        # Step 2: construct sequence-form strat
        # Lowkey can use game method for this but idk
        self.x = {seq: 0 for seq in self.Sigma}
        
        @lru_cache(None)
        def recurse(j: str):
            for a in self.J[j]["actions"]:
                p_j = self.game.par_seq[(j, a)]
                if p_j is None:
                    self.x[(j, a)] = self.b[j][a]
                else:
                    self.x[(j, a)] = self.x[p_j] * self.b[j][a]

        for j in self.J:
            recurse(j)
        
        return self.x


if __name__ == "__main__":
    game_type = "kuhn"

    game = Game(f"./efgs/{game_type}.txt")
    cfr = CFR(game, "1")

    pprint(cfr.next_strategy())
