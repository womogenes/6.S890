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
import matplotlib.pyplot as plt

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
        self.x.append(strat)
        return {a: float(strat[i]) for i, a in enumerate(self.A)}
        
    def observe_utility(self, g):
        # g is a dict, turn it into a gradient vector
        g = np.array([g[a] for a in self.A])

        prev_r = self.r[-1]
        assert g.shape == prev_r.shape, "Grad vector shape does not match regret shape"

        self.r.append(prev_r + g - np.dot(g, self.x[-1]) * np.ones_like(prev_r))

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
        self.player = player

    def next_strategy(self, as_vec=False):
        self.b = {}

        # Step 1: get local strategies per decision node
        for j in self.J:
            self.b[j] = self.R[j].next_strategy()
        
        # Step 2: construct sequence-form strat
        # Lowkey can use game method for this but idk
        
        @lru_cache(None)
        def x(seq: tuple):
            """
            This is just for computing sequence-form strat.
            """
            if seq is None:
                return 1
            
            j, a = seq
            for a in self.J[j]["actions"]:
                p_j = self.game.par_seq[(j, a)]
                if p_j is None:
                    return self.b[j][a]
                else:
                    return x(p_j) * self.b[j][a]

        self.x = {seq: x(seq) for seq in self.Sigma}
        self.x[None] = 1

        game.verify_seq_strat(self.player, self.x)

        if as_vec:
            return np.array([self.x[seq] for seq in self.Sigma])
        
        return self.x
    
    def observe_util(self, g: dict, as_vec=False):
        """
        g is a "gradient vector" dict that has a float for everything in self.Sigma
        """
        if as_vec:
            # Convert vector to dict
            g = {seq: g[idx] for idx, seq in enumerate(self.Sigma)}

        @lru_cache(None)
        def V(v: str):
            """
            Some recursion stuff ?? idk it's specified by the pseudocode
            """
            if v is None:
                return 0
            
            if v in self.J:
                j = v
                return sum([
                    self.b[j][a] * (g[(j, a)] + V(child)) \
                    for a, child in self.J[j]["children"].items()
                ])
            elif v in self.K:
                k = v
                return sum([V(child) for _, child in self.K[k]["children"].items()])
            else:
                # This is probably fine
                return 0

        # Local counterfactual utilities
        for j in self.J:
            gj = {}
            for a, child in self.J[j]["children"].items():
                gj[a] = g[(j, a)] + V(child)
            
            self.R[j].observe_utility(gj)


if __name__ == "__main__":
    game_type = "kuhn"

    game = Game(f"./efgs/{game_type}.txt")
    cfr = CFR(game, "1")

    cfr.next_strategy()
    cfr.observe_util({ seq: 0 for seq in cfr.game.all_seqs["1"] })

    # PROBLEM 5.2: CFR against uniform strategy
    MAX_T = 10

    n1 = len(game.all_seqs["1"])
    n2 = len(game.all_seqs["2"])
    x_hist = np.zeros((MAX_T, n1))
    y_hist = np.zeros((MAX_T, n2))

    p1_util_hist = [None] * MAX_T

    # Fix p2 to be uniform strat
    y = game.seq2vec("2", game.behav_to_seq("2", game.gen_uniform_behav_strat("2")))
    cfr1_uniform = CFR(game, "1")

    for t in range(MAX_T):
        x = cfr1_uniform.next_strategy(as_vec=True)
        x_hist[t] = x
        # y is static here
        y_hist[t] = y

        g = game.M @ y
        cfr1_uniform.observe_util(g, as_vec=True)

        x_avg = np.mean(x_hist[:t+1], axis=0)
        y_avg = np.mean(y_hist[:t+1], axis=0)

        p1_util_hist[t] = x_avg @ game.M @ y_avg
        print(p1_util_hist)
    
    plt.plot(p1_util_hist)
    plt.show()
