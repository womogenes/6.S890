import numpy as np
from pprint import pprint
from collections import defaultdict
from functools import lru_cache

class Game:
    def __init__(self, game_file: str):
        """
        Represents an extensive form game.

        self.nodes:
            key: history
            value:
                type: player | terminal
                player: str
                actions: str[]
                infoset: str

        self.infosets:
            key: history
            value:
                player: str
                nodes: str[]
                actions: str[]
        """
        self.parse_game_file(game_file)
        self.gen_sequences()

    def parse_game_file(self, game_file: str):
        """
        Take a game file path and extract nodes/infosets
        Returns: nodes, infosets
        """
        with open(game_file) as fin:
            lines = fin.read().strip().split("\n")

        nodes = {}
        infosets = {}
        players = set()

        for line in lines:
            parts = line.split(" ")
            
            if parts[0] == "node":
                hist = parts[1]
                node_type = parts[2]

                if node_type == "player":
                    # Looks like "/P1:r/ player 2 r p s"
                    player = parts[3]
                    players.add(player)
                    actions = parts[5:]
                    nodes[hist] = {
                        "type": "player",
                        "player": player,
                        "actions": tuple(actions),
                    }
                elif node_type == "terminal":
                    # Looks like "node /P1:r/P2:r/ terminal payoffs 1=0 2=0"
                    payoffs = {}
                    for payoff in parts[4:]:
                        p, v = payoff.split("=")
                        payoffs[p] = int(v)

                    nodes[hist] = {
                        "type": "terminal",
                        "payoffs": payoffs
                    }
                else:
                    assert node_type == "chance"
                    actions = []
                    probs = {}
                    for action in parts[4:]:
                        act, p = action.split("=")
                        actions.append(act)
                        probs[act] = float(p)

                    nodes[hist] = {
                        "type": "chance",
                        "actions": actions,
                        "probs": probs,
                    }
            
            else:
                assert parts[0] == "infoset"
                # Looks like "infoset /P1:?/ nodes /P1:r/ /P1:p/ /P1:s/"
                info_hist = parts[1]
                subnodes = parts[3:]

                actions = None
                player = None

                for node_hist in subnodes:
                    node = nodes[node_hist]
                    assert node["type"] == "player"
                    
                    if actions is None:
                        actions = node["actions"]
                    else:
                        # Must have same set of actions
                        assert node["actions"] == actions

                    if player is None:
                        player = node["player"]
                    else:
                        assert node["player"] == player

                    node["infoset"] = info_hist

                infosets[info_hist] = {
                    "player": player,
                    "nodes": tuple(subnodes),
                    "actions": actions,
                }

        self.nodes = nodes
        self.infosets = infosets
        self.players = sorted(players)
        self.n_players = len(self.players)
        self.player2idx = {player: idx for idx, player in enumerate(self.players)}

        assert self.n_players == 2
        assert self.players == ["1", "2"]

    @lru_cache(None)
    def get_children(self, node_hist: str):
        """
        Given a node history label in the game tree, generate list of children
        Returns a list of (action, child_node_hist) pairs
        """
        assert node_hist in self.nodes
        node = self.nodes[node_hist]

        if node["type"] == "player":
            key = f"P{node['player']}"
        elif node["type"] == "chance":
            key = "C"
        else:
            return []

        res = []
        for action in node["actions"]:
            res.append((action, f"{node_hist}{key}:{action}/"))
        return res

    def gen_sequences(self):
        """
        Given we already have self.nodes and self.infosets, generate
            list of sequences with parent relationships
        """
        # Enumerate all sequences by player
        # Deterministic, ordered, etc
        all_seqs = [set() for _ in range(self.n_players)]

        # Stores (hist, (most_recent_seqs))
        stack = [("/", (None,) * self.n_players)]
        visited = set()

        # Parent sequence dict
        par_seq = {}
        
        # Maintain the most recent sequence of each player as we go through the DFS
        while len(stack) > 0:
            node_hist, last_seqs = stack.pop()

            if node_hist in visited:
                continue
            visited.add(node_hist)

            node = self.nodes[node_hist]
            if node["type"] == "terminal":
                continue
            elif node["type"] == "chance":
                # Don't need to update last sequences as this is a chance node
                for _, child_hist in self.get_children(node_hist):
                    stack.append((child_hist, last_seqs))
                continue
            
            player = node["player"]
            player_idx = self.player2idx[player]

            children = self.get_children(node_hist)
            for action, child_hist in children:
                # Mark parent sequence
                seq = (node["infoset"], action)
                all_seqs[player_idx].add(seq)
                par_seq[seq] = last_seqs[player_idx]

                # DFS
                new_last_seqs = list(last_seqs)
                new_last_seqs[player_idx] = seq
                stack.append((child_hist, tuple(new_last_seqs)))
        
        self.par_seq = par_seq
        self.all_seqs = [sorted(seqs) for seqs in all_seqs]
        self.seq2idx = [{
            seq: idx for idx, seq in enumerate(seqs)
        } for seqs in self.all_seqs]

    def gen_uniform_strat(self, player: str):
        """
        Given player string, e.g. '1' or '2', generate uniform strategy
        Strategy is represented in behavioral form (oops)
        """
        strat = {}
        infoset_hists = [
            hist for hist, infoset in self.infosets.items() if infoset["player"] == player
        ]

        for infoset_hist in infoset_hists:
            infoset = self.infosets[infoset_hist]
            if infoset["player"] != player:
                continue
            actions = infoset["actions"]
            strat[infoset_hist] = {action: 1 / len(actions) for action in actions}

        return strat

    def get_best_response(self, player: str, opp_strat: list):
        """
        Finds best response for given player, knowing other player's
            strategy specification.
        """
        # Step 1: get "counterfactual probabilities" for each node
        # And then renormalize within infosets
        # Top-down recursion
        # NOTE: this is conditional on other player's strat
        cf_prob = {}
        stack = [("/", 1.0)]
        while len(stack) > 0:
            node_hist, cur_prob = stack.pop()

            if node_hist in cf_prob:
                continue
            cf_prob[node_hist] = cur_prob

            node = self.nodes[node_hist]
            if node["type"] == "player":
                if node["player"] == player:
                    # Play everything without modify probabilities
                    for _, child_hist in self.get_children(node_hist):
                        stack.append((child_hist, cur_prob))
                else:
                    # Play everything, DO modify probabilities based on other player's strat
                    for action, child_hist in self.get_children(node_hist):
                        stack.append((
                            child_hist,
                            cur_prob * opp_strat[node["infoset"]][action]
                        ))

            elif node["type"] == "chance":
                for action, child_hist in self.get_children(node_hist):
                    stack.append((
                        child_hist,
                        cur_prob * node["probs"][action]
                    ))
        
        # Normalize within infosets
        for infoset in self.infosets.values():
            net_p = sum([cf_prob[subnode] for subnode in infoset["nodes"]])
            for subnode in infoset["nodes"]:
                cf_prob[subnode] /= net_p

        # Step 2: make strategy for player 1
        strat = {}

        # Utility per node
        @lru_cache(None)
        def get_util(hist):
            """
            Get utility of given infoset.
            """
            if (hist in self.nodes) and (hist not in self.infosets):
                # Dealing with individual node here (not infoset)
                node = self.nodes[hist]
                if node["type"] == "terminal":
                    res = node["payoffs"][player]
                elif node["type"] == "chance":
                    res = sum([
                        node["probs"][action] * get_util(child_hist) \
                        for action, child_hist in self.get_children(hist)
                    ])
                else:
                    assert node["type"] == "player"
                    if node["player"] == player:
                        # Delegate to the infoset
                        res = get_util(node["infoset"])
                    else:
                        # Play according to opp strat distribution
                        res = sum([
                            opp_strat[node["infoset"]][action] * get_util(child_hist) \
                            for action, child_hist in self.get_children(hist)
                        ])

            # Dealing with infoset
            else:
                assert hist in self.infosets
                infoset = self.infosets[hist]

                # For each *action*, figure out the expected utility
                E_util = defaultdict(float)

                for subnode_hist in infoset["nodes"]:
                    # Precomputed conterfactual probability for this node
                    subnode_p = cf_prob[subnode_hist]
                    for action, child_hist in self.get_children(subnode_hist):
                        E_util[action] += subnode_p * get_util(child_hist)

                # E_util reports exp. utility of playing each action at infoset
                # Pick action with highest expected utility
                E_util_sorted = sorted(E_util.items(), key=lambda util: -util[1])
                best_action, exp_util = E_util_sorted[0]

                strat[hist] = best_action
                res = exp_util
            
            return res

        return get_util("/"), strat


if __name__ == "__main__":
    game = Game("./efgs/kuhn.txt")

    uniform_2 = game.gen_uniform_strat("2")
    br_1 = game.get_best_response("1", uniform_2)
    pprint(br_1)
