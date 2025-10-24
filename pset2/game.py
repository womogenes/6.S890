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
        all_seqs = {player: set() for player in self.players}

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
                all_seqs[player].add(seq)
                par_seq[seq] = last_seqs[player_idx]

                # DFS
                new_last_seqs = list(last_seqs)
                new_last_seqs[player_idx] = seq
                stack.append((child_hist, tuple(new_last_seqs)))
        
        self.par_seq = par_seq
        self.all_seqs = {player: sorted(seqs) for player, seqs in all_seqs.items()}
        self.seq2idx = [{
            seq: idx for idx, seq in enumerate(seqs)
        } for seqs in self.all_seqs]

    def behav_to_seq(self, player: str, behav_strat: dict):
        """
        Convert behavioral strat to sequential strat.

        behav_strat:
            key: infoset
            value: dict
                key: action
                value: probability
        """
        # We're being lazy screw it

        @lru_cache(None)
        def get_seq_prob(seq):
            """
            Convert probability described in behav_strat to sequential form.
            i.e. multiply everything by parent probability.
            """
            if seq == None:
                return 1
            return behav_strat[seq] * get_seq_prob(self.par_seq[seq])

        res = {
            seq: get_seq_prob(seq)
            for seq in self.all_seqs[player]
        }
        res[None] = 1
        return res

    def is_seq_strat(self, player: str, seq_strat: dict):
        """
        Verify that a strategy is indeed sequential, i.e. that
            x[None] == 1 and
            sum_(a in A_j) x[ja] == x_(p_j), for all j in J.
            i.e. that at every decision (infoset) node, sum of entires of children
                equals entry at parent of node.
        """
        assert seq_strat[None] == 1
        
        for info_hist, info_node in self.infosets.items():
            if info_node["player"] != player:
                continue
            
            actions = info_node["actions"]
            v1 = sum([
                seq_strat[(info_hist, action)] \
                for action in actions
            ])
            v2 = seq_strat[self.par_seq[(info_hist, actions[0])]]
            assert np.isclose(v1, v2)

        return True

    def gen_uniform_behav_strat(self, player: str):
        """
        Given player string, e.g. '1' or '2', generate uniform strategy
        Strategy is represented in behavioral form (oops)
        """
        strat = {}
        infoset_hists = self.get_infosets(player)

        for infoset_hist in infoset_hists:
            infoset = self.infosets[infoset_hist]
            if infoset["player"] != player:
                continue
            actions = infoset["actions"]

            for action in actions:
                strat[(infoset_hist, action)] = 1 / len(actions)

        return strat

    def get_best_response(self, player: str, opp_strat: dict):
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
                            cur_prob * opp_strat[(node["infoset"], action)]
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
                            opp_strat[(node["infoset"], action)] * get_util(child_hist) \
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

    def get_nash_gap(self, p1_strat: dict, p2_strat: dict):
        """
        The Nash gap is defined as
            gamma(x, y) = max_(x* in X) u_1(x*, y) - min(y* in Y) u_1(x, y*)
        For two-player zero-sum games, this reduces to
            max_(x* in X)(x* in X)(x*, y) + max_(y* in Y) u_2(x, y*)
        """
        max_u1 = self.get_best_response("1", p2_strat)[0]
        max_u2 = self.get_best_response("2", p1_strat)[0]
        return max_u1 + max_u2
    
    @lru_cache(None)
    def get_infosets(self, player: str):
        return [
            info_hist \
            for info_hist, infoset in self.infosets.items() if infoset["player"] == player
        ]
    
    def build_tfdp(self, player: str):
        """
        Construct a tree with decision nodes and observation nodes.
        Assumes perfect recall structure of the game.
        """
        tree = {}

        # stack tracks (node id, representative game node)]
        stack = [("/", "/")]
        visited = set()

        while len(stack) > 0:
            tfdp_hist, repr_hist = stack.pop()
            if tfdp_hist in visited:
                continue
            visited.add(tfdp_hist)

            assert repr_hist in self.nodes
            node = self.nodes[repr_hist]

            if node["type"] == "player" and node["player"] == player:
                tree[tfdp_hist] = {
                    "type": "decision",
                    "actions": [],
                    "children": {},
                }

                for action, child_hist in self.get_children(repr_hist):
                    # If child is terminal node
                    child = self.nodes[child_hist]

                    # Player should never make two choices in a row
                    assert not (child["type"] == "player" and child["player"] == player)

                    tfdp_child_hist = f"{tfdp_hist}P{player}:{action}/"
                    stack.append((tfdp_child_hist, child_hist))
                    tree[tfdp_hist]["actions"].append(action)
                    tree[tfdp_hist]["children"][action] = tfdp_child_hist

            else:
                # OBSERVATION NODE
                if node["type"] == "player" or node["type"] == "chance":
                    # If it's a player node, should be opp
                    assert node["type"] != "player" or node["player"] != player
                    
                    tree[tfdp_hist] = {
                        "type": "obs",
                        "signals": [],
                        "children": {},
                    }
                    for action, child_hist in self.get_children(repr_hist):
                        child_node = self.nodes[child_hist]

                        # ASSUME CHILD IS DECISION NODE
                        if child_node["type"] == "terminal":
                            assert node["type"] == "player", "Node before terminal was not player node"
                            tfdp_child_hist = f"{tfdp_hist}O:{action}/"
                        else:
                            assert child_node["type"] == "player" and child_node["player"] == player, \
                                f"Expected opp node to lead to player node\n{child_node}"
                            tfdp_child_hist = child_node["infoset"]

                        tree[tfdp_hist]["signals"].append(action)
                        tree[tfdp_hist]["children"][action] = tfdp_child_hist
                        stack.append((tfdp_child_hist, child_hist))

                else:
                    assert node["type"] == "terminal"
                    tree[tfdp_hist] = {
                        "type": "terminal",
                    }

        return tree


if __name__ == "__main__":
    for game_type in ["rpss", "kuhn", "leduc2"]:
        print(f"\n===== {game_type} =====")
        game = Game(f"./efgs/{game_type}.txt")

        # Uniform strategy for P2
        u2 = game.gen_uniform_behav_strat("2")

        # Best response to U2 for P1
        br1 = game.get_best_response("1", u2)
        print(f"Exp. utility of P1's best response to P2 uniform: {br1[0]:.5f}")

        # Find Nash gap of both players playing uniform
        u1 = game.gen_uniform_behav_strat("1")

        print(f"Nash gap of both players playing uniform: {game.get_nash_gap(u1, u2):.5f}")

        pprint(game.build_tfdp("1"))
