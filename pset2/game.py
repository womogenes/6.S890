import numpy as np
from pprint import pprint
from collections import defaultdict
from functools import lru_cache
import re

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
        self.gen_payoff_mat()

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
        self.all_seqs = {player: [None] + sorted(seqs) for player, seqs in all_seqs.items()}
        self.seq2idx = [{
            seq: idx for idx, seq in enumerate(seqs)
        } for seqs in self.all_seqs]


    def gen_payoff_mat(self):
        """Payoff matrix compatible with full sequential-form vectors."""
        seqs1, seqs2 = self.all_seqs["1"], self.all_seqs["2"]
        s1idx = {s: i for i, s in enumerate(seqs1)}
        s2idx = {s: i for i, s in enumerate(seqs2)}
        M = np.zeros((len(seqs1), len(seqs2)))

        def dfs(node_hist, cur_s1, cur_s2, chance_p=1.0):
            node = self.nodes[node_hist]
            if node["type"] == "terminal":
                M[s1idx[cur_s1], s2idx[cur_s2]] += chance_p * node["payoffs"]["1"]
                return

            if node["type"] == "chance":
                for action, child in self.get_children(node_hist):
                    dfs(child, cur_s1, cur_s2, chance_p * node["probs"][action])
                return

            # player node
            p = node["player"]
            for action, child in self.get_children(node_hist):
                seq = (node["infoset"], action)
                if p == "1":
                    dfs(child, seq, cur_s2, chance_p)
                else:
                    dfs(child, cur_s1, seq, chance_p)

        dfs("/", None, None, 1.0)
        self.M = M


    def get_cf_prob(self, player: str, opp_seq_strat: dict):
        """
        seq_strat is opponent's strat in sequential form.
        This calculates the likelihood of reaching each node
            given we always play to get to that node, but opponent
            and chance play according to randomness.
        """
        cf_prob = {}

        for node_hist in self.nodes:
            if node_hist == "/":
                cf_prob[node_hist] = 1
                continue

            parts = node_hist.strip("/").split("/")
            prob = 1
            cur_hist = "/"
            for part in parts:
                party, action = part.split(":")
                if party[1:] == player:
                    cur_hist += f"{part}/"
                    continue

                cur_node = self.nodes[cur_hist]
                if cur_node["type"] == "player":
                    assert cur_node["player"] != player
                    prob *= opp_seq_strat[(cur_node["infoset"], action)]

                else:
                    # We can't be at a terminal node
                    assert cur_node["type"] == "chance"
                    prob *= cur_node["probs"][action]

                cur_hist += f"{part}/"

            cf_prob[node_hist] = prob

        return cf_prob

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
        return res

    def seq2vec(self, player: str, seq_strat: dict):
        """
        Vectorize
        """
        return np.array([seq_strat[seq] for seq in self.all_seqs[player]])

    def verify_seq_strat(self, player: str, seq_strat: dict):
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
    
    def get_best_response_seq(self, player: str, opp_seq_strat: np.ndarray):
        """
        Convert sequential strategy in vector form to behavioral strategy
            so that we can use it with get_best_response.
        """
        pass

    def get_best_response(self, player: str, opp_behav_strat: dict):
        """
        Finds best response for given player, knowing other player's
            strategy specification.
        """
        # Step 1: get "counterfactual probabilities" for each node
        # And then renormalize within infosets
        # Top-down recursion
        # NOTE: this is conditional on other player's strat
        opp = "2" if player == "1" else "1"

        opp_seq_strat = self.behav_to_seq(opp, opp_behav_strat)
        cf_prob = self.get_cf_prob(player, opp_seq_strat)
        
        # Normalize within infosets
        for infoset in self.infosets.values():
            net_p = sum([cf_prob[subnode] for subnode in infoset["nodes"]])
            for subnode in infoset["nodes"]:
                cf_prob[subnode] /= net_p

        # Step 2: make strategy for player 1
        strat = {seq: 0 for seq in self.all_seqs[player]}

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
                            opp_behav_strat[(node["infoset"], action)] * get_util(child_hist) \
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

                strat[(hist, best_action)] = 1
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
        Build a TFDP tree for the given player.
        Decision nodes: infoset labels (J)
        Observation nodes: history prefixes up to next player decision (K)
        Terminal nodes: full history strings
        """
        tree = {}

        def join_parts(parts):
            if len(parts) == 0:
                return "/"
            return f"/{'/'.join(parts)}/"

        # Exploit strings <3
        # NOTE: terminal nodes not included, but they don't matter for CFR
        for info_hist, infoset in self.infosets.items():
            if infoset["player"] != player:
                continue
                
            # Add children
            tree[info_hist] = {
                "type": "decision",
                "actions": [],
                "children": {},
            }
            for action in infoset["actions"]:
                tree[info_hist]["actions"].append(action)
                tree[info_hist]["children"][action] = f"{info_hist}P{player}:{action}/"

            parts = info_hist.strip("/").split("/")
            if info_hist == "/":
                continue
            
            # Backtrack until we reach an info node
            for i in range(len(parts), -1, -1):
                pre_parts = join_parts(parts[:i])
                pre_pre_parts = join_parts(parts[:i-1])

                # One extra piece of information for pre_pre_parts node
                if pre_pre_parts in self.infosets:
                    assert parts[i-1].startswith(f"P{player}")
                    break

                if pre_pre_parts not in tree:
                    tree[pre_pre_parts] = {
                        "type": "obs",
                        "signals": [],
                        "children": {},
                    }
                
                last_sig = parts[i-1]
                if last_sig in tree[pre_pre_parts]["children"]:
                    continue

                tree[pre_pre_parts]["signals"].append(last_sig)
                tree[pre_pre_parts]["children"][last_sig] = pre_parts

        # We should include terminal nodes here
        # They are probably important to fill in action space
        # Yes, use regex because we are lazy
        for terminal_hist, terminal_node in self.nodes.items():
            if terminal_node["type"] != "terminal":
                continue

            parts = terminal_hist.strip("/").split("/")
            pre_parts = join_parts(parts[:-1])
            
            final_player, final_action = parts[-1].split(":")
            new_term_nodes = {}
            terminal_hist_in_tree = None

            for tree_hist, tree_node in tree.items():
                # Does this match a decision node?
                hist_re = tree_hist.replace("?", ".")

                if not re.fullmatch(hist_re, pre_parts):
                    continue

                if tree_node["type"] == "decision":
                    assert final_player[1:] == player
                    if final_action in tree_node["children"]:
                        continue
                    tree_node["actions"].append(final_action)
                    tree_node["children"][final_action] = None

                else:
                    assert tree_node["type"] == "obs"
                    if parts[-1] in tree_node["children"]:
                        continue
                    tree_node["signals"].append(parts[-1])
                    tree_node["children"][parts[-1]] = None

                # Add to tree as observation node for completeness
                terminal_hist_in_tree = f"{tree_hist}{parts[-1]}/"\

            else:
                # In this case, the terminal is the final value and not in an infoset
                #   simply because the game has ended at this point.
                terminal_hist_in_tree = None
            
            # Insert terminal node info
            if terminal_hist_in_tree is not None:
                tree[terminal_hist_in_tree] = {
                    "type": "obs",
                    "signals": [],
                    "children": {},
                }

        return tree



if __name__ == "__main__":
    for game_type in ["rpss", "kuhn", "leduc2"]:
        print(f"\n===== {game_type} =====")
        game = Game(f"./efgs/{game_type}.txt")

        # Uniform strategy for P2
        u2 = game.gen_uniform_behav_strat("2")

        # Best response to U2 for P1
        br1_exp, br1 = game.get_best_response("1", u2)
        print(f"Exp. utility of P1's best response to P2 uniform: {br1_exp:.5f}")

        # Find Nash gap of both players playing uniform
        u1 = game.gen_uniform_behav_strat("1")

        print(f"Nash gap of both players playing uniform: {game.get_nash_gap(u1, u2):.5f}")

        # pprint(game.build_tfdp("1"))

        u2_seq = game.behav_to_seq("2", u2)
        br1_seq = game.behav_to_seq("1", br1)

        u2_seq_vec = game.seq2vec("2", u2_seq)
        br1_seq_vec = game.seq2vec("1", br1_seq)
        print(u2_seq_vec.shape, br1_seq_vec.shape)
