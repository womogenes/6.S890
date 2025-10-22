import numpy as np
from pprint import pprint

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
                    probs = []
                    for action in parts[4:]:
                        act, p = action.split("=")
                        actions.append(act)
                        probs.append(float(p))

                    nodes[hist] = {
                        "type": "chance",
                        "actions": actions,
                        "probs": probs,
                    }
            
            else:
                assert parts[0] == "infoset"
                # Looks like "infoset /P1:?/ nodes /P1:r/ /P1:p/ /P1:s/"
                info_hist = parts[1]
                info_nodes = parts[3:]

                actions = None
                player = None

                for node_hist in info_nodes:
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
                    "nodes": tuple(info_nodes),
                    "actions": actions,
                }

        self.nodes = nodes
        self.infosets = infosets
        self.players = sorted(players)
        self.n_players = len(self.players)
        self.player2idx = {player: idx for idx, player in enumerate(self.players)}

    def get_children(self, node_hist: str):
        """
        Given a node history label in the game tree, generate list of children
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
                par_seq[seq] = last_seqs[player_idx]

                # DFS
                new_last_seqs = list(last_seqs)
                new_last_seqs[player_idx] = seq
                stack.append((child_hist, tuple(new_last_seqs)))
        
        self.par_seq = par_seq


if __name__ == "__main__":
    game = Game("./efgs/kuhn.txt")

    print("===== NODES =====")
    pprint(game.nodes)
    print()
    print("===== INFOSETS =====")
    pprint(game.infosets)
    print()
    print("===== PLAYERS =====")
    print(game.players)
    print()
    print("===== PAR_SEQ =====")
    pprint(game.par_seq)
