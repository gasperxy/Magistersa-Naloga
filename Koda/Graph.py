import networkx as nx
import random as rnd
import matplotlib.pyplot as plt




class Graph:
    def __init__(self, G):
        self.graph = G  # networkx graph object
        self.edge_weights = {e: 0 for e in self.graph.edges}  # of type dict[tuple(int, int), int]
        self.sums = {0: [i for i in self.graph.nodes]}  # of type dict[int, list[int]]
        self.node_sums = {i: 0 for i in self.graph.nodes}  # dict with node sums
        self.conflicts = {e for e in self.graph.edges}  # set of conflicts
        self.history = {} # dict where history of edge weight modifications will be stored

    def clone(self):
        g = Graph(self.graph)
        g.edge_weights = {e: self.edge_weights[e] for e in self.edge_weights}
        g.sums = {s: [i for i in self.sums[s]] for s in self.sums}
        g.node_sums = {n: self.node_sums[n] for n in self.node_sums}
        g.conflicts = {c for c in iter(self.conflicts)}
        g.history = {e: self.history[e] for e in self.history}

        return g

    def get_edge_weight(self, e):
        """
        Returns edge weight for an edge e. It checks if e = (u, v) in edge_weights or of
        e = (v, u) in edge weights.
        :param e:
        :return:
        """
        if e in self.edge_weights:
            return self.edge_weights[e]
        else:
            return self.edge_weights[(e[1], e[0])]

    def randomize_weights(self):
        """
        Assings random weights to edges (1,3) and set up a sums dict
        :return:
        """
        rnd.seed(30)
        for e in self.graph.edges:
            self.modify_weight(e, rnd.randint(1, 3))

    def unit_weights(self, w):
        """
        Assigns weight w to all edges in a graph
        :param w:
        :return:
        """

        for e in self.graph.edges:
            self.modify_weight(e, w)

    def modify_weight(self, e, w):
        """
        Sets a new weight w for an edge e. Doing so also corrects lookup objects for sum conflicts.
        :param e:
        :param w:
        :return:
        """
        old_weight = self.get_edge_weight(e)

        # Set a new weight
        self.edge_weights[e] = w

        # Remove nodes:
        self.sums[self.node_sums[e[0]]].remove(e[0])
        self.sums[self.node_sums[e[1]]].remove(e[1])

        # Remove from conflicts any edge that was in beetwen n1 and sums[n1]
        for u in self.graph.neighbors(e[0]):
            if (u, e[0]) in self.conflicts:
                self.conflicts.remove((u, e[0]))
            if (e[0], u) in self.conflicts:
                self.conflicts.remove((e[0], u))

        for u in self.graph.neighbors(e[1]):
            if (u, e[1]) in self.conflicts:
                self.conflicts.remove((u, e[1]))
            if (e[1], u) in self.conflicts:
                self.conflicts.remove((e[1], u))

        # Modify nodes sum
        self.node_sums[e[0]] += (w - old_weight)
        self.node_sums[e[1]] += (w - old_weight)

        # Add nodes back:
        if self.node_sums[e[0]] in self.sums:
            self.sums[self.node_sums[e[0]]].append(e[0])
        else:
            self.sums[self.node_sums[e[0]]] = [e[0]]

        # Add nodes back:
        if self.node_sums[e[1]] in self.sums:
            self.sums[self.node_sums[e[1]]].append(e[1])
        else:
            self.sums[self.node_sums[e[1]]] = [e[1]]

        # Add newly created conflicts
        for u in self.graph.neighbors(e[0]):
            if u in self.sums[self.node_sums[e[0]]]:
                self.conflicts.add((e[0], u))

        for u in self.graph.neighbors(e[1]):
            if u in self.sums[self.node_sums[e[1]]]:
                self.conflicts.add((e[1], u))

    def solve_conflict(self, c=None, in_depth=False):
        """
        Function takes a single conflict as an edge = (u, v) and try to solve it. Since a conflict can only be solved
        by modifying weights of {u} x N(u) union {v} x N(v) without (u,v) it modifies weight on all those edges
        and saves the number of conflicts after modification. At the end it performs modification with smallest number of conflicts.
        :param c:
        :param in_depth:
        :return:
        """

        c = self.conflicts.pop() if c is None else c

        # Save conflicts size before weight modification
        conflicts_size = len(self.conflicts)

        v1 = c[0]
        v2 = c[1]

        # Go throught all neighbors of v1 and v2 and modify weight. Save length of conflits after change
        conflict_changes = {}

        for u in self.graph.neighbors(v1):
            if u != v2:
                # (v1, u) is an edge where changing weight will resolve conflict c
                w = self.get_edge_weight((v1, u))
                nw = 2 if w == 3 else 1 if w == 2 else 2
                if (v2, u) not in self.history and (u, v2) not in self.history:
                    self.modify_weight((v1, u), nw)
                    conflict_changes[(v1, u)] = len(self.conflicts)
                    # Undo modification
                    self.modify_weight((v1, u), w)

        for u in self.graph.neighbors(v2):
            if u != v1:
                # (v1, u) is an edge where changing weight will resolve conflict c
                w = self.get_edge_weight((v2, u))
                nw = 2 if w == 3 else 1 if w == 2 else 2
                if (v2, u) not in self.history and (u, v2) not in self.history:
                    self.modify_weight((v2, u), nw)
                    conflict_changes[(v2, u)] = len(self.conflicts)
                    # Undo modification
                    self.modify_weight((v2, u), w)

        # Get the modification with minimum number of conflicts
        e = min(conflict_changes, key=conflict_changes.get)

        if conflict_changes[e] <= conflicts_size:
            w = self.get_edge_weight(e)
            nw = 2 if w == 3 else 1 if w == 2 else 2
            self.modify_weight(e, nw)
        else:
            #print("No modification can decrese number of conflicts.")
            if in_depth:
                # Even thoug number of conflicts does not decrease still make weight modification
                w = self.get_edge_weight(e)
                nw = 2 if w == 3 else 1 if w == 2 else 2
                if e not in self.history:
                    self.modify_weight(e, nw)
            else:
                return False
        return True

    def solve(self, max_depth=0):
        """
        Function tries to solve all conflicts in a graph. It solves conflict after conflict
        until there are no more conflicts or solve conflict function can not find a modification that decreaser number
        of conflicts.
        :param max_depth:
        :return:
        """
        depth = 0

        while len(self.conflicts) != 0:

            # get a conflict from a difference
            c = self.conflicts.pop()
            success = self.solve_conflict(c=c)

            if not success:
                if depth < max_depth:
                    self.solve_conflict(c=c, in_depth=True)
                    depth += 1
                else:
                    self.conflicts.add(c)
                    #print("Cannot solve graph")
                    return False
        return True

    def draw(self):
        """
        Draws a graph with edge weights and node sums.
        :return:
        """
        node_labels = {n: str(n) + "," + str(self.node_sums[n]) for n in self.node_sums}
        pos = nx.spring_layout(self.graph)
        plt.figure()
        nx.draw(self.graph, pos)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=self.edge_weights, font_color='red')
        nx.draw_networkx_labels(self.graph, pos, node_labels)
        plt.axis('off')
        plt.show()


def solve_recursive(graph, depth=0):
        """
        Recursively tries to solve graph weightening.
        :param graph:
        :param depth:
        :return:
        """

        # First try to solve it using local search algorithm
        graph.solve()

        if len(graph.conflicts) > 1:
            # Pop a random conflict
            un_visited_conflicts = graph.conflicts - set(graph.history.keys())
            if len(un_visited_conflicts) == 0:
                print("There are not any unmarked conflicts left")
                return False
            c = un_visited_conflicts.pop()

            # Create a copy, where w(c) = 1
            g1 = graph.clone()
            g1.modify_weight(c, 1)
            g1.history[c] = (1, depth)

            g1_result = solve_recursive(g1, depth+1)

            if not g1_result:
                return g1_result

            # Create a copy, where w(c) = 1
            g2 = graph.clone()
            g2.modify_weight(c, 2)
            g2.history[c] = (2, depth)

            g2_result = solve_recursive(g2, depth + 1)

            if not g2_result:
                return g2_result

            # Create a copy, where w(c) = 3
            g3 = graph.clone()
            g3.modify_weight(c, 3)
            g3.history[c] = (3, depth)

            g3_result = solve_recursive(g3, depth + 1)

            if not g3_result:
                return g3_result

            # If execution comes to this line, then algorithm did not find a solution
            print("End of execution")
            return False
        else:
            return graph














def read_graph6(file_path):
    """
    Reads the file and returns array of nx.Graph object, one for each
    line of input file.
    :param file_path:
    :return:
    """
    G = nx.read_graph6(file_path)
    return G

#
# G = read_graph6("graph_examples/graphs_5.txt")
# g = Graph(G[5])
# print(g.graph.edges)
# g.randomize_weights()
# print(g.conflicts)
# print(g.edge_weights)
# print(g.node_sums)
# print(g.sums)
#
# g.solve_conflict(c=(4,3))
# print("solving conflict")
# print(g.conflicts)
# print(g.edge_weights)
# print(g.node_sums)
# print(g.sums)


