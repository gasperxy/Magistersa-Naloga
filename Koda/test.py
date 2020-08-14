from Graph import *
import sys

def test():

    fname = sys.argv[1]
    fname = "graph_examples/"+fname

    graphs = read_graph6(fname)

    for graph in graphs:
        g = Graph(graph)
        g.randomize_weights()
        s = solve_recursive(g)
        if not s:
            print("Graph is not solvable!")
            g.draw()

    print("Solved graphs!")



if __name__ == '__main__':
    test()
