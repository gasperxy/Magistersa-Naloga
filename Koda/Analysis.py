from Graph import *
import numpy as np
import pandas as pd
import networkx as nx
import pyodbc


class GraphRunner:
    def __init__(self, input_file):
        # Input files with multiple graphs in graph6 format
        self.input_file = input_file

        # List of networkx graphs
        self.graphs = self.read_file(input_file)

        # DataFrame with results of analysis
        self.df = self.create_df()

        self.graphs_g6 = self.save_graphs(input_file)

    def read_file(self, file):
        return nx.read_graph6(file)

    def save_graphs(self, file):
        f = open(file, "r")
        return  f.readlines()



    def create_df(self):
        df= pd.DataFrame(columns=["Graph", "n", "m", "minDeg", "maxDeg", "RandomSolvable", "LocalSolvable","RecursiveSolvable", "RecursiveDepth"])
        df = df.astype({"n": int,"m": int,"minDeg": int,"maxDeg": int, "RandomSolvable": float,"LocalSolvable": float,"RecursiveSolvable": float, "RecursiveDepth": float})
        return df

    def analyze(self, rep=10):
        for i, graph in enumerate(self.graphs):
            g = Graph(graph)
            gid = self.graphs_g6[i]
            n = len(graph.nodes)
            m = len(graph.edges)

            d = min(g.graph.degree())[1]
            D = max(g.graph.degree())[1]

            for _ in range(rep):
                g.randomize_weights()
                if len(g.conflicts) == 0:
                    # Graph in solvable using random weights
                    r = pd.Series([gid, n, m, d, D, 1, 0, 0, np.nan], index=self.df.columns)
                    self.df = self.df.append(r, ignore_index=True)
                    continue

                succ = g.solve()
                if succ:
                    # Graph is solvable using local search!
                    r = pd.Series([gid, n, m,d, D, 0, 1, 0, np.nan], index=self.df.columns)
                    self.df = self.df.append(r, ignore_index=True)
                    continue

                solved_graph = solve_recursive(g, h=1)
                d = len(solved_graph.history)
                r = pd.Series([gid, n, m,d, D, 0, 0, 1, d], index=self.df.columns)
                self.df = self.df.append(r, ignore_index=True)
            if i % 100 == 0:
                print('Processed ' + str(i) + ' graphs.')

    def summerize_sql(self, tbl_name):
        agg = self.df.groupby('Graph').agg(
            {
                "n":np.mean,
                "m":np.mean,
                "minDeg":np.mean,
                "maxDeg":np.mean,
                "RandomSolvable" :np.mean,
                "LocalSolvable": np.mean,
                "RecursiveSolvable": np.mean,
                "RecursiveDepth" : lambda x:  np.count_nonzero(x) / np.size(x)
            }
        ).reset_index()
        print(agg)
        pivoted = agg.melt(id_vars=['Graph', 'n', 'm', 'minDeg', 'maxDeg'], value_vars=['RandomSolvable', 'LocalSolvable','RecursiveSolvable', 'RecursiveDepth'])
        self.bulk_upload_sql(pivoted, tbl_name)

    def summerize_xlsx(self, file_name, pivot=True):
        agg = self.df.groupby('Graph').agg(
            {
                "n": np.mean,
                "m": np.mean,
                "minDeg": np.mean,
                "maxDeg": np.mean,
                "RandomSolvable": np.mean,
                "LocalSolvable": np.mean,
                "RecursiveSolvable": np.mean,
                "RecursiveDepth": np.nanmean
            }
        ).reset_index()
        if pivot:

            pivoted = agg.melt(id_vars=['Graph', 'n', 'm', 'minDeg', 'maxDeg'],
                               value_vars=['RandomSolvable', 'LocalSolvable', 'RecursiveSolvable', 'RecursiveDepth'])
            pivoted.to_excel(file_name)
        else:
            agg.to_excel(file_name)

    def bulk_upload_sql(self, df, table_name):
        server = 'netflow.database.windows.net'
        database = 'DW_Storage_Netica'
        username = 'gasperxy'
        password = 'Asdf1234'
        driver = '{SQL Server Native Client 11.0}'
        conn =  pyodbc.connect('DRIVER='+driver+';SERVER=tcp:'+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
        insert_to_tmp_tbl_stmt = f"INSERT INTO {table_name} VALUES (?,?,?,?,?,?,?)"
        cursor = conn.cursor()
        cursor.fast_executemany = True
        cursor.executemany(insert_to_tmp_tbl_stmt, df.values.tolist())
        print(f'{len(df)} rows inserted to the {table_name} table')
        cursor.commit()
        cursor.close()
        conn.close()





analysis = GraphRunner("graph_examples/random_1000_60.txt")
analysis.analyze()
analysis.summerize_sql('graph_results_agg')

analysis = GraphRunner("graph_examples/random_1000_70.txt")
analysis.analyze()
analysis.summerize_sql('graph_results_agg')

analysis = GraphRunner("graph_examples/random_1000_80.txt")
analysis.analyze()
analysis.summerize_sql('graph_results_agg')
#analysis.summerize_xlsx(r"C:\Users\gaspe\Documents\Magisterska\Magistersa-Naloga\Koda\g_test.xlsx", False)







