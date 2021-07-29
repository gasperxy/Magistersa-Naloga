from Graph import *
import numpy as np
import pandas as pd
import networkx as nx
import pyodbc
from os import listdir
import os
import time


class GraphRunner:
    """
    A class used to perform 3-weighting algorithem on some specified graphs.
    It can process all graphs in a
    """

    def __init__(self, input_path, folder = False):
        # Input files with multiple graphs in graph6 format
        self.input_file = input_path

        # Specify id input path is as folder or a file
        self.folder = folder

        # List of networkx graphs
        self.graphs = self.read_file(input_path)

        # DataFrame with results of analysis
        self.df = self.create_df()

        # List of all graphs in graph6 format. Used as a index/ID of aggregated tbl
        self.graphs_g6 = self.save_graphs(input_path)

        # DataFrame that wil hold aggregated results of experiments
        self.summary = pd.DataFrame()

        # DataFrame that wil hold aggregated results of experiments in unpivoted format!
        self.unpivoted = pd.DataFrame()

    def read_file(self, path):
        if self.folder:
            nx_graph = []
            for file in listdir(path):
                nx_graph.extend(nx.read_graph6(os.path.join(path, file)))
            return nx_graph
        return nx.read_graph6(path)

    def save_graphs(self, path):
        if self.folder:
            graphs = []
            for file in listdir(path):
                f = open(os.path.join(path, file), "r")
                graphs.extend(f.readlines())
            return graphs
        f = open(path, "r")
        return f.readlines()


    def create_df(self):
        df= pd.DataFrame(columns=["Graph", "n", "m", "minDeg", "maxDeg", "Time", "RandomSolvable", "LocalSolvable","RecursiveSolvable", "RecursiveDepth"])
        df = df.astype({"n": int,"m": int,"minDeg": int,"maxDeg": int, "Time":float, "RandomSolvable": float,"LocalSolvable": float,"RecursiveSolvable": float, "RecursiveDepth": float})
        return df

    def analyze(self, rep=10):
        for i, graph in enumerate(self.graphs):
            g = Graph(graph)
            gid = self.graphs_g6[i]
            n = len(graph.nodes)
            m = len(graph.edges)
            ddg = g.graph.degree()
            degrees = [x[1] for x in g.graph.degree()]
            d = min(degrees)
            D = max(degrees)

            for _ in range(rep):
                start_time = time.time()
                g.randomize_weights()
                if len(g.conflicts) == 0:
                    # Graph in solvable using random weights
                    end_time = time.time() - start_time
                    r = pd.Series([gid, n, m, d, D,end_time, 1, 0, 0, np.nan], index=self.df.columns)
                    self.df = self.df.append(r, ignore_index=True)
                    continue

                start_time = time.time()
                succ = g.solve()
                if succ:
                    # Graph is solvable using local search!
                    end_time = time.time() - start_time
                    r = pd.Series([gid, n, m,d, D,end_time, 0, 1, 0, np.nan], index=self.df.columns)
                    self.df = self.df.append(r, ignore_index=True)
                    continue

                start_time = time.time()
                solved_graph = solve_recursive(g, h=1)
                end_time = time.time() - start_time
                d = len(solved_graph.history)
                r = pd.Series([gid, n, m,d, D,end_time, 0, 0, 1, d], index=self.df.columns)
                self.df = self.df.append(r, ignore_index=True)
            if i % 100 == 0:
                print('Processed ' + str(i) + ' graphs.')

    def summerize(self):
        agg = self.df.groupby('Graph').agg(
            {
                "n": np.mean,
                "m": np.mean,
                "minDeg": np.min,
                "maxDeg": np.max,
                "Time": np.mean,
                "RandomSolvable": np.mean,
                "LocalSolvable": np.mean,
                "RecursiveSolvable": np.mean,
                "RecursiveDepth":np.nanmean
            }
        ).reset_index()
        self.summary = self.summary.append(agg, ignore_index=True)

    def unpivot(self):
        unpivoted = self.summary.melt(id_vars=['Graph', 'n', 'm', 'minDeg', 'maxDeg', 'Time'], value_vars=['RandomSolvable', 'LocalSolvable','RecursiveSolvable', 'RecursiveDepth'])
        self.unpivoted = self.unpivoted.append(unpivoted)

    def export_summary_sql(self, tbl_name):
        self.bulk_upload_sql(self.summary, tbl_name)

    def export_unpivoted_sql(self, tbl_name):
        self.bulk_upload_sql(self.unpivoted, tbl_name, pivot=False)

    def export_results_xlsx(self, file_name):
        self.summary.to_excel(file_name)

    def export_unpivoted_xlsx(self, file_name):
        self.unpivoted.to_excel(file_name)

    def export_results_csv(self, file_name):
        self.summary.to_csv(file_name,index=False)

    def export_unpivoted_csv(self, file_name):
        self.unpivoted.to_csv(file_name, index=False)

    def bulk_upload_sql(self, df, table_name, pivot=True):
        server = 'netflow.database.windows.net'
        database = 'DW_Storage_Netica'
        username = 'gasperxy'
        password = 'Asdf1234'
        driver = '{SQL Server Native Client 11.0}'
        conn =  pyodbc.connect('DRIVER='+driver+';SERVER=tcp:'+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
        insert_to_tmp_tbl_stmt =  f"INSERT INTO {table_name} VALUES (?,?,?,?,?,?,?)" if not pivot else f"INSERT INTO {table_name} VALUES (?,?,?,?,?,?,?, ?, ?)"
        cursor = conn.cursor()
        cursor.fast_executemany = True
        cursor.executemany(insert_to_tmp_tbl_stmt, df.values.tolist())
        print(f'{len(df)} rows inserted to the {table_name} table')
        cursor.commit()
        cursor.close()
        conn.close()





analysis = GraphRunner("graph_examples_100", folder=True)
analysis.analyze()
analysis.summerize()
analysis.unpivot()
analysis.export_unpivoted_xlsx('graph_results/results.xlsx')
analysis.export_unpivoted_csv('graph_results/results.csv')
analysis.export_unpivoted_sql("graph_results_agg_2")


#analysis.summerize_xlsx(r"C:\Users\gaspe\Documents\Magisterska\Magistersa-Naloga\Koda\g_test.xlsx", False)







