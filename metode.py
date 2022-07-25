import json
import re
import pandas as pd
import openpyxl
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from flask import request
from requests import Response



awal = np.array([])
akhir = np.array([])
def AddData():
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ_FeidgeJH61pcpyFZ68e4ob7CgjxmrF5Z9DG_ruJDizPD7sLdAkfJfe-UeTwV-KX2ARXIRdiHudB4/pub?output=xlsx"
    wks = pd.ExcelFile(url,"openpyxl")
    nodes = pd.read_excel(wks, "Nodes")
    routes = pd.read_excel(wks, "Routes")
    latAwal = request.form["edtLatawal"]
    longAwal = request.form["edtLongawal"]
    latAkir = request.form["edtLatakhir"]
    longAkhir = request.form["edtLongakhir"]
    print(latAwal)
    print(longAwal)
    print(latAkir)
    print(longAkhir)
    awal = np.append(float(latAwal), float(longAkhir))
    akhir = np.append(float(latAkir), float(longAkhir))
    print(awal, akhir)
    def min_distance(point1, points):
        distances = [np.linalg.norm(point1 - p) for p in points]
        minimum = min(distances)
        return distances.index(minimum)
    points = nodes.iloc[:, 2:4].values
    contoh_awal = awal#np.append(float(latAwal), float(longAwal))
    contoh_akhir = akhir#np.append(float(latAkhir), float(longAkhir))
    print(contoh_awal, contoh_akhir)

    index_awal = min_distance(contoh_awal, points)
    titik_awal = nodes.iloc[index_awal, :]
    print(titik_awal)

    index_akhir = min_distance(contoh_akhir, points)
    titik_akhir = nodes.iloc[index_akhir, :]
    print(titik_akhir)
    
    class Graph():
        def __init__(self):
            """
            self.edges is a dict of all possible next nodes
            e.g. {'X': ['A', 'B', 'C', 'E'], ...}
            self.weights has all the weights between two nodes,
            with the two nodes as a tuple as the key
            e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
            """
            self.edges = defaultdict(list)
            self.weights = {}

        def add_edge(self, from_node, to_node, weight):
            # Note: assumes edges are bi-directional
            self.edges[from_node].append(to_node)
            self.edges[to_node].append(from_node)
            self.weights[(from_node, to_node)] = weight
            self.weights[(to_node, from_node)] = weight


    def dijsktra(graph, initial, end):
        # shortest paths is a dict of nodes
        # whose value is a tuple of (previous node, weight)
        shortest_paths = {initial: (None, 0)}
        current_node = initial
        visited = set()

        while current_node != end:
            visited.add(current_node)
            destinations = graph.edges[current_node]
            weight_to_current_node = shortest_paths[current_node][1]

            for next_node in destinations:
                weight = graph.weights[(current_node, next_node)] + weight_to_current_node
                if next_node not in shortest_paths:
                    shortest_paths[next_node] = (current_node, weight)
                else:
                    current_shortest_weight = shortest_paths[next_node][1]
                    if current_shortest_weight > weight:
                        shortest_paths[next_node] = (current_node, weight)

            next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
            if not next_destinations:
                return "Route Not Possible"
            # next node is the destination with the lowest weight
            current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

        # Work back through destinations in shortest path
        path = []
        while current_node is not None:
            path.append(current_node)
            next_node = shortest_paths[current_node][0]
            current_node = next_node
        # Reverse path
        path = path[::-1]
        return path

    g = Graph()
    for index, row in routes.iterrows():
        g.add_edge(row.node_origin, row.node_dest, row.criminal_weight)
    routing = dijsktra(g, index_awal, index_akhir)
    print(routing)

    point_routing = []
    coordinates = []
    distance_meter = 0
    for i in routing:
      point_routing.append(nodes.iloc[int(i)])
    df_point_routing = pd.DataFrame(point_routing)

    for i in range(len(routing)-1):
        hehe = routes[(routes["node_origin"] == routing[i]) & (routes["node_dest"] == routing[i+1])]
        route_coords = json.loads(hehe.route_coord.values[0])
        coordinates += route_coords
        distance_meter += hehe.distance_meter.values[0]

    c = np.arange(len(coordinates))
    df_coordinates = pd.DataFrame(coordinates)
    df_coordinates.plot(x=1, y=0)

    print(distance_meter)
    print(df_point_routing)
    df_point_routing.plot(x="longitude", y="latitude", kind="scatter")
    format_point = "%s,%s/"
    glink = "https://www.google.com/maps/dir/"
    glink += format_point % (contoh_awal[0], contoh_awal[1])
    for index, df in df_point_routing.iterrows():
        output = format_point % (df["latitude"], df["longitude"])
        glink += output
    glink += format_point % (contoh_akhir[0], contoh_akhir[1])
    print(glink)
    plt.show()

#-------------------------------------------------------------------------#
# class Djikstra(object):
#     # contoh_awal = np.array([])
#     # contoh_akhir = np.array([])
#     # awal = np.array([])
#     # akhir = np.array([])
#     url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ_FeidgeJH61pcpyFZ68e4ob7CgjxmrF5Z9DG_ruJDizPD7sLdAkfJfe-UeTwV-KX2ARXIRdiHudB4/pub?output=xlsx"
#     wks = pd.ExcelFile(url,"openpyxl")
#     nodes = pd.read_excel(wks, "Nodes")
#     routes = pd.read_excel(wks, "Routes")


#     def min_distance(point1, points):
#         distances = [np.linalg.norm(point1 - p) for p in points]
#         minimum = min(distances)
#         return distances.index(minimum)
    
#     # latAwal = request.form["edtLatawal"]
#     # longAwal = request.form["edtLongawal"]
#     # latAkir = request.form["edtLatakhir"]
#     # longAkhir = request.form["edtLongakhir"]
#     # print(latAwal)
#     # print(longAwal)
#     # print(latAkir)
#     # print(longAkhir)
#     # awal = np.append(float(latAwal), float(longAkhir))
#     # akhir = np.append(float(latAkir), float(longAkhir))
#     # print(awal, akhir)

#     # print("masukan titik Awal")
#     # latAwal = input('latitude : ')
#     # longAwal = input('longitude : ')
#     # print("masukan titik Akhirl")
#     # latAkhir = input('latitude : ')
#     # longAkhir = input('longitude : ')
#     points = nodes.iloc[:, 2:4].values
#     contoh_awal = awal#np.append(float(latAwal), float(longAwal))
#     contoh_akhir = akhir#np.append(float(latAkhir), float(longAkhir))
#     print(contoh_awal, contoh_akhir)

#     index_awal = min_distance(contoh_awal, points)
#     titik_awal = nodes.iloc[index_awal, :]
#     print(titik_awal)

#     index_akhir = min_distance(contoh_akhir, points)
#     titik_akhir = nodes.iloc[index_akhir, :]
#     print(titik_akhir)


#     class Graph():
#         def __init__(self):
#             """
#             self.edges is a dict of all possible next nodes
#             e.g. {'X': ['A', 'B', 'C', 'E'], ...}
#             self.weights has all the weights between two nodes,
#             with the two nodes as a tuple as the key
#             e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
#             """
#             self.edges = defaultdict(list)
#             self.weights = {}

#         def add_edge(self, from_node, to_node, weight):
#             # Note: assumes edges are bi-directional
#             self.edges[from_node].append(to_node)
#             self.edges[to_node].append(from_node)
#             self.weights[(from_node, to_node)] = weight
#             self.weights[(to_node, from_node)] = weight


#     def dijsktra(graph, initial, end):
#         # shortest paths is a dict of nodes
#         # whose value is a tuple of (previous node, weight)
#         shortest_paths = {initial: (None, 0)}
#         current_node = initial
#         visited = set()

#         while current_node != end:
#             visited.add(current_node)
#             destinations = graph.edges[current_node]
#             weight_to_current_node = shortest_paths[current_node][1]

#             for next_node in destinations:
#                 weight = graph.weights[(current_node, next_node)] + weight_to_current_node
#                 if next_node not in shortest_paths:
#                     shortest_paths[next_node] = (current_node, weight)
#                 else:
#                     current_shortest_weight = shortest_paths[next_node][1]
#                     if current_shortest_weight > weight:
#                         shortest_paths[next_node] = (current_node, weight)

#             next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
#             if not next_destinations:
#                 return "Route Not Possible"
#             # next node is the destination with the lowest weight
#             current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

#         # Work back through destinations in shortest path
#         path = []
#         while current_node is not None:
#             path.append(current_node)
#             next_node = shortest_paths[current_node][0]
#             current_node = next_node
#         # Reverse path
#         path = path[::-1]
#         return path

#     g = Graph()
#     for index, row in routes.iterrows():
#         g.add_edge(row.node_origin, row.node_dest, row.criminal_weight)
#     routing = dijsktra(g, index_awal, index_akhir)
#     print(routing)

#     point_routing = []
#     coordinates = []
#     distance_meter = 0
#     for i in routing:
#       point_routing.append(nodes.iloc[int(i)])
#     df_point_routing = pd.DataFrame(point_routing)

#     for i in range(len(routing)-1):
#         hehe = routes[(routes["node_origin"] == routing[i]) & (routes["node_dest"] == routing[i+1])]
#         route_coords = json.loads(hehe.route_coord.values[0])
#         coordinates += route_coords
#         distance_meter += hehe.distance_meter.values[0]

#     c = np.arange(len(coordinates))
#     df_coordinates = pd.DataFrame(coordinates)
#     df_coordinates.plot(x=1, y=0)

#     print(distance_meter)
#     print(df_point_routing)
#     df_point_routing.plot(x="longitude", y="latitude", kind="scatter")
#     format_point = "%s,%s/"
#     glink = "https://www.google.com/maps/dir/"
#     glink += format_point % (contoh_awal[0], contoh_awal[1])
#     for index, df in df_point_routing.iterrows():
#         output = format_point % (df["latitude"], df["longitude"])
#         glink += output
#     glink += format_point % (contoh_akhir[0], contoh_akhir[1])
#     print(glink)
#     plt.show()