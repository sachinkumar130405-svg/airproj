import osmnx as ox
import networkx as nx
import numpy as np
import networkx as nx

def compute_safe_route(start, end, grid):
    """
    Dummy safe-route function (MVP placeholder).
    In the real version, it would use pollution-weighted paths.
    """
    # Placeholder graph
    G = nx.Graph()
    route_safe = [start, end]
    route_fast = [start, end]
    return G, route_safe, route_fast


def compute_safe_route(start_coords, end_coords, grid_df, exposure_factor=1.5):
    """
    Compute the least-exposure route between two points.
    start_coords, end_coords: (lat, lon)
    grid_df: interpolated PM2.5 grid
    """
    # Build road network graph
    G = ox.graph_from_point(start_coords, dist=5000, network_type='drive')

    # Assign exposure weight to each edge
    for u, v, data in G.edges(data=True):
        lat = (G.nodes[u]['y'] + G.nodes[v]['y']) / 2
        lon = (G.nodes[u]['x'] + G.nodes[v]['x']) / 2
        nearest = grid_df.iloc[((grid_df['lat'] - lat) ** 2 + (grid_df['lon'] - lon) ** 2).idxmin()]
        pm = nearest['pm25_est']
        data['weight'] = data.get('length', 1) * (1 + exposure_factor * pm / 200.0)

    # Compute shortest (safe) and fastest routes
    orig = ox.distance.nearest_nodes(G, start_coords[1], start_coords[0])
    dest = ox.distance.nearest_nodes(G, end_coords[1], end_coords[0])
    route_safe = nx.shortest_path(G, orig, dest, weight='weight')
    route_fast = nx.shortest_path(G, orig, dest, weight='length')

    return G, route_safe, route_fast
