"""Road network algorithms used by the analysis."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


# Data containers
@dataclass
class ReachabilityResult:
    """Store reachable nodes, distances, and runtime info."""

    algorithm: str
    origin_node: int
    reachable_nodes: set[int]
    distances: dict[int, float]
    elapsed_seconds: float
    cutoff_value: float
    cutoff_unit: str


# Reachability algorithms
def dijkstra_reachability(graph, origin_node: int, cutoff_m: float, weight: str = "length") -> ReachabilityResult:
    """Find nodes reachable within a weighted travel-distance limit."""
    start = perf_counter()
    distances = nx.single_source_dijkstra_path_length(
        graph,
        source=origin_node,
        cutoff=cutoff_m,
        weight=weight,
    )
    elapsed = perf_counter() - start
    return ReachabilityResult(
        algorithm="Dijkstra",
        origin_node=origin_node,
        reachable_nodes=set(distances.keys()),
        distances={int(k): float(v) for k, v in distances.items()},
        elapsed_seconds=elapsed,
        cutoff_value=cutoff_m,
        cutoff_unit="meters",
    )


def estimate_bfs_depth_from_cutoff(graph, cutoff_m: float) -> int:
    """Estimate a BFS hop limit from the travel-distance limit."""
    lengths = []
    for _, _, _, data in graph.edges(keys=True, data=True):
        length = data.get("length")
        if length and length > 0:
            lengths.append(float(length))
    if not lengths:
        return 1
    median_edge_length = float(np.median(lengths))
    return max(1, int(round(cutoff_m / median_edge_length)))


def bfs_reachability(graph, origin_node: int, depth_limit: int) -> ReachabilityResult:
    """Find nodes reachable within a fixed number of unweighted road hops."""
    start = perf_counter()
    visited: set[int] = {origin_node}
    depth: dict[int, int] = {origin_node: 0}
    queue: deque[int] = deque([origin_node])

    while queue:
        node = queue.popleft()
        if depth[node] >= depth_limit:
            continue
        for neighbor in graph.successors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                depth[neighbor] = depth[node] + 1
                queue.append(neighbor)

    elapsed = perf_counter() - start
    return ReachabilityResult(
        algorithm="BFS",
        origin_node=origin_node,
        reachable_nodes=visited,
        distances={int(k): float(v) for k, v in depth.items()},
        elapsed_seconds=elapsed,
        cutoff_value=float(depth_limit),
        cutoff_unit="hops",
    )


# Graph and geometry helpers
def subgraph_from_reachability(graph, result: ReachabilityResult):
    """Create a graph copy from the reachable nodes."""
    return graph.subgraph(result.reachable_nodes).copy()


def service_area_polygon_from_subgraph(subgraph, edge_buffer_m: float = 35.0) -> Polygon | MultiPolygon:
    """Create a service-area polygon from reachable road edges."""
    if subgraph.number_of_edges() == 0:
        nodes = ox.graph_to_gdfs(subgraph, edges=False)
        if nodes.empty:
            raise ValueError("Cannot build a polygon from an empty subgraph.")
        return nodes.geometry.unary_union.buffer(edge_buffer_m)

    edges = ox.graph_to_gdfs(subgraph, nodes=False, fill_edge_geometry=True)
    buffered = edges.geometry.buffer(edge_buffer_m)
    return unary_union(buffered)


def graph_nodes_gdf(graph) -> gpd.GeoDataFrame:
    """Convert graph nodes into a GeoDataFrame."""
    return ox.graph_to_gdfs(graph, edges=False)


def graph_edges_gdf(graph) -> gpd.GeoDataFrame:
    """Convert graph edges into a GeoDataFrame."""
    return ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True)
