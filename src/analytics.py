"""Business and road network metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon


# Data containers
@dataclass
class BusinessMetrics:
    """Store metrics for one service area."""

    algorithm: str
    cutoff_value: float
    cutoff_unit: str
    reachable_node_count: int
    reachable_edge_count: int
    service_area_sqkm: float
    road_length_km: float
    road_density_km_per_sqkm: float
    intersection_count: int
    intersection_density_per_sqkm: float
    avg_node_degree: float
    business_count: int
    direct_competitor_count: int
    candidate_customer_proxy_count: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics into a plain dictionary."""
        return self.__dict__.copy()


# CRS and geometry helpers
def project_gdf(gdf: gpd.GeoDataFrame, crs) -> gpd.GeoDataFrame:
    """Project a GeoDataFrame to the target CRS."""
    if gdf.empty:
        return gdf.copy().set_crs("EPSG:4326", allow_override=True).to_crs(crs)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(crs)


def polygon_area_sqkm(polygon: Polygon | MultiPolygon) -> float:
    """Return polygon area in square kilometers."""
    return float(polygon.area) / 1_000_000.0


# Road network metrics
def count_intersections(graph) -> int:
    """Count intersection-like nodes using undirected degree >= 3."""
    undirected = nx.Graph(graph)
    return sum(1 for _, degree in undirected.degree() if degree >= 3)


def average_node_degree(graph) -> float:
    """Return the average undirected node degree."""
    undirected = nx.Graph(graph)
    if undirected.number_of_nodes() == 0:
        return 0.0
    return float(np.mean([degree for _, degree in undirected.degree()]))


def road_length_km(graph) -> float:
    """Return total road length in kilometers."""
    total = 0.0
    for _, _, _, data in graph.edges(keys=True, data=True):
        total += float(data.get("length", 0.0))
    return total / 1000.0


# Service area filters
def filter_points_inside(points: gpd.GeoDataFrame, polygon: Polygon | MultiPolygon) -> gpd.GeoDataFrame:
    """Keep points that are inside or touch a polygon."""
    if points.empty:
        return points.copy()
    return points[points.geometry.within(polygon) | points.geometry.intersects(polygon)].copy()


def filter_service_area_features(
    points: gpd.GeoDataFrame,
    polygon: Polygon | MultiPolygon,
    cutoff_value: float,
    cutoff_unit: str,
) -> gpd.GeoDataFrame:
    """Keep features counted inside a service area."""
    if cutoff_unit == "meters" and "network_distance_m" in points.columns:
        distance = pd.to_numeric(points["network_distance_m"], errors="coerce")
        return points[distance <= float(cutoff_value)].copy()
    return filter_points_inside(points, polygon)


def nearest_nodes_for_points(graph, points: gpd.GeoDataFrame) -> pd.Series:
    """Map each point to its nearest road node."""
    if points.empty:
        return pd.Series(dtype="int64")
    xs = points.geometry.x.to_numpy()
    ys = points.geometry.y.to_numpy()
    nearest = ox.distance.nearest_nodes(graph, X=xs, Y=ys)
    return pd.Series(nearest, index=points.index, name="nearest_node")


def add_network_distance(
    points: gpd.GeoDataFrame,
    nearest_nodes: pd.Series,
    distances: dict[int, float],
    distance_column: str = "network_distance_m",
) -> gpd.GeoDataFrame:
    """Add network distance from the candidate location to each feature."""
    result = points.copy()
    result["nearest_node"] = nearest_nodes
    result[distance_column] = result["nearest_node"].map(distances)
    return result


def competitor_mask(features: gpd.GeoDataFrame, competitor_category: str) -> pd.Series:
    """Build a mask for businesses that match the competitor category."""
    if features.empty:
        return pd.Series([], dtype=bool)
    mask = pd.Series(False, index=features.index)
    for col in ("amenity", "shop", "office", "tourism"):
        if col in features.columns:
            mask = mask | (features[col].astype(str) == competitor_category)
    return mask


# Business metrics and scoring
def compute_business_metrics(
    algorithm: str,
    cutoff_value: float,
    cutoff_unit: str,
    subgraph,
    service_polygon_proj: Polygon | MultiPolygon,
    businesses_proj: gpd.GeoDataFrame,
    competitors_proj: gpd.GeoDataFrame,
    residential_proj: gpd.GeoDataFrame,
    elapsed_seconds: float,
) -> BusinessMetrics:
    """Compute business and road metrics for one algorithm result."""
    area = polygon_area_sqkm(service_polygon_proj)
    road_km = road_length_km(subgraph)
    intersections = count_intersections(subgraph)
    avg_degree = average_node_degree(subgraph)

    if area <= 0:
        road_density = 0.0
        intersection_density = 0.0
    else:
        road_density = road_km / area
        intersection_density = intersections / area

    businesses_inside = filter_service_area_features(
        businesses_proj, service_polygon_proj, cutoff_value, cutoff_unit
    )
    competitors_inside = filter_service_area_features(
        competitors_proj, service_polygon_proj, cutoff_value, cutoff_unit
    )
    residential_inside = filter_service_area_features(
        residential_proj, service_polygon_proj, cutoff_value, cutoff_unit
    )

    return BusinessMetrics(
        algorithm=algorithm,
        cutoff_value=cutoff_value,
        cutoff_unit=cutoff_unit,
        reachable_node_count=subgraph.number_of_nodes(),
        reachable_edge_count=subgraph.number_of_edges(),
        service_area_sqkm=area,
        road_length_km=road_km,
        road_density_km_per_sqkm=road_density,
        intersection_count=intersections,
        intersection_density_per_sqkm=intersection_density,
        avg_node_degree=avg_degree,
        business_count=len(businesses_inside),
        direct_competitor_count=len(competitors_inside),
        candidate_customer_proxy_count=len(residential_inside),
        elapsed_seconds=elapsed_seconds,
    )


def normalize(value: float, reference: float, inverse: bool = False) -> float:
    """Normalize a metric into a 0-100 score."""
    if reference <= 0:
        return 0.0
    score = min(value / reference * 100.0, 100.0)
    return 100.0 - score if inverse else score


def business_potential_score(metrics: BusinessMetrics) -> dict[str, float]:
    """Compute a simple score from customer, access, activity, and competition signals."""
    customer_score = normalize(metrics.candidate_customer_proxy_count, reference=1000)
    accessibility_score = 0.5 * normalize(metrics.road_density_km_per_sqkm, reference=20) + 0.5 * normalize(
        metrics.intersection_density_per_sqkm, reference=150
    )
    activity_score = normalize(metrics.business_count, reference=250)
    competition_score = normalize(metrics.direct_competitor_count, reference=50, inverse=True)

    final = (
        0.35 * customer_score
        + 0.25 * accessibility_score
        + 0.20 * activity_score
        + 0.20 * competition_score
    )
    return {
        "customer_score": round(customer_score, 2),
        "accessibility_score": round(accessibility_score, 2),
        "activity_score": round(activity_score, 2),
        "competition_score": round(competition_score, 2),
        "business_potential_score": round(final, 2),
    }


def compare_summaries(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the final summary table."""
    df = pd.DataFrame(rows)
    preferred_cols = [
        "algorithm",
        "cutoff_value",
        "cutoff_unit",
        "business_potential_score",
        "business_count",
        "direct_competitor_count",
        "candidate_customer_proxy_count",
        "service_area_sqkm",
        "reachable_node_count",
        "reachable_edge_count",
        "road_length_km",
        "road_density_km_per_sqkm",
        "intersection_count",
        "intersection_density_per_sqkm",
        "elapsed_seconds",
    ]
    cols = [c for c in preferred_cols if c in df.columns] + [c for c in df.columns if c not in preferred_cols]
    return df[cols]
