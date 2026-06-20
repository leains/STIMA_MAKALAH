"""Command-line pipeline for the analysis."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from .analytics import (
    add_network_distance,
    business_potential_score,
    compare_summaries,
    competitor_mask,
    compute_business_metrics,
    nearest_nodes_for_points,
    project_gdf,
)
from .algorithms import (
    ReachabilityResult,
    bfs_reachability,
    dijkstra_reachability,
    estimate_bfs_depth_from_cutoff,
    service_area_polygon_from_subgraph,
    subgraph_from_reachability,
)
from .config import BUSINESS_TAGS, COMPETITOR_TAG_LOOKUP, RESIDENTIAL_BUILDING_TAGS, AnalysisConfig
from .mapping import ensure_dir, render_map, save_geojson, to_wgs84_gdf
from .osm_fetch import (
    download_road_graph,
    feature_representative_points,
    fetch_features_within,
    geocode_place,
    get_candidate_point,
    project_graph,
    setup_osmnx,
    tag_value,
)


# Data containers
@dataclass(frozen=True)
class ServiceArea:
    """Store the reachable road network and service-area polygon."""

    result: ReachabilityResult
    subgraph: Any
    polygon_proj: Polygon | MultiPolygon
    polygon_wgs84: Polygon | MultiPolygon


@dataclass(frozen=True)
class MarketFeatures:
    """Store business and residential features in WGS84 and metric CRS."""

    businesses_wgs84: gpd.GeoDataFrame
    businesses_proj: gpd.GeoDataFrame
    residential_wgs84: gpd.GeoDataFrame
    residential_proj: gpd.GeoDataFrame


# CLI parsing
def parse_args() -> AnalysisConfig:
    """Parse command-line arguments into an AnalysisConfig object."""
    parser = argparse.ArgumentParser(
        description=(
            "Count businesses, direct competitors, and customer proxies within "
            "a road-network travel distance from a candidate location."
        )
    )
    parser.add_argument("--place", default="Coblong, Bandung, Indonesia")
    parser.add_argument("--network-type", default="walk", choices=["walk", "drive", "drive_service", "bike", "all"])
    parser.add_argument("--cutoff-m", type=float, default=5000.0)
    parser.add_argument("--competitor-category", default="cafe", choices=sorted(COMPETITOR_TAG_LOOKUP))
    parser.add_argument("--candidate-lat", type=float, default=None)
    parser.add_argument("--candidate-lon", type=float, default=None)
    parser.add_argument("--edge-buffer-m", type=float, default=35.0)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    return AnalysisConfig(
        place=args.place,
        network_type=args.network_type,
        cutoff_m=args.cutoff_m,
        competitor_category=args.competitor_category,
        candidate_lat=args.candidate_lat,
        candidate_lon=args.candidate_lon,
        edge_buffer_m=args.edge_buffer_m,
        output_dir=args.output_dir,
    )


# Service area building
def build_service_area(
    graph_proj,
    graph_crs,
    result: ReachabilityResult,
    edge_buffer_m: float,
) -> ServiceArea:
    """Build the reachable subgraph and map polygon for one result."""
    subgraph = subgraph_from_reachability(graph_proj, result)
    polygon_proj = service_area_polygon_from_subgraph(subgraph, edge_buffer_m)
    polygon_wgs84 = to_wgs84_gdf(polygon_proj, graph_crs).geometry.iloc[0]
    return ServiceArea(
        result=result,
        subgraph=subgraph,
        polygon_proj=polygon_proj,
        polygon_wgs84=polygon_wgs84,
    )


def build_dijkstra_area(graph_proj, graph_crs, origin_node: int, cfg: AnalysisConfig) -> ServiceArea:
    """Build the main service area with Dijkstra."""
    result = dijkstra_reachability(graph_proj, origin_node, cfg.cutoff_m, weight="length")
    return build_service_area(graph_proj, graph_crs, result, cfg.edge_buffer_m)


def build_bfs_area(graph_proj, graph_crs, origin_node: int, cfg: AnalysisConfig) -> tuple[ServiceArea, int]:
    """Build the BFS comparison area from an estimated hop limit."""
    depth_limit = estimate_bfs_depth_from_cutoff(graph_proj, cfg.cutoff_m)
    result = bfs_reachability(graph_proj, origin_node, depth_limit=depth_limit)
    return build_service_area(graph_proj, graph_crs, result, cfg.edge_buffer_m), depth_limit


# Market feature loading
def load_market_features(fetch_polygon_wgs84: Polygon | MultiPolygon, graph_crs) -> MarketFeatures:
    """Load business and residential features from OSM."""
    businesses_wgs84 = feature_representative_points(fetch_features_within(fetch_polygon_wgs84, BUSINESS_TAGS))
    residential_wgs84 = feature_representative_points(fetch_features_within(fetch_polygon_wgs84, RESIDENTIAL_BUILDING_TAGS))

    if not businesses_wgs84.empty:
        businesses_wgs84["category"] = businesses_wgs84.apply(tag_value, axis=1)
    if not residential_wgs84.empty:
        residential_wgs84["category"] = residential_wgs84.apply(tag_value, axis=1)

    return MarketFeatures(
        businesses_wgs84=businesses_wgs84,
        businesses_proj=project_gdf(businesses_wgs84, graph_crs),
        residential_wgs84=residential_wgs84,
        residential_proj=project_gdf(residential_wgs84, graph_crs),
    )


# Distance enrichment and filtering
def attach_dijkstra_distances(
    graph_proj,
    features_wgs84: gpd.GeoDataFrame,
    features_proj: gpd.GeoDataFrame,
    distances: dict[int, float],
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Add nearest road node and Dijkstra travel distance to features."""
    if features_proj.empty:
        return features_wgs84.copy(), features_proj.copy()

    nearest_nodes = nearest_nodes_for_points(graph_proj, features_proj)
    features_proj = add_network_distance(features_proj, nearest_nodes, distances)

    features_wgs84 = features_wgs84.copy()
    features_wgs84["nearest_node"] = features_proj["nearest_node"]
    features_wgs84["network_distance_m"] = features_proj["network_distance_m"]
    return features_wgs84, features_proj


def split_direct_competitors(
    businesses_wgs84: gpd.GeoDataFrame,
    businesses_proj: gpd.GeoDataFrame,
    competitor_category: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Split direct competitors from all business features."""
    if businesses_proj.empty:
        return businesses_wgs84.iloc[0:0].copy(), businesses_proj.copy()

    mask = competitor_mask(businesses_proj, competitor_category)
    competitors_proj = businesses_proj[mask].copy()
    competitors_wgs84 = businesses_wgs84.loc[competitors_proj.index].copy()
    return competitors_wgs84, competitors_proj


def filter_by_travel_distance(features: gpd.GeoDataFrame, cutoff_m: float) -> gpd.GeoDataFrame:
    """Keep features within the Dijkstra travel-distance limit."""
    if features.empty or "network_distance_m" not in features.columns:
        return features.iloc[0:0].copy()
    distance = pd.to_numeric(features["network_distance_m"], errors="coerce")
    return features[distance <= cutoff_m].copy()


# Output formatting
def feature_table_for_csv(features: gpd.GeoDataFrame) -> pd.DataFrame:
    """Convert point features into a simple CSV table."""
    if features.empty:
        return pd.DataFrame()

    table = pd.DataFrame(features.drop(columns=["geometry", "original_geometry"], errors="ignore"))
    table = table.drop(columns=["latitude", "longitude"], errors="ignore")
    table.insert(0, "longitude", features.geometry.x.to_numpy())
    table.insert(0, "latitude", features.geometry.y.to_numpy())
    return table


def metric_row(area: ServiceArea, businesses_proj, competitors_proj, residential_proj) -> dict[str, Any]:
    """Build one summary row with metrics and score."""
    metrics = compute_business_metrics(
        algorithm=area.result.algorithm,
        cutoff_value=area.result.cutoff_value,
        cutoff_unit=area.result.cutoff_unit,
        subgraph=area.subgraph,
        service_polygon_proj=area.polygon_proj,
        businesses_proj=businesses_proj,
        competitors_proj=competitors_proj,
        residential_proj=residential_proj,
        elapsed_seconds=area.result.elapsed_seconds,
    )
    return metrics.to_dict() | business_potential_score(metrics)


def write_outputs(
    out_dir: Path,
    cfg: AnalysisConfig,
    graph_proj,
    origin_node: int,
    bfs_depth_limit: int,
    candidate_point_wgs84: Point,
    dijkstra_area: ServiceArea,
    bfs_area: ServiceArea,
    summary: pd.DataFrame,
    businesses_in_travel_area: gpd.GeoDataFrame,
    competitors_in_travel_area: gpd.GeoDataFrame,
    residential_in_travel_area: gpd.GeoDataFrame,
) -> None:
    """Write all analysis outputs to disk."""
    output_files = [
        "summary.csv",
        "businesses_within_travel_area.csv",
        "direct_competitors_within_travel_area.csv",
        "candidate_customers_within_travel_area.csv",
        "service_area_dijkstra.geojson",
        "service_area_bfs.geojson",
        "road_network_projected.graphml",
        "map.html",
        "run_metadata.json",
    ]

    summary.to_csv(out_dir / "summary.csv", index=False)
    feature_table_for_csv(businesses_in_travel_area).to_csv(
        out_dir / "businesses_within_travel_area.csv", index=False
    )
    feature_table_for_csv(competitors_in_travel_area).to_csv(
        out_dir / "direct_competitors_within_travel_area.csv", index=False
    )
    feature_table_for_csv(residential_in_travel_area).to_csv(
        out_dir / "candidate_customers_within_travel_area.csv", index=False
    )

    save_geojson(dijkstra_area.polygon_proj, graph_proj.graph["crs"], out_dir / "service_area_dijkstra.geojson")
    save_geojson(bfs_area.polygon_proj, graph_proj.graph["crs"], out_dir / "service_area_bfs.geojson")
    ox.save_graphml(graph_proj, out_dir / "road_network_projected.graphml")

    render_map(
        output_path=out_dir / "map.html",
        candidate_point_wgs84=candidate_point_wgs84,
        dijkstra_polygon_wgs84=dijkstra_area.polygon_wgs84,
        bfs_polygon_wgs84=bfs_area.polygon_wgs84,
        businesses_wgs84=businesses_in_travel_area,
        competitors_wgs84=competitors_in_travel_area,
        residential_wgs84=residential_in_travel_area,
        summary=summary,
    )

    metadata: dict[str, Any] = {
        "place": cfg.place,
        "network_type": cfg.network_type,
        "travel_cutoff_m": cfg.cutoff_m,
        "travel_cutoff_km": cfg.cutoff_m / 1000.0,
        "competitor_category": cfg.competitor_category,
        "candidate_lat": candidate_point_wgs84.y,
        "candidate_lon": candidate_point_wgs84.x,
        "origin_node": int(origin_node),
        "bfs_depth_limit": int(bfs_depth_limit),
        "graph_nodes": graph_proj.number_of_nodes(),
        "graph_edges": graph_proj.number_of_edges(),
        "businesses_within_travel_area": len(businesses_in_travel_area),
        "direct_competitors_within_travel_area": len(competitors_in_travel_area),
        "candidate_customer_proxies_within_travel_area": len(residential_in_travel_area),
        "feature_fetch_area": "union of Dijkstra and BFS service areas",
        "outputs": output_files,
    }
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


# Main pipeline
def main() -> None:
    """Run the full analysis pipeline."""
    cfg = parse_args()
    setup_osmnx(cache=True, log_console=True)
    out_dir = ensure_dir(cfg.output_dir)

    print(f"[1/9] Geocoding place: {cfg.place}")
    place_gdf = geocode_place(cfg.place)
    candidate_point_wgs84 = get_candidate_point(place_gdf, cfg.candidate_lat, cfg.candidate_lon)

    print(f"[2/9] Downloading road graph: network_type={cfg.network_type}")
    graph_wgs84 = download_road_graph(cfg.place, network_type=cfg.network_type)

    print("[3/9] Projecting graph to metric CRS")
    graph_proj = project_graph(graph_wgs84)
    graph_crs = graph_proj.graph["crs"]
    candidate_proj = gpd.GeoDataFrame(geometry=[candidate_point_wgs84], crs="EPSG:4326").to_crs(graph_crs)
    candidate_point_proj = candidate_proj.geometry.iloc[0]
    origin_node = ox.distance.nearest_nodes(graph_proj, X=candidate_point_proj.x, Y=candidate_point_proj.y)

    print(f"[4/9] Building Dijkstra service area for {cfg.cutoff_m / 1000:.2f} km travel distance")
    dijkstra_area = build_dijkstra_area(graph_proj, graph_crs, origin_node, cfg)

    print("[5/9] Building BFS comparison area")
    bfs_area, bfs_depth_limit = build_bfs_area(graph_proj, graph_crs, origin_node, cfg)
    print(f"BFS depth from median road length: {bfs_depth_limit} hops")

    print("[6/9] Fetching businesses and residential buildings from OpenStreetMap")
    fetch_polygon_wgs84 = unary_union([dijkstra_area.polygon_wgs84, bfs_area.polygon_wgs84])
    market = load_market_features(fetch_polygon_wgs84, graph_crs)

    print("[7/9] Mapping market features to the road graph")
    businesses_wgs84, businesses_proj = attach_dijkstra_distances(
        graph_proj, market.businesses_wgs84, market.businesses_proj, dijkstra_area.result.distances
    )
    residential_wgs84, residential_proj = attach_dijkstra_distances(
        graph_proj, market.residential_wgs84, market.residential_proj, dijkstra_area.result.distances
    )
    competitors_wgs84, competitors_proj = split_direct_competitors(
        businesses_wgs84, businesses_proj, cfg.competitor_category
    )

    businesses_in_travel_area = filter_by_travel_distance(businesses_wgs84, cfg.cutoff_m)
    competitors_in_travel_area = filter_by_travel_distance(competitors_wgs84, cfg.cutoff_m)
    residential_in_travel_area = filter_by_travel_distance(residential_wgs84, cfg.cutoff_m)

    print("[8/9] Computing business metrics")
    summary = compare_summaries(
        [
            metric_row(dijkstra_area, businesses_proj, competitors_proj, residential_proj),
            metric_row(bfs_area, businesses_proj, competitors_proj, residential_proj),
        ]
    )

    print("[9/9] Writing CSV, GeoJSON, GraphML, metadata, and map")
    write_outputs(
        out_dir=out_dir,
        cfg=cfg,
        graph_proj=graph_proj,
        origin_node=origin_node,
        bfs_depth_limit=bfs_depth_limit,
        candidate_point_wgs84=candidate_point_wgs84,
        dijkstra_area=dijkstra_area,
        bfs_area=bfs_area,
        summary=summary,
        businesses_in_travel_area=businesses_in_travel_area,
        competitors_in_travel_area=competitors_in_travel_area,
        residential_in_travel_area=residential_in_travel_area,
    )

    print("\nDone. Summary:")
    print(summary.to_string(index=False))
    print(f"\nOutput folder: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
