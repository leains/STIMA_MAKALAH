"""Helpers for loading and cleaning OpenStreetMap data."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon


# OSMnx setup and geocoding
def setup_osmnx(cache: bool = True, log_console: bool = True) -> None:
    """Set OSMnx cache, console logging, and request timeout."""
    ox.settings.use_cache = cache
    ox.settings.log_console = log_console
    ox.settings.timeout = 180


def geocode_place(place: str) -> gpd.GeoDataFrame:
    """Convert a place name into an OSM boundary polygon."""
    gdf = ox.geocode_to_gdf(place)
    if gdf.empty:
        raise ValueError(f"Place not found or has no boundary polygon: {place}")
    return gdf


def get_candidate_point(place_gdf: gpd.GeoDataFrame, lat: float | None, lon: float | None) -> Point:
    """Return the candidate business point from coordinates or the place polygon."""
    if (lat is None) != (lon is None):
        raise ValueError("candidate_lat and candidate_lon must be provided together.")
    if lat is not None and lon is not None:
        return Point(lon, lat)
    place_geom = place_gdf.geometry.iloc[0]
    return place_geom.representative_point()


# Road graph loading
def download_road_graph(place: str, network_type: str = "walk"):
    """Download the OSM road graph for a place and network type."""
    graph = ox.graph_from_place(place, network_type=network_type, simplify=True)
    if graph.number_of_nodes() == 0:
        raise RuntimeError("Downloaded graph has no nodes. Try a broader place query.")
    return graph


def project_graph(graph):
    """Project a WGS84 graph to a metric CRS."""
    return ox.project_graph(graph)


# Feature loading
def _features_from_polygon_compat(polygon: Polygon | MultiPolygon, tags: dict[str, Any]) -> gpd.GeoDataFrame:
    """Call the OSMnx feature API available in the installed version."""
    if hasattr(ox, "features_from_polygon"):
        return ox.features_from_polygon(polygon, tags)
    if hasattr(ox, "geometries_from_polygon"):
        return ox.geometries_from_polygon(polygon, tags)
    if hasattr(ox, "features") and hasattr(ox.features, "features_from_polygon"):
        return ox.features.features_from_polygon(polygon, tags)
    raise AttributeError("Cannot find an OSMnx feature loading API.")


def fetch_features_within(polygon_wgs84: Polygon | MultiPolygon, tags: dict[str, Any]) -> gpd.GeoDataFrame:
    """Fetch OSM features inside a WGS84 polygon."""
    try:
        gdf = _features_from_polygon_compat(polygon_wgs84, tags)
    except Exception as exc:
        print(f"Warning: feature query failed: {exc}")
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    if gdf.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    gdf = gdf.reset_index()
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


# Feature cleanup
def feature_representative_points(features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert point and polygon features into representative points."""
    if features.empty:
        return features.copy()

    result = features.copy()
    result["original_geometry"] = result.geometry
    result["geometry"] = result.geometry.apply(
        lambda geom: geom if geom.geom_type == "Point" else geom.representative_point()
    )
    return result.set_geometry("geometry")


def tag_value(row: pd.Series) -> str:
    """Return a short category label from common OSM tag columns."""
    for key in ("amenity", "shop", "office", "tourism", "building"):
        value = row.get(key)
        if pd.notna(value):
            return f"{key}={value}"
    return "unknown"
