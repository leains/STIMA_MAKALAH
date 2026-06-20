"""Map and GeoJSON helpers."""

from __future__ import annotations

from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon


# Output folders
def ensure_dir(path: str | Path) -> Path:
    """Create an output folder if needed."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


# Geometry export
def to_wgs84_gdf(geometry, crs) -> gpd.GeoDataFrame:
    """Convert a geometry to a WGS84 GeoDataFrame."""
    return gpd.GeoDataFrame(geometry=[geometry], crs=crs).to_crs("EPSG:4326")


def save_geojson(geometry, crs, path: str | Path) -> None:
    """Save one geometry as a WGS84 GeoJSON file."""
    gdf = to_wgs84_gdf(geometry, crs)
    gdf.to_file(path, driver="GeoJSON")


# Interactive map
def render_map(
    output_path: str | Path,
    candidate_point_wgs84: Point,
    dijkstra_polygon_wgs84: Polygon | MultiPolygon,
    bfs_polygon_wgs84: Polygon | MultiPolygon,
    businesses_wgs84: gpd.GeoDataFrame,
    competitors_wgs84: gpd.GeoDataFrame,
    residential_wgs84: gpd.GeoDataFrame,
    summary: pd.DataFrame,
) -> None:
    """Create an interactive map for the candidate location and service areas."""
    m = folium.Map(
        location=[candidate_point_wgs84.y, candidate_point_wgs84.x],
        zoom_start=14,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    folium.Marker(
        [candidate_point_wgs84.y, candidate_point_wgs84.x],
        popup="Candidate business location",
        tooltip="Candidate location",
        icon=folium.Icon(color="red", icon="star"),
    ).add_to(m)

    folium.GeoJson(
        dijkstra_polygon_wgs84.__geo_interface__,
        name="Dijkstra service area",
        style_function=lambda _: {"fillColor": "#2b8cbe", "color": "#045a8d", "weight": 2, "fillOpacity": 0.25},
        tooltip="Dijkstra weighted service area",
    ).add_to(m)

    folium.GeoJson(
        bfs_polygon_wgs84.__geo_interface__,
        name="BFS comparison area",
        style_function=lambda _: {"fillColor": "#fdae6b", "color": "#e6550d", "weight": 2, "fillOpacity": 0.18},
        tooltip="BFS unweighted comparison area",
    ).add_to(m)

    if not competitors_wgs84.empty:
        competitor_layer = folium.FeatureGroup(name="Direct competitors")
        for _, row in competitors_wgs84.iterrows():
            geom = row.geometry
            name = row.get("name", "competitor")
            distance = row.get("network_distance_m")
            popup = f"{name}<br>{distance:.0f} m" if pd.notna(distance) else str(name)
            folium.CircleMarker(
                [geom.y, geom.x],
                radius=4,
                color="#b10026",
                fill=True,
                fill_opacity=0.8,
                popup=popup,
            ).add_to(competitor_layer)
        competitor_layer.add_to(m)

    if not businesses_wgs84.empty:
        business_layer = folium.FeatureGroup(name="Businesses within travel area", show=False)
        for _, row in businesses_wgs84.head(500).iterrows():
            geom = row.geometry
            name = row.get("name", "business")
            category = row.get("category", "unknown")
            distance = row.get("network_distance_m")
            popup = f"{name}<br>{category}<br>{distance:.0f} m" if pd.notna(distance) else f"{name}<br>{category}"
            folium.CircleMarker(
                [geom.y, geom.x],
                radius=2.5,
                color="#238b45",
                fill=True,
                fill_opacity=0.6,
                popup=popup,
            ).add_to(business_layer)
        business_layer.add_to(m)

    if not residential_wgs84.empty:
        residential_layer = folium.FeatureGroup(name="Customer proxies", show=False)
        for _, row in residential_wgs84.head(500).iterrows():
            geom = row.geometry
            distance = row.get("network_distance_m")
            popup = f"{distance:.0f} m from candidate" if pd.notna(distance) else None
            folium.CircleMarker(
                [geom.y, geom.x],
                radius=2,
                color="#756bb1",
                fill=True,
                fill_opacity=0.5,
                popup=popup,
            ).add_to(residential_layer)
        residential_layer.add_to(m)

    html = summary.to_html(index=False, float_format=lambda x: f"{x:.3f}")
    folium.Marker(
        [candidate_point_wgs84.y, candidate_point_wgs84.x],
        icon=folium.DivIcon(
            html=f"""
            <div style='font-size: 10pt; width: 420px; background: white; border: 1px solid #444; padding: 6px;'>
            <b>Algorithm Summary</b><br>{html}
            </div>
            """
        ),
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(output_path)
