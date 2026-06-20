# Geospatial Business Analytics with Dijkstra and BFS

This project analyzes a candidate business location using OpenStreetMap road data. It answers a simple question: **how many businesses, direct competitors, and customer proxies can be reached within X km of travel distance?**

The distance is not a straight-line radius. It follows the road network. Dijkstra is used for the main travel-distance calculation, while BFS is included as a simple comparison that counts road hops instead of real road length.

## What It Answers

- How many businesses are reachable within `x` km from the candidate location?
- How many direct competitors are reachable for a category like `cafe` or `pharmacy`?
- How many residential buildings can be used as a simple customer proxy?
- How dense is the road network inside the reachable area?
- How different are the Dijkstra and BFS results?

## Project Structure

| File | Purpose |
|---|---|
| `run_analysis.py` | Entry point from the project root |
| `src/main_analysis.py` | Main analysis pipeline |
| `src/config.py` | Settings, OSM tags, and competitor categories |
| `src/osm_fetch.py` | OSM geocoding, road graph loading, and feature loading |
| `src/algorithms.py` | Dijkstra service area and BFS comparison |
| `src/analytics.py` | Business metrics, road metrics, and scoring |
| `src/mapping.py` | GeoJSON export and Folium map rendering |
| `requirements.txt` | Python dependencies |

## Installation

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Examples

Cafe analysis in Coblong with a 5 km walking distance:

```bash
python run_analysis.py --place "Coblong, Bandung, Indonesia" --network-type walk --cutoff-m 5000 --competitor-category cafe --output-dir outputs/coblong_cafe
```

Pharmacy analysis in Bandung with a 10 km driving distance:

```bash
python run_analysis.py --place "Bandung, West Java, Indonesia" --network-type drive --cutoff-m 10000 --competitor-category pharmacy --output-dir outputs/bandung_pharmacy
```

If `--candidate-lat` and `--candidate-lon` are not provided, the program uses a representative point inside the selected place.

## Outputs

The output folder contains:

| File | Content |
|---|---|
| `summary.csv` | Dijkstra vs BFS comparison with business counts, competitor counts, customer proxies, area, road density, and score |
| `businesses_within_travel_area.csv` | Businesses within the Dijkstra travel-distance limit |
| `direct_competitors_within_travel_area.csv` | Direct competitors within the Dijkstra travel-distance limit |
| `candidate_customers_within_travel_area.csv` | Residential buildings used as customer proxies |
| `service_area_dijkstra.geojson` | Dijkstra service-area polygon |
| `service_area_bfs.geojson` | BFS comparison-area polygon |
| `road_network_projected.graphml` | Projected OSM road graph |
| `map.html` | Interactive map with the candidate point, service areas, businesses, competitors, and customer proxies |
| `run_metadata.json` | Run settings and basic graph metadata |
