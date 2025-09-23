#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dijkstra con datos reales (OSMnx) — Montevideo (solo AUTO) [v2]
- FIX: graph_from_bbox requiere argumentos con nombre en OSMnx 2.x
- Fallback adicional: si falla el bbox, intenta con polígono (buffer) y graph_from_polygon
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import folium
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, MultiLineString, Point
from shapely import union_all


DEFAULT_ORIGIN_ADDR = "Rambla República de Chile 4275, Montevideo, Uruguay"
DEFAULT_ORIGIN_FALLBACK = (-34.8976182, -56.1189754)
DEFAULT_DEST_LATLON = (-34.8894, -56.1601)
DEFAULT_DEST_ADDR = "Av. 8 de Octubre & Av. Garibaldi, Montevideo, Uruguay"

DEFAULT_BUFFER_KM = 3.0
DEFAULT_MAX_TRIES = 3
DEFAULT_MODE = "drive"


@dataclass
class RouteMetrics:
    time_s: float
    length_m: float
    nodes: int


def human_time(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m} min {s:02d}s"


def human_km(meters: float) -> float:
    return round(meters / 1000.0, 2)


def geocode_or_fallback(address: Optional[str],
                        fallback_latlon: Optional[Tuple[float, float]] = None
                        ) -> Tuple[float, float]:
    if address is None:
        if fallback_latlon is None:
            raise ValueError("No se proporcionó dirección ni fallback para geocodificar.")
        return fallback_latlon
    try:
        lat, lon = ox.geocoder.geocode(address)
        return float(lat), float(lon)
    except Exception as e:
        if fallback_latlon is not None:
            print(f"[WARN] Geocodificación fallida para '{address}'. Usando fallback {fallback_latlon}. Error: {e}")
            return fallback_latlon
        raise


def bbox_around_points(orig: Tuple[float, float], dest: Tuple[float, float], buffer_km: float):
    o_lat, o_lon = orig
    d_lat, d_lon = dest
    lat_mid = (o_lat + d_lat) / 2.0
    delta_lat = buffer_km / 111.0
    cos_lat = max(0.01, math.cos(math.radians(lat_mid)))
    delta_lon = buffer_km / (111.0 * cos_lat)

    north = max(o_lat, d_lat) + delta_lat
    south = min(o_lat, d_lat) - delta_lat
    east = max(o_lon, d_lon) + delta_lon
    west = min(o_lon, d_lon) - delta_lon
    return north, south, east, west


def build_graph_bbox(north: float, south: float, east: float, west: float, mode: str = DEFAULT_MODE):
    # FIX: en OSMnx 2.x pasar args con nombre
    G = ox.graph_from_bbox(north=north, south=south, east=east, west=west, network_type=mode)
    G = ox.routing.add_edge_speeds(G)
    G = ox.routing.add_edge_travel_times(G)
    return G


def build_graph_polygon(orig: Tuple[float, float], dest: Tuple[float, float], buffer_km: float, mode: str = DEFAULT_MODE):
    # Polígono = buffer (círculos) alrededor de origen y destino unido
    o = Point(orig[1], orig[0])  # shapely usa (lon, lat)
    d = Point(dest[1], dest[0])
    # Aproximación burda deg/metro: 1 deg lat ~ 111km, lon ajustado por lat media
    lat_mid = (orig[0] + dest[0]) / 2.0
    cos_lat = max(0.01, math.cos(math.radians(lat_mid)))
    deg_lat = buffer_km / 111.0
    deg_lon = buffer_km / (111.0 * cos_lat)

    o_buf = o.buffer(deg_lon).buffer(0)  # deg_lon para ambos ejes por simplicidad
    d_buf = d.buffer(deg_lon).buffer(0)
    poly = union_all([o_buf, d_buf]).buffer(0)

    G = ox.graph_from_polygon(polygon=poly, network_type=mode)
    G = ox.routing.add_edge_speeds(G)
    G = ox.routing.add_edge_travel_times(G)
    return G


def nearest_nodes(G, orig_latlon: Tuple[float, float], dest_latlon: Tuple[float, float]):
    o_lat, o_lon = orig_latlon
    d_lat, d_lon = dest_latlon
    orig_node = ox.distance.nearest_nodes(G, X=o_lon, Y=o_lat)
    dest_node = ox.distance.nearest_nodes(G, X=d_lon, Y=d_lat)
    return orig_node, dest_node


def compute_routes(G, orig_node, dest_node):
    fastest = ox.routing.shortest_path(G, orig_node, dest_node, weight="travel_time")
    shortest = ox.routing.shortest_path(G, orig_node, dest_node, weight="length")
    if fastest is None or shortest is None:
        raise nx.NetworkXNoPath("No se encontró ruta entre los nodos dados.")
    return fastest, shortest


def route_metrics(G, route) -> RouteMetrics:
    gdf_time = ox.routing.route_to_gdf(G, route, weight="travel_time")
    gdf_len  = ox.routing.route_to_gdf(G, route, weight="length")
    time_s   = float(gdf_time["travel_time"].sum())
    length_m = float(gdf_len["length"].sum())
    return RouteMetrics(time_s=time_s, length_m=length_m, nodes=len(route))


def unify_geometry(G, route):
    gdf = ox.routing.route_to_gdf(G, route, weight="length")
    return gdf.geometry.union_all


def line_to_coords(geom):
    if isinstance(geom, LineString):
        return [(lat, lon) for lon, lat in geom.coords]
    elif isinstance(geom, MultiLineString):
        coords = []
        for part in geom.geoms:
            coords.extend([(lat, lon) for lon, lat in part.coords])
        return coords
    return []


def main():
    parser = argparse.ArgumentParser(description="Dijkstra con OSMnx (Montevideo, solo auto)")
    parser.add_argument("--origen", type=str, default=DEFAULT_ORIGIN_ADDR)
    parser.add_argument("--orig-lat", type=float, default=None)
    parser.add_argument("--orig-lon", type=float, default=None)

    parser.add_argument("--dest", type=str, default=DEFAULT_DEST_ADDR)
    parser.add_argument("--dest-lat", type=float, default=DEFAULT_DEST_LATLON[0])
    parser.add_argument("--dest-lon", type=float, default=DEFAULT_DEST_LATLON[1])

    parser.add_argument("--buffer-km", type=float, default=DEFAULT_BUFFER_KM)
    parser.add_argument("--max-tries", type=int, default=DEFAULT_MAX_TRIES)
    parser.add_argument("--out-html", type=str, default="ruta_auto_montevideo.html")
    parser.add_argument("--out-csv", type=str, default="resumen_ruta_auto.csv")
    args = parser.parse_args()

    ox.settings.use_cache = True
    ox.settings.log_console = True

    # Origen
    if args.orig_lat is not None and args.orig_lon is not None:
        orig = (float(args.orig_lat), float(args.orig_lon))
        origen_str = f"({orig[0]:.6f}, {orig[1]:.6f})"
    else:
        orig = geocode_or_fallback(args.origen, fallback_latlon=DEFAULT_ORIGIN_FALLBACK)
        origen_str = args.origen

    # Destino (lat/lon robusto)
    dest = (float(args.dest_lat), float(args.dest_lon))
    destino_str = f"{args.dest} (lat,lon={dest[0]:.6f},{dest[1]:.6f})"

    buffer_km = max(0.5, float(args.buffer_km))
    tries = max(1, int(args.max_tries))

    last_err = None
    for attempt in range(1, tries + 1):
        print(f"[INFO] Intento {attempt}/{tries} con BUFFER_KM={buffer_km:.2f} (bbox)")
        north, south, east, west = bbox_around_points(orig, dest, buffer_km)
        try:
            G = build_graph_bbox(north=north, south=south, east=east, west=west, mode=DEFAULT_MODE)
            o_node, d_node = nearest_nodes(G, orig, dest)
            fastest_route, shortest_route = compute_routes(G, o_node, d_node)
        except Exception as e_bbox:
            print(f"[WARN] BBox falló ({e_bbox}). Probando polígono con BUFFER_KM={buffer_km:.2f}…")
            try:
                G = build_graph_polygon(orig, dest, buffer_km, mode=DEFAULT_MODE)
                o_node, d_node = nearest_nodes(G, orig, dest)
                fastest_route, shortest_route = compute_routes(G, o_node, d_node)
            except Exception as e_poly:
                last_err = (e_bbox, e_poly)
                buffer_km *= 1.8
                continue  # siguiente intento
        
        fastest_metrics = route_metrics(G, fastest_route)
        shortest_metrics = route_metrics(G, shortest_route)

        fastest_geom = unify_geometry(G, fastest_route)
        shortest_geom = unify_geometry(G, shortest_route)

        print("\n— Resultados —")
        print(f"Origen:  {origen_str}")
        print(f"Destino: {destino_str}")
        print("[Ruta MÁS RÁPIDA por 'travel_time']  "
              f"{human_time(fastest_metrics.time_s)}  |  {human_km(fastest_metrics.length_m)} km  |  nodos: {fastest_metrics.nodes}")
        print("[Ruta MÁS CORTA por 'length']       "
              f"{human_time(shortest_metrics.time_s)}  |  {human_km(shortest_metrics.length_m)} km  |  nodos: {shortest_metrics.nodes}")
        return

    raise SystemExit(f"[ERROR] No se pudo hallar una ruta tras {tries} intentos. Último error: {last_err}")


if __name__ == "__main__":
    main()
