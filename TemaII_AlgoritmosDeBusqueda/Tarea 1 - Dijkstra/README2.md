# Dijkstra + OSMnx (dijikstra_osmnx_mvd.py) — README

Este proyecto busca **visualizar una implementación real** del algoritmo de **Dijkstra**, aplicada al problema de encontrar el camino más rápido en la ciudad de Montevideo.

En lugar de un ejemplo sintético con grafos pequeños y sencillo de programar, con ayuda de OpenAI utilicé la red vial real obtenida desde **OpenStreetMap** mediante **OSMnx**, lo que permite trabajar con calles, distancias y tiempos de viaje de forma práctica y cercana a un caso real: ir desde mi dirección personal (Rambla República de Chile 4275) hasta la **Universidad Católica del Uruguay (UCU)** en Av. 8 de Octubre & Garibaldi.

---

## 1) Requisitos

* Python 3.10+
* Paquetes necesarios:

  ```bash
  pip install osmnx networkx shapely scykit-learn
  ```

---

## 2) Uso rápido

Ejecuta el script principal con los valores por defecto:

```bash
python dijkstra_osmnx_mvd.py
```

Esto construirá la red vial, imputará velocidades y tiempos de viaje, y calculará:

* **Ruta más rápida** (minimiza `travel_time`).
* **Ruta más corta** (minimiza `length`).

Los resultados se mostrarán por consola (tiempo estimado y distancia en kilómetros).

---

## 3) CLI / Parámetros

```bash
python dijkstra_osmnx_mvd.py \
  --origen "Rambla República de Chile 4275, Montevideo, Uruguay" \
  --dest-lat -34.8894 --dest-lon -56.1601 \  # No me reconocia la dirección de la católica, use directamente lat-lon
  --buffer-km 3 \
  --max-tries 3
```

**Descripción de flags:**

* `--origen` dirección del origen (se intenta geocodificar; si falla, usa fallback).
* `--orig-lat`, `--orig-lon` permiten definir coordenadas manualmente (ignoran `--origen`).
* `--dest-lat`, `--dest-lon` coordenadas del destino (usadas por robustez).
* `--buffer-km` tamaño del área inicial alrededor de la línea origen–destino.
* `--max-tries` reintentos ampliando el área si no se encuentra ruta.

---

## 4) Qué hace el script

1. **Geocodifica** la dirección de origen (con fallback a coordenadas si falla).
2. Usa coordenadas fijas para el destino (evita ambigüedades en intersecciones).
3. Construye un **bbox** (o polígono de fallback) alrededor de los puntos de origen y destino.
4. Descarga la red vial de Montevideo para autos.
5. **Imputa velocidades** (`speed_kph`) y **tiempos de viaje** (`travel_time`).
6. Ejecuta **Dijkstra** para calcular:
   * La **ruta más rápida** (en minutos).
   * La **ruta más corta** (en kilómetros).
7. Muestra los resultados en consola.

---

## 5) Fundamento algorítmico

* El cálculo de la **ruta más rápida** se realiza con el algoritmo de **Dijkstra** ponderado por `travel_time`.
* La **ruta más corta** usa el mismo grafo ponderado por `length`.
* Complejidad: O((E+V)\log V) en grafos escasos con heap binario.

---

## 6) Licencia y créditos

* **OpenStreetMap**: datos ODbL de la comunidad OSM.
* **OSMnx** y **NetworkX**: librerías open source utilizadas.
* **ChatGPT (OpenAI)**: asistencia en la escritura del código.