"""
Starter: Greedy Best-First & A*, usando listas para OPEN
Formato y estructura inspirados en un starter de caminos mínimos.
Objetivo: que el/la estudiante implemente los algoritmos (cascarón con TODOs).

Instrucciones:
1) Implementá Greedy Best-First (f = h) en greedy_best_first():
   - Estructura de datos: OPEN como lista simple.
   - Selección del siguiente nodo: elegir el de menor h(n) con búsqueda lineal (O(N)).
   - Usá CLOSED para evitar reexpansión; registrá 'parent' y 'orden_expansion'.
2) Implementá A* (f = g + h) en a_star():
   - OPEN como lista simple; seleccionar por f(n) = g(n) + h(n).
   - Relajación: actualizá g[v] y parent[v] si encontrás mejor camino.
   - Evitá reexpansión con CLOSED (si h es consistente).
3) Completá reconstruct_path() para reconstruir el camino con 'parent'.
4) Probá con el grafo y heurísticas del final; compará orden de expansión y caminos.
"""

from typing import Dict, List, Tuple, Any, Callable, Optional
import math

Graph = Dict[Any, List[Tuple[Any, float]]]
Heuristic = Callable[[Any, Any], float]


# ---------------------------------------------------------------------------
# Greedy Best-First (f = h)
# ---------------------------------------------------------------------------

# Función auxiliar para argmin lineal, hace que el código sea más legible.
def argmin_index(lst: List[Any], key: Callable[[Any], float]) -> int: # índice del mínimo según 'key'
        return min(range(len(lst)), key=lambda i: key(lst[i])) # argmin_index([a,b,c], key=f) = índice de min(f(a), f(b), f(c))

def greedy_best_first(graph: Graph, start: Any, goal: Any, h: Heuristic) -> Tuple[Dict[Any, Any], List[Any]]:
    """
    Devuelve:
      - parent: antecesor inmediato para reconstruir el camino.
      - orden_expansion: lista de nodos en el orden de expansión (cuando salen de OPEN).
    Estrategia:
      - OPEN como lista simple; en cada iteración elegir el n con menor h(n).
      - CLOSED evita reexpansiones.
      - Detener cuando se expande 'goal'.
    TO DO: completar el cuerpo del algoritmo (ver comentarios dentro).
    """
    parent: Dict[Any, Any] = {start: None}
    orden_expansion: List[Any] = []

    # TO DO: inicializar OPEN con 'start'
    # TO DO: inicializar CLOSED como conjunto vacío
    OPEN: List[Any] = [start]
    CLOSED: set = set()

    # Sugerencia: función auxiliar (opcional) para argmin por h:
    # min_index = argmin_index(OPEN, key=lambda n: h(n, goal))

    # Bucle principal
    # while OPEN:
    #     1) seleccionar y remover de OPEN el nodo u con menor h(u)
    #     2) si u ya está en CLOSED, continuar
    #     3) agregar u a CLOSED y a 'orden'
    #     4) si u == goal: break
    #     5) para cada (v,w) en graph[u]:
    #           si v no está en CLOSED y v no está en parent:
    #               parent[v] = u
    #               agregar v a OPEN

    while OPEN:
        select_index = argmin_index(OPEN, key=lambda n: h(n, goal)) # Selecciona por h(n)
        i = OPEN.pop(select_index)  # Extrae el nodo con la heurística mínima
        
        for v, _ in graph.get(i, []): # Probar ambas heurísticas y elegir la que da menor valor
            if v not in CLOSED and v not in parent: # Evita reexpansión y ciclos
                parent[v] = i # Registra el antecesor
                OPEN.append(v) # Agrega al OPEN los vecinos no expandidos

        if i in CLOSED: # Evita reexpansión
            continue  # Ignora si ya fue expandido
        orden_expansion.append(i) # Registra el orden de expansión
        CLOSED.add(i) # Marca como expandido

        if i == goal: # Hace que el algoritmo termine al llegar al objetivo
            break  # Detener si se alcanzó el objetivo


    # Nota: no se usa g(n); Greedy prioriza solo h(n).
    # Dejar lanzada la excepción para que el/la estudiante implemente
    return parent, orden_expansion


# ---------------------------------------------------------------------------
# A* (f = g + h)
# ---------------------------------------------------------------------------
def a_star(graph: Graph, start: Any, goal: Any, h: Heuristic) -> Tuple[Dict[Any, float], Dict[Any, Any], List[Any]]:
    """
    Devuelve:
      - g: costos acumulados desde start.
      - parent: antecesores para reconstruir camino óptimo.
      - orden_expansion: nodos expandidos en orden.
    Estrategia:
      - OPEN lista simple; en cada iteración elegir el nodo con menor f = g + h (argmin lineal).
      - CLOSED evita reexpansión (si h consistente, no hay reexpansiones necesarias).
      - Relajación estándar: si new_g < g[v], actualizar g[v], parent[v] y (si corresponde) agregar v a OPEN.
    TO DO: completar el cuerpo.
    """
    g: Dict[Any, float] = {start: 0.0}
    parent: Dict[Any, Any] = {start: None}
    orden: List[Any] = []

    # TO DO: inicializar OPEN con 'start'
    # TO DO: inicializar CLOSED como conjunto vacío
    OPEN: List[Any] = [start]
    CLOSED: set = set()

    # Bucle principal
    # while OPEN:
    #     1) seleccionar y remover de OPEN el nodo u con menor f(u) = g[u] + h(u, goal)
    #     2) si u ya está en CLOSED: continuar
    #     3) agregar u a CLOSED y a 'orden'
    #     4) si u == goal: break
    #     5) para cada (v,w) en graph[u]:
    #           new_g = g[u] + w
    #           si v no está en g o new_g < g[v]:
    #               g[v] = new_g
    #               parent[v] = u
    #               si v no está en CLOSED y v no está en OPEN: agregar v a OPEN

    while OPEN:
        select_index = argmin_index(OPEN, key=lambda n: g[n] + h(n, goal)) # Selecciona por f(n) = g(n) + h(n)
        u = OPEN.pop(select_index)  # Extrae el nodo con el menor f(n) = g(n) + h(n)

        for v, w in graph.get(u, []): # Explora vecinos
            new_g = g[u] + w # Costo acumulado al vecino
            if v not in g or new_g < g[v]: # Relajación
                g[v] = new_g # Actualiza costo
                parent[v] = u # Actualiza antecesor
                if v not in CLOSED and v not in OPEN: # Agregar a OPEN si no está en CLOSED ni en OPEN
                    OPEN.append(v) # Agrega a OPEN
        
        if u in CLOSED: # Evita reexpansión
            continue  # Ignora si ya fue expandido
        orden.append(u) # Registra el orden de expansión
        CLOSED.add(u) # Marca como expandido

        if u == goal: # Hace que el algoritmo termine al llegar al objetivo
            break  # Detener si se alcanzó el objetivo

    return g, parent, orden


# ---------------------------------------------------------------------------
# Reconstrucción de camino
# ---------------------------------------------------------------------------
def reconstruct_path(parent: Dict[Any, Any], start: Any, goal: Any) -> List[Any]:
    """
    Devuelve la lista de nodos desde 'start' hasta 'goal' (incluidos).
    Si no hay camino o falta información en 'parent', devuelve [].
    TO DO: completar recorriendo 'parent' hacia atrás.
    """
    path: List[Any] = [start]
    cur: Optional[Any] = goal

    if goal not in parent: # hace que no se pueda reconstruir
        return []  # sin camino conocido
    path = []
    cur = goal

    while cur is not None: # Reconstruye el camino hacia atrás
        path.append(cur) # Agrega el nodo actual al camino
        cur = parent.get(cur) # Mueve al antecesor
    path.reverse() # Invierte para obtener el camino de start a goal

    return path if path and path[0] == start else [] # Verifica que el camino comience en 'start'



# ---------------------------------------------------------------------------
# Grafo de ejemplo y heurísticas
# ---------------------------------------------------------------------------

# Grafo no dirigido (lista de adyacencia con costos)
G: Graph = {
    'A': [('B', 2), ('C', 3)],
    'B': [('A', 2), ('D', 2)],
    'C': [('A', 3), ('D', 2), ('E', 3)],
    'D': [('B', 2), ('C', 2), ('F', 2), ('H', 3)],
    'E': [('C', 3), ('F', 2)],
    'F': [('D', 2), ('E', 2), ('G', 2)],
    'H': [('D', 3), ('G', 2)],
    'G': []  # objetivo
}

# Posiciones (x,y) para heurísticas geométricas (opcional)
POS = {
    'A': (0, 4),
    'B': (2, 4),
    'C': (0, 2),
    'D': (2, 2),
    'E': (0, 0),
    'F': (2, 0),
    'H': (4, 2),
    'G': (4, 0)
}

def h_manhattan(n: Any, goal: Any) -> float:
    (x1, y1), (x2, y2) = POS[n], POS[goal]
    return abs(x1 - x2) + abs(y1 - y2)

def h_euclid(n: Any, goal: Any) -> float:
    (x1, y1), (x2, y2) = POS[n], POS[goal]
    return math.hypot(x1 - x2, y1 - y2)


# ---------------------------------------------------------------------------
# Ejecución de prueba
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    start, goal = 'A', 'G'
    print("Grafo de ejemplo (nodos, posiciones):", POS)

    print("\n--- GREEDY BEST-FIRST (h = Manhattan) ---")
    try:
        parent_gbfs, orden_gbfs = greedy_best_first(G, start, goal, lambda n,g=goal: h_manhattan(n, g))
        print("Orden de expansión:", orden_gbfs)
        print("Camino:", reconstruct_path(parent_gbfs, start, goal))
    except NotImplementedError as e:
        print("Pendiente Greedy:", e)

    print("\n--- A* (h = Manhattan) ---")
    try:
        g_scores, parent_astar, orden_astar = a_star(G, start, goal, lambda n,g=goal: h_manhattan(n, g))
        print("Orden de expansión:", orden_astar)
        print("g(n):", g_scores)
        print("Camino óptimo A->G:", reconstruct_path(parent_astar, start, goal))
    except NotImplementedError as e:
        print("Pendiente A*:", e)


# -----------------------------------------------------------
# Resultados esperados (pueden variar en orden de expansión)
# ---------------------------------------------------------------------------
# Grafo de ejemplo (nodos, posiciones): {'A': (0, 4), 'B': (2, 4), 'C': (0, 2), 
# 'D': (2, 2), 'E': (0, 0), 'F': (2, 0), 'H': (4, 2), 'G': (4, 0)}

# --- GREEDY BEST-FIRST (h = Manhattan) ---
# Orden de expansión: ['A', 'B', 'D', 'F', 'G']
# Camino: ['A', 'B', 'D', 'F', 'G']

# --- A* (h = Manhattan) ---
# Orden de expansión: ['A', 'B', 'D', 'F', 'G']
# g(n): {'A': 0.0, 'B': 2.0, 'C': 3.0, 'D': 4.0, 'F': 6.0, 'H': 7.0, 'E': 8.0, 'G': 8.0}
# Camino óptimo A->G: ['A', 'B', 'D', 'F', 'G']
# Camino A→G: ['A', 'B', 'D', 'F', 'G']  Distancia: 8.0
# -----------------------------------------------------------














# ---------------------------------------------------------------------------
# Segunda Parte del Proceso
# (opcional, para visualización y pruebas adicionales)
# ---------------------------------------------------------------------------

# Visualización del proceso de búsqueda A* (opcional)
# Representación visual del grafo dirijido utilizando las funciones ya definidas
# Utilización de las funciones de la Tarea 3, modificadas para este contexto
# Cada linea de código está comentada para no perderme en el proceso

import networkx as nx
import matplotlib.pyplot as plt
from math import inf


class adjacency_matrix: # Placeholder para la estructura de matriz de adyacencia
    def __init__(self, nvertices): # nvertices: número de vértices (1..n)
        self.nvertices = nvertices # número de vértices
        self.weight = [[inf]*(nvertices+1) for _ in range(nvertices+1)] # matriz de pesos (1-indexed)
        for i in range(1, nvertices+1): # inicializa la diagonal a 0
            self.weight[i][i] = 0 # costo 0 en la diagonal
        self.next = [[None]*(nvertices+1) for _ in range(nvertices+1)] # matriz para reconstrucción de caminos

    def as_dict(self):
        """Devuelve un grafo {u: [(v, w), ...]} ignorando 'inf' y la diagonal."""
        adj = {i: [] for i in range(1, self.nvertices+1)} # diccionario de adyacencia
        for i in range(1, self.nvertices+1): # recorre filas
            for j in range(1, self.nvertices+1): # recorre columnas
                w = self.weight[i][j] # peso de la arista i->j
                if i != j and w != inf: # ignora diagonal e inf
                    adj[i].append((j, w)) # agrega (j, w) a la lista de adyacencia de i
        return adj


# --------------------------
# Helpers de graficación
# --------------------------
def matrix_to_nx(g: adjacency_matrix) -> nx.DiGraph: # hace un DiGraph de networkx
    """Convierte la matriz de Skiena a un DiGraph de networkx."""
    G = nx.DiGraph() # grafo dirigido
    G.add_nodes_from(range(1, g.nvertices+1)) # agrega nodos 1..n
    for i in range(1, g.nvertices+1): # recorre filas
        for j in range(1, g.nvertices+1): # recorre columnas
            w = g.weight[i][j] # peso de la arista i->j
            if w != inf and i != j: # ignora inf y diagonal
                G.add_edge(i, j, weight=w) # agrega arista i->j con peso w
    return G


def plot_graph(G, path_nodes=None, title="Grafo", node_labels=None, pos=None): # path_nodes: lista de nodos en el camino
    plt.figure(figsize=(8, 6)) # tamaño de la figura
    if pos is None: # si no se pasan posiciones, usa spring layout
        pos = nx.spring_layout(G, seed=7) # posiciones de los nodos
    nx.draw_networkx_nodes(G, pos, node_size=700) # dibuja nodos

    if node_labels: # si se pasan etiquetas, las usa
        nx.draw_networkx_labels(G, pos, labels=node_labels) # etiquetas personalizadas
    else: # si no, usa los números de los nodos
        nx.draw_networkx_labels(G, pos) # etiquetas por defecto

    nx.draw_networkx_edges(G, pos, edge_color="black", arrows=True, width=1.5) # dibuja aristas
    edge_labels = {(u, v): f"{d.get('weight')}" for u, v, d in G.edges(data=True)} # etiquetas de pesos
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="black") # dibuja etiquetas de pesos

    if path_nodes and len(path_nodes) >= 2: # si hay camino, lo resalta
        path_edges = list(zip(path_nodes[:-1], path_nodes[1:])) # aristas del camino
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color="red", width=3, arrows=True) # resalta camino

    plt.title(title)
    plt.axis("off")
    plt.show()



def shortest_path_and_plot(g: adjacency_matrix, src: int, dst: int, title=None, label_map=None, pos=None): 
    g_scores, parent, orden = a_star(g.as_dict(), src, dst, lambda n, g=dst: 0.0) # h(n)=0
    path = reconstruct_path(parent, src, dst) # reconstruye el camino
    dist = g_scores.get(dst, math.inf) # distancia total (inf si no hay camino)
    G = matrix_to_nx(g) # convierte a networkx

    node_labels = None # etiquetas por defecto (números)
    if label_map: # si se pasa un mapeo, lo invierte para etiquetas
        node_labels = {v: k for k, v in label_map.items()} # etiquetas personalizadas

    plot_graph(G, path_nodes=path, title=title or f"Camino minimo desde {src} hasta {dst}", # título
               node_labels=node_labels, pos=pos)
    return path, dist



def example_graph():
    g = adjacency_matrix(8)
    idx = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'H':7, 'G':8}  # mapeo letras->índices

    def e(u, v, w): 
        g.weight[idx[u]][idx[v]] = w  # dirigido

    e('A','B',2); e('B','A',2)
    e('A','C',3); e('C','A',3)
    e('B','D',2); e('D','B',2)
    e('C','D',2); e('D','C',2)
    e('C','E',3); e('E','C',3)
    e('D','F',2); e('F','D',2)
    e('E','F',2); e('F','E',2)
    e('D','H',3); e('H','D',3)
    e('F','G',2); e('G','F',2)

    POS = {
        'A': (0, 4),
        'B': (2, 4),
        'C': (0, 2),
        'D': (2, 2),
        'E': (0, 0),
        'F': (2, 0),
        'H': (4, 2),
        'G': (4, 0)
    }
    pos_idx = {idx[k]: v for k, v in POS.items()} # posiciones con índices
    return g, idx, pos_idx



# ---------------------------------------------------------------------------
# Ejecución de prueba
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    g, idx, pos_idx = example_graph()

    src, dst = idx['A'], idx['G'] # Prueba de A* con h(n)=0 (Dijkstra)
    path, dist = shortest_path_and_plot(g, src, dst,
                                        title="Camino mínimo desde A hasta G",
                                        label_map=idx,
                                        pos=pos_idx)
    path_letras = [ {v:k for k,v in idx.items()}[p] for p in path ]
    print("Camino A→G:", path_letras, " Distancia:", dist)
# ---------------------------------------------------------------------------


