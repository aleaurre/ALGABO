"""
Starter: Greedy Best-First & A*, usando listas para OPEN
Objetivo: implementar y probar los algoritmos sobre un grafo ya definido (G, POS).
Se eliminó la segunda parte con adjacency_matrix/example_graph; todo usa el grafo inicial.
"""

from typing import Dict, List, Tuple, Any, Callable, Optional
import math

# ---------------------------------------------------------------------------
# Tipos
# ---------------------------------------------------------------------------
Graph = Dict[Any, List[Tuple[Any, float]]]
Heuristic = Callable[[Any, Any], float]


# ---------------------------------------------------------------------------
# Greedy Best-First (f = h)
# ---------------------------------------------------------------------------

def argmin_index(lst: List[Any], key: Callable[[Any], float]) -> int:
    """Índice del mínimo según 'key' (argmin lineal)."""
    return min(range(len(lst)), key=lambda i: key(lst[i]))

def greedy_best_first(graph: Graph, start: Any, goal: Any, h: Heuristic) -> Tuple[Dict[Any, Any], List[Any]]:
    """
    Devuelve:
      - parent: antecesor inmediato para reconstruir el camino.
      - orden_expansion: lista de nodos en el orden de expansión (cuando salen de OPEN).
    Estrategia:
      - OPEN como lista simple; en cada iteración elegir el n con menor h(n).
      - CLOSED evita reexpansiones.
      - Detener cuando se expande 'goal'.
    """
    parent: Dict[Any, Any] = {start: None}
    orden_expansion: List[Any] = []

    OPEN: List[Any] = [start]
    CLOSED: set = set()

    while OPEN:
        select_index = argmin_index(OPEN, key=lambda n: h(n, goal))
        u = OPEN.pop(select_index)

        if u in CLOSED:
            continue
        orden_expansion.append(u)
        CLOSED.add(u)

        if u == goal:
            break  # alcanzado objetivo

        for v, _ in graph.get(u, []):
            if v not in CLOSED and v not in parent:
                parent[v] = u
                OPEN.append(v)

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
    """
    g: Dict[Any, float] = {start: 0.0}
    parent: Dict[Any, Any] = {start: None}
    orden: List[Any] = []

    OPEN: List[Any] = [start]
    CLOSED: set = set()

    while OPEN:
        select_index = argmin_index(OPEN, key=lambda n: g[n] + h(n, goal))
        u = OPEN.pop(select_index)

        if u in CLOSED:
            continue
        orden.append(u)
        CLOSED.add(u)

        if u == goal:
            break

        for v, w in graph.get(u, []):
            new_g = g[u] + w
            if v not in g or new_g < g[v]:
                g[v] = new_g
                parent[v] = u
                if v not in CLOSED and v not in OPEN:
                    OPEN.append(v)

    return g, parent, orden


# ---------------------------------------------------------------------------
# Reconstrucción de camino
# ---------------------------------------------------------------------------
def reconstruct_path(parent: Dict[Any, Any], start: Any, goal: Any) -> List[Any]:
    """
    Devuelve la lista de nodos desde 'start' hasta 'goal' (incluidos).
    Si no hay camino o falta info en 'parent', devuelve [].
    """
    if goal not in parent:
        return []
    path: List[Any] = []
    cur: Optional[Any] = goal
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path if path and path[0] == start else []


# ---------------------------------------------------------------------------
# Grafo y heurísticas (ÚNICA fuente de verdad)
# ---------------------------------------------------------------------------
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
# (Opcional) Visualización directa desde G y POS con networkx
# ---------------------------------------------------------------------------
try:
    import networkx as nx
    import matplotlib.pyplot as plt

    def dict_graph_to_nx(graph: Graph) -> "nx.DiGraph":
        """Convierte el diccionario de adyacencias en un DiGraph de networkx (dirigido)."""
        H = nx.DiGraph()
        for u, neighs in graph.items():
            H.add_node(u)
            for v, w in neighs:
                H.add_edge(u, v, weight=w)
        return H

    def plot_graph_from_dict(graph: Graph, pos=POS, path_nodes=None, title="Grafo (G)"):
        """Dibuja el grafo, con camino resaltado si se provee path_nodes."""
        H = dict_graph_to_nx(graph)
        plt.figure(figsize=(8, 6))
        nx.draw_networkx_nodes(H, pos, node_size=700)
        nx.draw_networkx_labels(H, pos)
        nx.draw_networkx_edges(H, pos, edge_color="black", arrows=True, width=1.5)
        edge_labels = {(u, v): f"{d.get('weight')}" for u, v, d in H.edges(data=True)}
        nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels)

        if path_nodes and len(path_nodes) >= 2:
            path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
            nx.draw_networkx_edges(H, pos, edgelist=path_edges, edge_color="red", width=3, arrows=True)

        plt.title(title)
        plt.axis("off")
        plt.show()

except Exception as _e: # Si networkx/matplotlib no están disponibles, las funciones de plotting no se cargan.
    pass


# ---------------------------------------------------------------------------
# Ejecución de prueba
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    start, goal = 'A', 'G'
    print("Grafo (nodos, posiciones):", POS)

    print("\n--- GREEDY BEST-FIRST (h = Manhattan) ---")
    parent_gbfs, orden_gbfs = greedy_best_first(G, start, goal, lambda n, g=goal: h_manhattan(n, g))
    print("Orden de expansión:", orden_gbfs)
    print("Camino:", reconstruct_path(parent_gbfs, start, goal))

    print("\n--- A* (h = Manhattan) ---")
    g_scores, parent_astar, orden_astar = a_star(G, start, goal, lambda n, g=goal: h_manhattan(n, g))
    print("Orden de expansión:", orden_astar)
    print("g(n):", g_scores)
    path_astar = reconstruct_path(parent_astar, start, goal)
    print("Camino óptimo A->G:", path_astar)
    if goal in g_scores:
        print("Distancia:", g_scores[goal])

    # Visualización opcional (si networkx/matplotlib están disponibles)
    try:
        plot_graph_from_dict(G, pos=POS, path_nodes=path_astar, title="Camino A* resaltado (G)")
    except NameError:
        pass
