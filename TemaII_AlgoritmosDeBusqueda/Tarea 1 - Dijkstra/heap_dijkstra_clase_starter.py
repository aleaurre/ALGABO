# Heaps & Dijkstra — Starter

# Instrucciones:
# 1) Completa la función dijkstra usando heapq como cola de prioridad (min-heap).
# 2) Completa reconstruct_path para devolver la lista de nodos desde source a target.
# 3) Ejecuta este archivo para probarlo con el grafo de ejemplo.
#
# Tip: en heapq no existe "decrease-key"; empuja una nueva tupla (dist, nodo) 
# y descarta entradas obsoletas.

from typing import Dict, List, Tuple, Any
import heapq

Graph = Dict[Any, List[Tuple[Any, float]]]

def dijkstra(graph: Graph, source: Any) -> Tuple[Dict[Any, float], Dict[Any, Any]]:
    """
    Devuelve:
      dist: distancia mínima estimada desde 'source' a cada nodo.
      parent: antecesor inmediato para reconstruir caminos.
    Asume pesos no negativos.
    """
    # TO DO: inicializa dist con infinito y parent con None.
    dist = {node: float('inf') for node in graph}
    parent = {node: None for node in graph}
    
    dist[source] = 0.0

    # TO DO: crea un heap y agrega (0, source).
    heap = [(0, source)] # El primer elemento es la distancia, el segundo es el nodo.
    heapq.heapify(heap)

    # TO DO: bucle principal.
    while heap:
        d, u = heapq.heappop(heap) # Extrae el nodo con la distancia mínima.
        if d != dist[u]: # Ignora entradas obsoletas.
            continue 

        for v, w in graph.get(u, []): # Relaja aristas salientes, v es el vecino y w es el peso.
            if w < 0: # Asegúrate de que w sea no negativo.
                raise ValueError("Los pesos deben ser no negativos.")
            if dist[u] + w < dist[v]:  # Relajación. Distancia de la fuente a v.
                dist[v] = dist[u] + w  # Actualiza la distancia mínima.
                parent[v] = u
                heapq.heappush(heap, (dist[v], v))

    return dist, parent

def reconstruct_path(parent: Dict[Any, Any], source: Any, target: Any) -> List[Any]:
    """
    Devuelve la lista de nodos desde source hasta target (incluidos).
    Si no hay camino, devuelve [].
    """
    # TO DO: reconstruye desde target hacia source usando 'parent'.
    path = []
    cur = target

    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()  # Invertir para tener el camino desde source a target.

    if path[0] != source:
        return []  # Si el camino no comienza en source, devuelve lista vacía.
    
    return path


if __name__ == '__main__':
    # Grafo de ejemplo (dirigido). Pesos >= 0.
    graph: Graph = {
        'A': [('B', 2), ('C', 5)],
        'B': [('C', 1), ('D', 3)],
        'C': [('D', 2)],
        'D': []
    }
    dist, parent = dijkstra(graph, 'A')
    print("Distancias desde A:", dist)
    print("Camino A->D:", reconstruct_path(parent, 'A', 'D'))

