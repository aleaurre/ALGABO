from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional

INF = float("inf")

class IndexMinPQ:
    """
    Cola de prioridad mínima indexada por 'índices' enteros [0..N-1].
    Soporta: insert(i, key), contains(i), decrease_key(i, key), pop_min().
    Internamente: binary heap + arrays paralelos 'pq' (heap), 'qp' (posiciones), 'keys'.
    """
    def __init__(self, max_n: int):
        self.n = 0
        self.max_n = max_n
        self.pq = [0]*(max_n+1)        # heap de índices; 1-based
        self.qp = [-1]*(max_n)         # pos de cada índice en el heap, -1 si no está
        self.keys: List[float] = [INF]*max_n

    def is_empty(self) -> bool:
        return self.n == 0

    def contains(self, i: int) -> bool:
        return self.qp[i] != -1

    def insert(self, i: int, key: float) -> None:
        if self.contains(i):
            raise ValueError("El índice ya está en la PQ")
        self.n += 1
        self.qp[i] = self.n
        self.pq[self.n] = i
        self.keys[i] = key
        self._swim(self.n)

    def decrease_key(self, i: int, key: float) -> None:
        if not self.contains(i):
            raise ValueError("Índice no presente en la PQ")
        if key >= self.keys[i]:
            return
        self.keys[i] = key
        self._swim(self.qp[i])

    def pop_min(self) -> int:
        if self.n == 0:
            raise IndexError("pop_min en PQ vacía")
        min_idx = self.pq[1]
        self._exch(1, self.n)
        self.n -= 1
        self._sink(1)
        self.qp[min_idx] = -1
        return min_idx

    # --- helpers heap ---
    def _less(self, a: int, b: int) -> bool:
        return self.keys[self.pq[a]] < self.keys[self.pq[b]]

    def _exch(self, a: int, b: int) -> None:
        self.pq[a], self.pq[b] = self.pq[b], self.pq[a]
        self.qp[self.pq[a]] = a
        self.qp[self.pq[b]] = b

    def _swim(self, k: int) -> None:
        while k > 1 and self._less(k, k//2):
            self._exch(k, k//2)
            k //= 2

    def _sink(self, k: int) -> None:
        while 2*k <= self.n:
            j = 2*k
            if j < self.n and self._less(j+1, j):
                j += 1
            if not self._less(j, k):
                break
            self._exch(k, j)
            k = j


class Graph:
    """
    Grafo ponderado dirigido con pesos no negativos.
    Permite construir desde lista de adyacencia o matriz de adyacencia.
    Mantiene un mapeo {nodo:any -> idx:int} para usar con la PQ indexada.
    """
    def __init__(self):
        self.idx_of: Dict[Any, int] = {}
        self.node_of: List[Any] = []
        self.adj: List[List[Tuple[int, float]]] = []

    def _ensure_node(self, u: Any) -> int:
        if u in self.idx_of:
            return self.idx_of[u]
        idx = len(self.node_of)
        self.idx_of[u] = idx
        self.node_of.append(u)
        self.adj.append([])
        return idx

    def add_edge(self, u: Any, v: Any, w: float) -> None:
        if w < 0:
            raise ValueError("Dijkstra no admite pesos negativos")
        iu, iv = self._ensure_node(u), self._ensure_node(v)
        self.adj[iu].append((iv, w))

    @classmethod
    def from_adj_list(cls, alist: Dict[Any, List[Tuple[Any, float]]]) -> "Graph":
        g = cls()
        # asegurar todos los nodos
        for u in alist:
            g._ensure_node(u)
            for v, _ in alist[u]:
                g._ensure_node(v)
        # agregar aristas
        for u in alist:
            for v, w in alist[u]:
                g.add_edge(u, v, w)
        return g

    @classmethod
    def from_adj_matrix(cls, nodes: List[Any], W: List[List[Optional[float]]]) -> "Graph":
        """
        W[i][j] = peso (None o INF si no hay arista). Diagonal implícita 0.
        """
        g = cls()
        for u in nodes:
            g._ensure_node(u)
        n = len(nodes)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                w = W[i][j]
                if w is not None and w != INF:
                    g.add_edge(nodes[i], nodes[j], float(w))
        return g


class DijkstraSolver:
    """
    Dijkstra con Priority Queue indexada (decrease_key real).
    Retorna distancias y padres para reconstruir caminos.
    """
    def __init__(self, g: Graph):
        self.g = g

    def run(self, source: Any) -> Tuple[Dict[Any, float], Dict[Any, Any]]:
        if source not in self.g.idx_of:
            raise KeyError(f"source '{source}' no está en el grafo")
        n = len(self.g.node_of)
        s = self.g.idx_of[source]
        dist: List[float] = [INF]*n
        parent_idx: List[Optional[int]] = [None]*n

        pq = IndexMinPQ(n)
        dist[s] = 0.0
        pq.insert(s, 0.0)

        while not pq.is_empty():
            u = pq.pop_min()
            du = dist[u]
            for v, w in self.g.adj[u]:
                nd = du + w
                if nd < dist[v]:
                    dist[v] = nd
                    parent_idx[v] = u
                    if pq.contains(v):
                        pq.decrease_key(v, nd)
                    else:
                        pq.insert(v, nd)

        # mapear a etiquetas originales
        dist_map = {self.g.node_of[i]: dist[i] for i in range(n)}
        parent_map = {}
        for i, p in enumerate(parent_idx):
            if p is not None:
                parent_map[self.g.node_of[i]] = self.g.node_of[p]
        return dist_map, parent_map

    @staticmethod
    def reconstruct_path(parent: Dict[Any, Any], source: Any, target: Any) -> List[Any]:
        path: List[Any] = []
        cur: Optional[Any] = target
        while cur is not None:
            path.append(cur)
            if cur == source:
                break
            cur = parent.get(cur, None)
        path.reverse()
        return path if path and path[0] == source and path[-1] == target else []


# --- Ejemplo de uso ---
if __name__ == "__main__":
    # 1) Desde lista de adyacencia (igual que el ejemplo dimos en clase)
    alist = {
        "A": [("B", 2), ("C", 5)],
        "B": [("C", 1), ("D", 3)],
        "C": [("D", 2)],
        "D": []
    }
    g1 = Graph.from_adj_list(alist)
    solver = DijkstraSolver(g1)
    dist, parent = solver.run("A")
    print("Distancias desde A:", dist)        # {'A':0.0,'B':2.0,'C':3.0,'D':5.0}
    print("Camino A->D:", DijkstraSolver.reconstruct_path(parent, "A", "D"))

    # 2) Desde matriz de adyacencia (None = sin arista)
    nodes = ["A","B","C","D"]
    W = [
        [0,   2,   5,  None],
        [None,0,   1,   3   ],
        [None,None,0,   2   ],
        [None,None,None,0   ],
    ]
    g2 = Graph.from_adj_matrix(nodes, W)
    solver2 = DijkstraSolver(g2)
    dist2, parent2 = solver2.run("A")
    print("Distancias (matriz) desde A:", dist2)
    print("Camino (matriz) A->D:", DijkstraSolver.reconstruct_path(parent2, "A", "D"))


