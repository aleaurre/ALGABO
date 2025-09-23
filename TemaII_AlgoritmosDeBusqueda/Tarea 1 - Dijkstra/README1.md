# Dijkstra con POO e Indexed Min-Priority Queue (dijkstra.py)

Este proyecto implementa el algoritmo de **Dijkstra** para encontrar caminos mínimos en un grafo ponderado con pesos no negativos. Aquí se incorpora:
* **Programación Orientada a Objetos (POO)**, encapsulando la lógica en clases `Graph`, `IndexMinPQ` y `DijkstraSolver`.
* Una **cola de prioridad indexada (Indexed Min-PQ)**.
* Soporte para construir el grafo tanto desde **listas de adyacencia** como desde **matrices de adyacencia**.

---

## Diferencias principales con el código trabajado en clase

* El código de referencia usaba `heapq` con reinserciones múltiples y descartes de entradas viejas.
* En esta versión, cada nodo tiene **una sola clave viva en el heap**, lo que reduce tamaño de la cola de prioridad y evita sobrecarga de pops innecesarios.
* Se prioriza claridad estructural mediante clases, facilitando su extensión.

---

## Ejemplo de uso

```bash
python dijkstra.py
```

Salida esperada (para el grafo de ejemplo):

```
Distancias desde A: {'A': 0.0, 'B': 2.0, 'C': 3.0, 'D': 5.0}
Camino A->D: ['A', 'B', 'D']
```

---

## Bibliografía y aportes conceptuales

* **Sedgewick, R., & Wayne, K. (2022). Princeton Algorithms (IndexMinPQ.java).**
  Implementación de la **cola de prioridad indexada**, tomé estructuras de ese diseño en Java para Python.

* **Chen, Chowdhury, Ramachandran & Roche (2007). Priority queues and Dijkstra’s algorithm.**
  Qúe son y por qué se utilizan estructuras como `decrease_key` para mejorar la performance en Dijkstra.

---

## Diagrama de Clases

classDiagram
    class Graph {
        +dict idx_of
        +list node_of
        +list adj
        +add_edge(u, v, w) void
        +from_adj_list(alist) Graph
        +from_adj_matrix(nodes, W) Graph
        -_ensure_node(u) int}

    class IndexMinPQ {
        -int n
        -int max_n
        -list pq
        -list qp
        -list keys
        +is_empty() bool
        +contains(i) bool
        +insert(i, key) void
        +decrease_key(i, key) void
        +pop_min() int
        -_less(a, b) bool
        -_exch(a, b) void
        -_swim(k) void
        -_sink(k) void}

    class DijkstraSolver {
        -Graph g
        +run(source) (dict dist, dict parent)
        +reconstruct_path(parent, source, target) list}

    DijkstraSolver --> Graph : usa
    DijkstraSolver --> IndexMinPQ : usa

---

## Licencia

Incluye ideas de autores referenciados y videos de Youtube/Tiktok.

---