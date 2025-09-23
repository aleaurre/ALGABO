
# Tarea 2 - Implementación de algoritmo IDA
# En base a la tarea 1 vista en clase y la bibliografia sugerida. 
# (https://webasignatura.ucu.edu.uy/mod/page/view.php?id=831568) 
# Se pide:
#   1) Adaptar el código visto en la tarea e Implementar IDA* utilizando el mismo template base
#   2) implementar dos nuevas heurísticas sencillas.
# Se debe entregar un archivo en python.



# ----------------------------------------------------------
# Implementación del algoritmo IDA*
# ----------------------------------------------------------
from math import inf  
from typing import Any, Callable, Iterable, List, Tuple  

# Algoritmo IDA* (Iterative Deepening A*)
# El algoritmo busca un camino desde el nodo 'inicio' hasta un nodo meta
# utilizando una función heurística 'h' para guiar la búsqueda.
# Se suele utilizar en problemas de búsqueda en grafos donde se desea
# encontrar el camino óptimo de manera eficiente en memoria.

def ida_estrella(inicio: Any,
                 es_meta: Callable[[Any], bool],
                 sucesores: Callable[[Any], Iterable[Tuple[Any, float]]],
                 h: Callable[[Any], float],
                 ordenar_sucesores: bool = True) -> List[Any] | None:
    
    umbral = h(inicio)                           # cota inicial sobre f = g + h
    camino = [inicio]                            # camino parcial inicial
    while True:                                  # iteraciones con poda por umbral
        encontrado, nuevo_umbral = _dfs_acotado( # llamada a DFS acotado
            n=inicio, g=0.0, umbral=umbral,
            camino_parcial=camino, es_meta=es_meta,
            sucesores=sucesores, h=h,
            ordenar_sucesores=ordenar_sucesores
        )
        if isinstance(encontrado, list):         # si devuelve una lista, es el camino solución
            return encontrado                    # solución encontrada
        if nuevo_umbral == inf:                  # no hay nada por debajo de infinito
            return None                          # sin solución alcanzable
        umbral = nuevo_umbral                    # subo la cota y repito


def _dfs_acotado(n: Any, g: float, umbral: float, camino_parcial: List[Any],
                 es_meta: Callable[[Any], bool],
                 sucesores: Callable[[Any], Iterable[Tuple[Any, float]]],
                 h: Callable[[Any], float],
                 ordenar_sucesores: bool) -> Tuple[List[Any] | bool, float]:
    
    f = g + h(n)                                 # evaluación f(n) = g(n) + h(n)
    if f > umbral:                               # si excede la cota actual
        return False, f                          # propone nueva cota mínima (exceso)
    if es_meta(n):                               # test de objetivo
        return list(camino_parcial), f           # retorna el camino encontrado

    min_exceso = inf                             # mejor exceso observado en esta rama

    suces = list(sucesores(n))                   # obtiene sucesores (hijo, costo)
    if ordenar_sucesores:                        # opcional: explorar prometedores primero
        suces.sort(key=lambda hc: h(hc[0]))      # ordena por h(hijo)

    for hijo, costo in suces:                    # recorre cada sucesor
        if hijo in camino_parcial:               # evita ciclos en el camino actual
            continue                             # salta hijos ya visitados en el camino
        camino_parcial.append(hijo)              # agrega hijo al camino
        ok, exceso = _dfs_acotado(               # desciende recursivamente
            n=hijo, g=g + costo, umbral=umbral,
            camino_parcial=camino_parcial,
            es_meta=es_meta, sucesores=sucesores,
            h=h, ordenar_sucesores=ordenar_sucesores
        )
        if isinstance(ok, list):                 # si volvió con un camino solución
            return ok, exceso                    # propaga solución hacia arriba
        if exceso < min_exceso:                  # actualiza el menor exceso observado
            min_exceso = exceso                  # guarda nuevo umbral candidato
        camino_parcial.pop()                     # backtracking: quita hijo del camino

    return False, min_exceso                     # no se encontró solución en esta rama






    
# ----------------------------------------------------------
# Implementación de nuevas Heurísticas
# ----------------------------------------------------------
from math import sqrt  # raíz cuadrada para el costo diagonal


# Esta función genera una heurística de distancia octile
# La distancia octile es adecuada para movimientos en 8 direcciones (horizontales, verticales y diagonales)
# Se define como una combinación ponderada de la distancia Manhattan y la distancia diagonal.
def make_octile_h(coords: dict, goal, cost_straight: float = 1.0, cost_diag: float = sqrt(2.0)):
    gx, gy = coords[goal]                                    # coordenadas del objetivo
    def h(n):
        x, y = coords[n]                                     # coordenadas del nodo n
        dx = abs(x - gx)                                     # diferencia en x
        dy = abs(y - gy)                                     # diferencia en y
        m = min(dx, dy)                                      # pasos diagonales posibles
        r = max(dx, dy) - m                                  # pasos rectos restantes
        return m * cost_diag + r * cost_straight             # distancia octile (admisible)
    return h                                                 # devuelve h(n) cerrada sobre goal


# Esta función genera una heurística de distancia Chebyshev
# La distancia Chebyshev es adecuada para movimientos en 8 direcciones (horizontales, verticales y diagonales)
# Se define como el máximo de las diferencias absolutas en las coordenadas x e y.
def make_chebyshev_h(coords, goal):
    gx, gy = coords[goal]                      # coordenadas del objetivo
    def h(n):
        x, y = coords[n]                       # coordenadas del nodo n
        return max(abs(x - gx), abs(y - gy))   # n de pasos mínimos con diagonales
    return h                                   # devuelve la heurística h(n)


# Esta función genera una heurística basada en el grado del nodo
# La heurística devuelve la diferencia en el número de conexiones (grado) entre el nodo actual
# y el nodo objetivo. La idea es que los nodos con un grado similar al del objetivo pueden estar más cerca.
def make_degree_h(grafo, goal):
    def h(n):
        return abs(len(grafo[n]) - len(grafo[goal]))  # diferencia en conexiones
    return h                                          # devuelve la heurística h(n)


# Esta función genera una heurística que penaliza los nodos en posiciones bloqueadas
# La heurística devuelve la distancia Manhattan al objetivo, pero añade una penalización
# si el nodo actual está en una posición bloqueada (obstáculo).
def make_blocked_h(coords, obstacles, goal):
    gx, gy = coords[goal]                              # coordenadas del objetivo
    def h(n):
        x, y = coords[n]                               # coordenadas del nodo n
        base = abs(x - gx) + abs(y - gy)               # distancia Manhattan básica
        penalty = 2 if (x, y) in obstacles else 0      # suma 2 si está en obstáculo
        return base + penalty                          # heurística = base + penalización
    return h                                           # devuelve la heurística h(n)




# ----------------------------------------------------------
# Pruebas aplicadas al Ejercicio trabajado en Clase (Grafo Simple)
# ----------------------------------------------------------
from typing import Dict

Graph = Dict[Any, List[Tuple[Any, float]]]
G: Graph = {
    'A': [('B', 2), ('C', 3)],
    'B': [('A', 2), ('D', 2)],
    'C': [('A', 3), ('D', 2), ('E', 3)],
    'D': [('B', 2), ('C', 2), ('F', 2), ('H', 3)],
    'E': [('C', 3), ('F', 2)],
    'F': [('D', 2), ('E', 2), ('G', 2)],
    'H': [('D', 3), ('G', 2)],
    'G': []  
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

START, GOAL = 'A', 'G'                     # inicio y meta
STEP = 2                                   # cada “celda” en POS salta de 2 en 2
C_STRAIGHT, C_DIAG = 2, 3                  # costos de arista recta / diagonal

def sucesores_graph(n):                                   # sucesores(n): lista de (hijo,costo)
    return G.get(n, [])                                   # devuelve lo que ya está en G
def es_meta_graph(n):                                     # es_meta(n)
    return n == GOAL                                      # true si llegamos a GOAL

grafo_degree = {u: [v for (v, _) in G.get(u, [])]         # adyacencia sin pesos
                for u in G}                               # lo necesita make_degree_h

# Chebyshev de tu lib suele dar “número de pasos”; escalo a unidades del grafo
h_cheb_steps = make_chebyshev_h(POS, GOAL)                # h en pasos (según tus defs)
h_chebyshev  = lambda n: max(0, (h_cheb_steps(n) * C_STRAIGHT))  # x2 por paso             
h_octile = make_octile_h(POS, GOAL,                       # fábrica de octile ya definida
                         cost_straight=C_STRAIGHT,        # pasa costo recto = 2
                         cost_diag=C_DIAG)                # pasa costo diagonal = 3
h_degree = make_degree_h(grafo_degree, GOAL)              # |deg(n) - deg(goal)|
h_bloc_base = make_blocked_h(POS, set(), GOAL)            # en este caso sin obstáculos, conjunto vacio
h_bloc = lambda n: max(0, h_bloc_base(n) // (10//C_STRAIGHT) )  # adapta de 10->2 si tu bloc usa 10  


def costo_camino(path):                                   # costo real del camino con pesos G
    if not path:                                          # si no hay camino
        return None                                       # no se puede calcular costo
    total = 0                                             # acumulador
    for a, b in zip(path, path[1:]):                      # pares consecutivos del camino
        w = next(w for v, w in G[a] if v == b)            # busca peso (a-b) en G
        total += w                                        # suma el peso
    return total                                          # devuelve el costo total

tests = {                                                 # set de heurísticas a probar
    "Chebyshev": h_chebyshev,                             # chebyshev escalada a costo 2
    "Octile":  h_octile,                                  # octile con 2/3
    "Degree":       h_degree,                             # diferencia de grados
    "Bloc(adapt)":  h_bloc                                # bloc adaptada (sin obstáculos)
}

for nombre, h in tests.items():                           # recorre cada heurística
    camino = ida_estrella(START, es_meta_graph,           # corre IDA* con tu función
                          sucesores_graph, h)             # usa sucesores/heurística elegidos
    print(f"{nombre}: {camino}  costo={costo_camino(camino)}")  # muestra camino y costo

# Resultados esperados:
# Chebyshev: ['A', 'B', 'D', 'F', 'G']  costo=8
# Octile: ['A', 'B', 'D', 'F', 'G']  costo=8
# Degree: ['A', 'B', 'D', 'H', 'G']  costo=9
# Bloc(adapt): ['A', 'B', 'D', 'F', 'G']  costo=8






# ----------------------------------------------------------
# Pruebas aplicadas al Problema de la PPT de clase (Grilla con Obstáculos)
# ----------------------------------------------------------

# Definición del tablero
rows = [
    ["A","B","C","D","E","F"],      # fila 0
    ["G","H","I","J","K","L"],      # fila 1
    ["M","N","Ñ","O","P","Q"],      # fila 2
    ["R","S","T","U","V","W"],      # fila 3
    ["X","Y","Z","A2","B2","C2"]    # fila 4
]  # grilla rotulada como en la figura

coords = {lab: (r, c) for r, row in enumerate(rows) for c, lab in enumerate(row)}  # mapa celda (r,c)

START, GOAL = "S", "L"                            # inicio y meta según la PPT
OBST_LABELS = {"I","J","O","U","V"}               # celdas bloqueadas (azules)
OBST_COORDS = {coords[x] for x in OBST_LABELS}    # obstáculos como coordenadas

# Movimientos 8-dir con costos de la PPT (recto=10, diagonal=14)
DIRS = [(-1,0,10),(1,0,10),(0,-1,10),(0,1,10),(-1,-1,14),(-1,1,14),(1,-1,14),(1,1,14)]

# Funciones específicas del problema de la grilla
# Son necesarias para la aplicación del algoritmo
def in_bounds(r, c):                               # chequea límites
    return 0 <= r < len(rows) and 0 <= c < len(rows[0])

def label_at(r, c):                                # etiqueta en (r,c)
    return rows[r][c]

def sucesores_grid(nodo):                          # sucesores válidos para la grilla
    r, c = coords[nodo]                            # coordenadas del nodo
    out = []                                       # acumulador de (vecino, costo)
    for dr, dc, w in DIRS:                         # recorre las 8 direcciones
        nr, nc = r + dr, c + dc                    # vecino candidato
        if not in_bounds(nr, nc):                  # fuera del tablero
            continue                               # descarta
        if (nr, nc) in OBST_COORDS:                # es obstáculo
            continue                               # descarta
        out.append((label_at(nr, nc), w))          # agrega vecino transitable
    return out                                     # devuelve lista de sucesores

def es_meta_grid(nodo):                            # predicado de meta
    return nodo == GOAL

grafo = {lab: [v for (v, _) in sucesores_grid(lab)]  # Grafo para heurística degree
         for lab in coords if coords[lab] not in OBST_COORDS} # lista de adyacencia (sin pesos)


h_chebyshev_steps = make_chebyshev_h(coords, GOAL)          # Chebyshev devuelve pasos, pasos mínimos con diagonales
h_chebyshev = lambda n: 10 * h_chebyshev_steps(n)           # pasa a unidades de costo
h_octile = make_octile_h(coords, GOAL, cost_straight=10.0, cost_diag=14.0) # Octile en unidades 10/14 (no usar los defaults 1/raiz2)
h_degree = make_degree_h(grafo, GOAL)                       # Degree usa el grafo de arriba
h_bloc = make_blocked_h(coords, OBST_COORDS, GOAL)          # No es admisible estricta


def costo_camino(path):                         # calcula costo real del camino
    if not path: 
        return None
    total = 0
    for a, b in zip(path, path[1:]):           # recorre pares consecutivos
        ra, ca = coords[a]; rb, cb = coords[b] # coords de a y b
        diag = (ra != rb and ca != cb)         # detecta diagonal
        total += 14 if diag else 10            # suma costo correspondiente
    return total

tests = {                                                 # set de heurísticas a probar
    "Chebyshev": h_chebyshev,                             # chebyshev escalada a costo 10
    "Octile":  h_octile,                                  # octile con 10/14
    "Degree":       h_degree,                             # diferencia de grados
    "Bloc(adapt)":  h_bloc                                # bloc adaptada (sin obstáculos)
}

for nombre, h in tests.items():                                    # recorre todas las h
    camino = ida_estrella(START, es_meta_grid, sucesores_grid, h)  # ejecuta IDA*
    print(f"{nombre}: {camino}  costo={costo_camino(camino)}")     # muestra resultado

# Resultados esperados:
# Chebyshev(10): ['S', 'T', 'A2', 'B2', 'W', 'Q', 'L']  costo=68
# Octile(10/14): ['S', 'T', 'A2', 'B2', 'W', 'Q', 'L']  costo=68
# Degree: ['S', 'T', 'A2', 'B2', 'W', 'Q', 'L']  costo=68
# Bloc: ['S', 'N', 'H', 'C', 'D', 'K', 'L']  costo=68








# ---------------------------------------------------------------------------
# Pruebas aplicadas al grafo de la PPT (Dibujo de un Grafo discutido en Clase)
# ---------------------------------------------------------------------------
START, GOAL = 'P', 'S'                                   
G2 = {
    'P': [('A', 4), ('C', 4), ('R', 4)],                 # salientes de P
    'A': [('M', 3)],                                     # A -> M (3)
    'M': [('U', 5), ('L', 2)],                           # M -> U (5), M -> L (2)
    'L': [('N', 5)],                                     # L -> N (5)
    'C': [('R', 2), ('U', 3), ('M', 6)],                 # C -> R (2), U (3), M (6)
    'R': [('E', 5)],                                     # R -> E (5)
    'E': [('U', 5), ('S', 1)],                           # E -> U (5), S (1)
    'U': [('S', 4), ('N', 5)],                           # U -> S (4), N (5)
    'N': [('S', 6)],                                     # N -> S (6)
    'S': []                                              # S sin salientes
}

def sucesores_case(n):                                    # sucesores(n)
    return G2.get(n, [])                                  # lista de (hijo, costo)
def es_meta_case(n):                                      # es_meta(n)
    return n == GOAL                                      # true si llegamos a S

H = {
    'P': 10, 'A': 11, 'M': 9, 'L': 9, 'C': 6, 'R': 5,
    'E': 1,  'U': 4,  'N': 6, 'S': 0
} # Heurística de la imagen (número entre paréntesis en cada nodo)        
h = lambda n: H[n]                                       # h(n) desde tabla de la lámina
                        
def costo_camino_case(path):
    if not path:                                         # si no hay camino
        return None                                      # no hay costo
    total = 0                                            # acumulador
    for a, b in zip(path, path[1:]):                     # pares consecutivos
        w = next(w for v, w in G2[a] if v == b)          # busca peso (a->b)
        total += w                                       # suma peso
    return total                                         # costo total

camino = ida_estrella(START, es_meta_case, sucesores_case, h)  # corre IDA* con h-tabla
print("IDA*:", camino, "costo=", costo_camino_case(camino))  # muestra resultado

# Resultado esperado:
# IDA*: ['P', 'R', 'E', 'S'] costo= 10
# Lo cual concuerda con lo discutido en clase.
