#include <omp.h>
#include <vector>
#include <atomic>
#include <random>
#include <iostream>
#include <algorithm>

const int N = 500000000;  // 10M nodos
const int AVG_DEGREE = 50;

int main() {
    // ============================================================
    // FASE 1: CONSTRUCCIÓN DEL GRAFO
    // ============================================================
    
    // Reservar memoria para el grafo (vector de vectores)
    std::vector<std::vector<int>> graph(N);
    
    // Generador de números aleatorios con semilla fija (reproducibilidad)
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(0, N-1);
    
    // Construir grafo aleatorio: cada nodo tiene AVG_DEGREE vecinos
    std::cout << "Construyendo grafo..." << std::endl;
    for(int i = 0; i < N; i++) {
        graph[i].reserve(AVG_DEGREE);  // Pre-reservar espacio (eficiencia)
        for(int j = 0; j < AVG_DEGREE; j++) {
            graph[i].push_back(dis(gen));  // Agregar vecino aleatorio
        }
    }
    
    
    // ============================================================
    // FASE 2: INICIALIZACIÓN DE ESTRUCTURAS BFS
    // ============================================================
    
    // Vector de distancias: -1 = no visitado, >=0 = nivel BFS
    std::vector<int> distance(N, -1);
    distance[0] = 0;  // El nodo 0 es la raíz (distancia 0)
    
    // Vector atómico para marcar visitados sin race conditions
    // std::atomic<bool> no funciona en vector, usamos vector de atomics manual
    std::vector<std::atomic<bool>> visited(N);
    for(int i = 0; i < N; i++) {
        visited[i].store(false, std::memory_order_relaxed);  // Inicializar a false
    }
    visited[0].store(true, std::memory_order_relaxed);  // Marcar raíz como visitada
    
    // Frontier actual (nodos en el nivel actual del BFS)
    std::vector<int> frontier = {0};
    
    
    // ============================================================
    // FASE 3: CONFIGURACIÓN OPENMP
    // ============================================================
    
    omp_set_num_threads(128);  // Establecer número de threads
    
    // Variables para estadísticas
    int level = 0;
    long long total_edges_explored = 0;
    
    std::cout << "Iniciando BFS con " << omp_get_max_threads() << " threads..." << std::endl;
    double start = omp_get_wtime();  // Tiempo inicial
    
    
    // ============================================================
    // FASE 4: BFS PARALELO (BUCLE PRINCIPAL)
    // ============================================================
    
    while(!frontier.empty()) {  // Mientras haya nodos por explorar
        
        // Vector para el siguiente nivel (next_frontier)
        // Tamaño estimado: peor caso es frontier_size * AVG_DEGREE
        std::vector<int> next_frontier;
        next_frontier.reserve(frontier.size() * AVG_DEGREE / 10);
        
        // Contador atómico para saber dónde insertar en next_frontier
        std::atomic<size_t> next_frontier_size(0);
        
        // Buffer temporal grande para evitar sincronización
        // Cada thread escribe aquí sin locks
        std::vector<int> temp_buffer(frontier.size() * AVG_DEGREE);
        
        
        // --------------------------------------------------------
        // REGIÓN PARALELA
        // --------------------------------------------------------
        #pragma omp parallel
        {
            // Cada thread tiene su propio vector local (sin sincronización)
            std::vector<int> local_frontier;
            local_frontier.reserve(1000);  // Capacidad inicial
            
            // --------------------------------------------------------
            // PARALELIZAR EXPLORACIÓN DE FRONTIER
            // schedule(dynamic, 64): distribución dinámica en chunks de 64
            // Bueno para balance de carga cuando nodos tienen grados variables
            // --------------------------------------------------------
            #pragma omp for schedule(dynamic, 64) reduction(+:total_edges_explored)
            for(size_t i = 0; i < frontier.size(); i++) {
                int node = frontier[i];  // Nodo actual a explorar
                
                // Explorar todos los vecinos del nodo
                for(int neighbor : graph[node]) {
                    total_edges_explored++;  // Contar arista explorada
                    
                    // ------------------------------------------------
                    // INTENTO ATÓMICO DE MARCAR COMO VISITADO
                    // compare_exchange_strong: compara y cambia atómicamente
                    // Si visited[neighbor] es false, lo cambia a true y retorna true
                    // Si ya era true, retorna false (otro thread lo visitó primero)
                    // ------------------------------------------------
                    bool expected = false;
                    if(visited[neighbor].compare_exchange_strong(
                        expected,                        // Valor esperado (false)
                        true,                           // Nuevo valor (true)
                        std::memory_order_release,      // Memoria order para escritura
                        std::memory_order_relaxed)) {   // Memoria order si falla
                        
                        // Solo este thread logró marcar el nodo
                        distance[neighbor] = level + 1;  // Asignar distancia
                        local_frontier.push_back(neighbor);  // Agregar a frontier local
                    }
                    // Si compare_exchange retornó false, otro thread ya procesó este nodo
                }
            }
            
            // --------------------------------------------------------
            // FUSIÓN DE FRONTIERS LOCALES EN GLOBAL
            // Sección crítica solo para fusionar vectores locales
            // (mucho más eficiente que critical por cada nodo)
            // --------------------------------------------------------
            #pragma omp critical
            {
                next_frontier.insert(next_frontier.end(), 
                                    local_frontier.begin(), 
                                    local_frontier.end());
            }
        }
        // FIN REGIÓN PARALELA
        
        
        // --------------------------------------------------------
        // ACTUALIZAR PARA SIGUIENTE ITERACIÓN
        // --------------------------------------------------------
        frontier = std::move(next_frontier);  // move: evita copiar, transfiere ownership
        level++;  // Incrementar nivel BFS
        
        // Imprimir progreso cada 5 niveles
        if(level % 5 == 0) {
            std::cout << "Nivel " << level << ": " << frontier.size() 
                      << " nodos en frontier" << std::endl;
        }
    }
    
    
    // ============================================================
    // FASE 5: RESULTADOS Y ESTADÍSTICAS
    // ============================================================
    
    double end = omp_get_wtime();
    double elapsed = end - start;
    
    // Contar nodos alcanzados
    long long nodes_reached = 0;
    #pragma omp parallel for reduction(+:nodes_reached)
    for(int i = 0; i < N; i++) {
        if(distance[i] != -1) nodes_reached++;
    }
    
    std::cout << "\n========== RESULTADOS ==========" << std::endl;
    std::cout << "Tiempo total: " << elapsed << " segundos" << std::endl;
    std::cout << "Niveles BFS: " << level << std::endl;
    std::cout << "Nodos alcanzados: " << nodes_reached << " / " << N 
              << " (" << (100.0*nodes_reached/N) << "%)" << std::endl;
    std::cout << "Aristas exploradas: " << total_edges_explored << std::endl;
    std::cout << "Throughput: " << (total_edges_explored/elapsed/1e6) 
              << " M aristas/seg" << std::endl;
    
    // Calcular distribución de distancias
    std::vector<int> distance_histogram(level + 1, 0);
    for(int i = 0; i < N; i++) {
        if(distance[i] != -1) {
            distance_histogram[distance[i]]++;
        }
    }
    
    std::cout << "\nDistribución por nivel:" << std::endl;
    for(int i = 0; i <= std::min(10, level); i++) {
        std::cout << "  Nivel " << i << ": " << distance_histogram[i] << " nodos" << std::endl;
    }
    
    return 0;
}
