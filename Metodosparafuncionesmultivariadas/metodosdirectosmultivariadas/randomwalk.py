import numpy as np

def random_walk(f, terminar, x0, generacion_aleatoria):
    x_mejor = x0
    x_actual = x0

    while not terminar(x_actual):
        x_siguiente = generacion_aleatoria(x_actual)
        if f(x_siguiente) < f(x_mejor):
            x_mejor = x_siguiente
        x_actual = x_siguiente
    
    return x_mejor

# Funciones de prueba
def sphere_function(x):
    x = np.array(x)
    return np.sum(x**2)

def himmelblau_function(x):
    x = np.array(x)
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def rastrigin_function(x):
    x = np.array(x)
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rosenbrock_function(x):
    x = np.array(x)
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

# Criterio de terminación
def criterio_terminacion(x, max_iter=100):
    criterio_terminacion.iteraciones += 1
    return criterio_terminacion.iteraciones >= max_iter

# Inicialización del contador de iteraciones
criterio_terminacion.iteraciones = 0

# Generación de pasos aleatorios
def generacion_aleatoria(x, mu=0, sigma=0.5):
    return x + np.random.normal(mu, sigma, size=len(x))

# Parámetros iniciales
functions = [sphere_function, himmelblau_function, rastrigin_function, rosenbrock_function]
initial_points = [
    [5.0, 5.0],  # Punto inicial para sphere_function
    [0.0, 0.0],  # Punto inicial para himmelblau_function
    [5.12, 5.12],  # Punto inicial para rastrigin_function
    [1.2, 1.2]   # Punto inicial para rosenbrock_function
]
delta = 0.5

# Ejecución del algoritmo para cada función de prueba
mejores_soluciones = []
for f, x0 in zip(functions, initial_points):
    # Reiniciar el contador de iteraciones para cada ejecución
    criterio_terminacion.iteraciones = 0
    mejor_solucion = random_walk(f, criterio_terminacion, x0, generacion_aleatoria)
    mejores_soluciones.append(mejor_solucion)

for i, mejor_solucion in enumerate(mejores_soluciones):
    print(f"Mejor solución encontrada para la función {functions[i].__name__}: {mejor_solucion}")
