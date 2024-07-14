import numpy as np

def nelder_mead(f, x0, gamma=2, beta=0.5, epsilon=1e-5, max_iter=1000):
    N = len(x0)
    # Crear el simplex inicial
    simplex = [x0]
    for i in range(N):
        x = np.copy(x0)
        x[i] = x[i] + (x[i] + 1)
        simplex.append(x)
    simplex = np.array(simplex)

    for iteration in range(max_iter):
        # Ordenar el simplex según los valores de la función objetivo
        simplex = sorted(simplex, key=lambda x: f(x))
        xh = simplex[-1]   # El peor punto
        xl = simplex[0]    # El mejor punto
        xg = simplex[-2]   # El siguiente peor punto

        # Calcular el centroide
        xc = np.mean(simplex[:-1], axis=0)

        # Reflejar el punto xh
        xr = 2 * xc - xh
        if f(xr) < f(xl):
            # Expansión
            xe = (1 + gamma) * xc - gamma * xh
            if f(xe) < f(xr):
                xnew = xe
            else:
                xnew = xr
        elif f(xr) < f(xg):
            xnew = xr
        else:
            if f(xr) < f(xh):
                xh = xr
            # Contracción
            if f(xr) < f(xh):
                xc = xc - beta * (xc - xr)
            else:
                xc = xc - beta * (xc - xh)
            xnew = xc

        # Reemplazar el peor punto con el nuevo punto
        simplex[-1] = xnew

        # Comprobar la condición de terminación
        f_values = np.array([f(x) for x in simplex])
        if np.sqrt(np.sum((f_values - np.mean(f_values)) ** 2) / (N + 1)) <= epsilon:
            break

    return simplex[0]  # Devuelve el mejor punto encontrado

# Funciones de prueba

def sphere_function(x):
    return np.sum(x**2)

def himmelblau_function(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def rastrigin_function(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rosenbrock_function(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

# Pruebas

# Sphere function
x0 = np.array([-1.0, 1.5])
resultado_sphere = nelder_mead(sphere_function, x0)
print("Sphere function resultado:", resultado_sphere)

# Himmelblau's function
x0 = np.array([0.0, 0.0])
resultado_himmelblau = nelder_mead(himmelblau_function, x0)
print("Himmelblau's function resultado:", resultado_himmelblau)

# Rastrigin function
x0 = np.array([-2.0, -2.0, -2.0])
resultado_rastrigin = nelder_mead(rastrigin_function, x0)
print("Rastrigin function resultado:", resultado_rastrigin)

# Rosenbrock function
x0 = np.array([2.0, 1.5, 3.0, -1.5, -2.0])
resultado_rosenbrock = nelder_mead(rosenbrock_function, x0)
print("Rosenbrock function resultado:", resultado_rosenbrock)
