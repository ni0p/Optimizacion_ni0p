import numpy as np
import matplotlib.pyplot as plt

# Funciones de prueba

def rosenbrock_constrained(x, y):
    f = (1 - x)**2 + 100 * (y - x**2)**2
    constraint1 = (x - 1)**3 - y + 1
    constraint2 = x + y - 2
    return f if (constraint1 <= 0) and (constraint2 <= 0) else np.inf

def rosenbrock_disk(x, y):
    f = (1 - x)**2 + 100 * (y - x**2)**2
    constraint = x**2 + y**2 - 2
    return f if (constraint <= 0) else np.inf

def mishra_bird(x, y):
    f = np.sin(y) * np.exp((1 - np.cos(x))**2) + np.cos(x) * np.exp((1 - np.sin(y))**2) + (x - y)**2
    constraint = (x + 5)**2 + (y + 5)**2 - 25
    return f if (constraint < 0) else np.inf

def townsend(x, y):
    f = -np.cos((x - 0.1) * y)**2 - x * np.sin(3 * x + y)
    t = np.arctan2(x, y)
    constraint = x**2 + y**2 - (2 * np.cos(t) - 0.5 * np.cos(2 * t) - 0.25 * np.cos(3 * t) - 0.125 * np.cos(4 * t))**2 - (2 * np.sin(t))**2
    return f if (constraint < 0) else np.inf

def gomez_levy(x, y):
    f = 4 * x**2 - 2.1 * x**4 + (1/3) * x**6 + x * y - 4 * y**2 + 4 * y**4
    constraint = -np.sin(4 * np.pi * x) + 2 * np.sin(2 * np.pi * y)**2 - 1.5
    return f if (constraint <= 0) else np.inf

def simionescu(x, y):
    f = 0.1 * x * y
    r_T = 1
    r_S = 0.2
    n = 8
    constraint = (r_T + r_S * np.cos(n * np.arctan(x / y)))**2
    return f if (x**2 + y**2 <= constraint) else np.inf

# Vectorizar las funciones
rosenbrock_constrained_vec = np.vectorize(rosenbrock_constrained)
rosenbrock_disk_vec = np.vectorize(rosenbrock_disk)
mishra_bird_vec = np.vectorize(mishra_bird)
townsend_vec = np.vectorize(townsend)
gomez_levy_vec = np.vectorize(gomez_levy)
simionescu_vec = np.vectorize(simionescu)

# Rangos de las variables
x = np.linspace(-2.5, 2.5, 500)
y = np.linspace(-2.5, 2.5, 500)
X, Y = np.meshgrid(x, y)

# Evaluar las funciones en la malla
Z1 = rosenbrock_constrained_vec(X, Y)
Z2 = rosenbrock_disk_vec(X, Y)
Z3 = mishra_bird_vec(X, Y)
Z4 = townsend_vec(X, Y)
Z5 = gomez_levy_vec(X, Y)
Z6 = simionescu_vec(X, Y)

# Graficar las funciones
plt.figure(figsize=(18, 12))

plt.subplot(231)
plt.contourf(X, Y, Z1, levels=50, cmap='viridis')
plt.title('Rosenbrock constrained')
plt.colorbar()

plt.subplot(232)
plt.contourf(X, Y, Z2, levels=50, cmap='plasma')
plt.title('Rosenbrock disk constrained')
plt.colorbar()

plt.subplot(233)
plt.contourf(X, Y, Z3, levels=50, cmap='inferno')
plt.title("Mishra's Bird constrained")
plt.colorbar()

plt.subplot(234)
plt.contourf(X, Y, Z4, levels=50, cmap='magma')
plt.title('Townsend function constrained')
plt.colorbar()

plt.subplot(235)
plt.contourf(X, Y, Z5, levels=50, cmap='cividis')
plt.title('Gomez and Levy function constrained')
plt.colorbar()

plt.subplot(236)
plt.contourf(X, Y, Z6, levels=50, cmap='twilight')
plt.title('Simionescu function constrained')
plt.colorbar()

plt.tight_layout()
plt.show()
