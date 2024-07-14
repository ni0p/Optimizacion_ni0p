import numpy as np
import matplotlib.pyplot as plt

# Funciones ya definidas
def rastrigin(x, y, A=10, n=2):
    return A * n + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))

def ackley(x, y, a=20, b=0.2, c=2*np.pi):
    sum_sq_term = -b * np.sqrt(0.5 * (x**2 + y**2))
    cos_term = 0.5 * (np.cos(c * x) + np.cos(c * y))
    return -a * np.exp(sum_sq_term) - np.exp(cos_term) + a + np.exp(1)

def sphere(x, y):
    return x**2 + y**2

def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

def beale(x, y):
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

def goldstein_price(x, y):
    part1 = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2))
    part2 = (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return part1 * part2

def booth(x, y):
    return (x + 2*y - 7)**2 + (2*x + y - 5)**2

def bukin_n6(x, y):
    return 100 * np.sqrt(np.abs(y - 0.01*x**2)) + 0.01 * np.abs(x + 10)

def matyas(x, y):
    return 0.26 * (x**2 + y**2) - 0.48 * x * y

def levi_n13(x, y):
    return np.sin(3 * np.pi * x)**2 + ((x - 1)**2) * (1 + np.sin(3 * np.pi * y)**2) + ((y - 1)**2) * (1 + np.sin(2 * np.pi * y)**2)

def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def three_hump_camel(x, y):
    return 2*x**2 - 1.05*x**4 + (x**6) / 6 + x*y + y**2

# Nuevas funciones
def easom(x, y):
    return -np.cos(x) * np.cos(y) * np.exp(-((x - np.pi)**2 + (y - np.pi)**2))

def cross_in_tray(x, y):
    return -0.0001 * (np.abs(np.sin(x) * np.sin(y) * np.exp(np.abs(100 - np.sqrt(x**2 + y**2) / np.pi))) + 1) ** 0.1

def eggholder(x, y):
    return -(y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) - x * np.sin(np.sqrt(np.abs(x - (y + 47))))

def holder_table(x, y):
    return -np.abs(np.sin(x) * np.cos(y) * np.exp(np.abs(1 - np.sqrt(x**2 + y**2) / np.pi)))

def mccormick(x, y):
    return np.sin(x + y) + (x - y)**2 - 1.5 * x + 2.5 * y + 1

def schaffer_n2(x, y):
    return 0.5 + (np.sin(x**2 - y**2)**2 - 0.5) / (1 + 0.001 * (x**2 + y**2))**2

def schaffer_n4(x, y):
    return 0.5 + (np.cos(np.sin(np.abs(x**2 - y**2))))**2 - 0.5 / (1 + 0.001 * (x**2 + y**2))**2

def styblinski_tang(x, y):
    return (x**4 - 16 * x**2 + 5 * x + y**4 - 16 * y**2 + 5 * y) / 2

# Cálculo de Z 
def calculo_z(func, x, y):
    z = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i, j] = func(x[i, j], y[i, j])
    return z

# Arreglos con los límites generados para cada función
x_r = np.linspace(-5.12, 5.12, 100)
y_r = np.linspace(-5.12, 5.12, 100)
x_rm, y_rm = np.meshgrid(x_r, y_r)

x_a = np.linspace(-5, 5, 100)
y_a = np.linspace(-5, 5, 100)
x_am, y_am = np.meshgrid(x_a, y_a)

x_s = np.linspace(-5, 5, 100)
y_s = np.linspace(-5, 5, 100)
x_sm, y_sm = np.meshgrid(x_s, y_s)

x_rs = np.linspace(-2, 2, 100)
y_rs = np.linspace(-2, 2, 100)
x_rsm, y_rsm = np.meshgrid(x_rs, y_rs)

x_b = np.linspace(-4.5, 4.5, 100)
y_b = np.linspace(-4.5, 4.5, 100)
x_bm, y_bm = np.meshgrid(x_b, y_b)

x_g = np.linspace(-2, 2, 100)
y_g = np.linspace(-2, 2, 100)
x_gm, y_gm = np.meshgrid(x_g, y_g)

x_bo = np.linspace(-10, 10, 100)
y_bo = np.linspace(-10, 10, 100)
x_bom, y_bom = np.meshgrid(x_bo, y_bo)

x_bn = np.linspace(-15, -5, 100)
y_bn = np.linspace(-3, 3, 100)
x_bnm, y_bnm = np.meshgrid(x_bn, y_bn)

x_m = np.linspace(-10, 10, 100)
y_m = np.linspace(-10, 10, 100)
x_mm, y_mm = np.meshgrid(x_m, y_m)

x_l = np.linspace(-10, 10, 100)
y_l = np.linspace(-10, 10, 100)
x_lm, y_lm = np.meshgrid(x_l, y_l)

x_h = np.linspace(-5, 5, 100)
y_h = np.linspace(-5, 5, 100)
x_hm, y_hm = np.meshgrid(x_h, y_h)

x_th = np.linspace(-5, 5, 100)
y_th = np.linspace(-5, 5, 100)
x_thm, y_thm = np.meshgrid(x_th, y_th)

x_e = np.linspace(-100, 100, 100)
y_e = np.linspace(-100, 100, 100)
x_em, y_em = np.meshgrid(x_e, y_e)

x_c = np.linspace(-10, 10, 100)
y_c = np.linspace(-10, 10, 100)
x_cm, y_cm = np.meshgrid(x_c, y_c)

x_eg = np.linspace(-512, 512, 100)
y_eg = np.linspace(-512, 512, 100)
x_egm, y_egm = np.meshgrid(x_eg, y_eg)

x_ho = np.linspace(-10, 10, 100)
y_ho = np.linspace(-10, 10, 100)
x_hom, y_hom = np.meshgrid(x_ho, y_ho)

x_mc = np.linspace(-1.5, 4, 100)
y_mc = np.linspace(-3, 4, 100)
x_mcm, y_mcm = np.meshgrid(x_mc, y_mc)

x_s2 = np.linspace(-100, 100, 100)
y_s2 = np.linspace(-100, 100, 100)
x_s2m, y_s2m = np.meshgrid(x_s2, y_s2)

x_schaffer4 = np.linspace(-100, 100, 400)
y_schaffer4 = np.linspace(-100, 100, 400)
x_s4m, y_s4m = np.meshgrid(x_schaffer4, y_schaffer4)

x_styblinski = np.linspace(-5, 5, 400)
y_styblinski = np.linspace(-5, 5, 400)
x_stm, y_stm = np.meshgrid(x_styblinski, y_styblinski)

# Calcular Z para cada función
Z_rastrigin = calculo_z(rastrigin, x_rm, y_rm)
Z_ackley = calculo_z(ackley, x_am, y_am)
Z_sphere = calculo_z(sphere, x_sm, y_sm)
Z_rosenbrock = calculo_z(rosenbrock, x_rsm, y_rsm)
Z_beale = calculo_z(beale, x_bm, y_bm)
Z_goldstein = calculo_z(goldstein_price, x_gm, y_gm)
Z_booth = calculo_z(booth, x_bom, y_bom)
Z_bukin_n6 = calculo_z(bukin_n6, x_bnm, y_bnm)
Z_matyas = calculo_z(matyas, x_mm, y_mm)
Z_levi_n13 = calculo_z(levi_n13, x_lm, y_lm)
Z_himmelblau = calculo_z(himmelblau, x_hm, y_hm)
Z_three_hump_camel = calculo_z(three_hump_camel, x_thm, y_thm)
Z_easom = calculo_z(easom, x_em, y_em)
Z_cross_in_tray = calculo_z(cross_in_tray, x_cm, y_cm)
Z_eggholder = calculo_z(eggholder, x_egm, y_egm)
Z_holder_table = calculo_z(holder_table, x_hom, y_hom)
Z_mccormick = calculo_z(mccormick, x_mcm, y_mcm)
Z_schaffer_n2 = calculo_z(schaffer_n2, x_s2m, y_s2m)
Z_schaffer_n4 = schaffer_n4(x_s4m, y_s4m)
Z_styblinski_tang = styblinski_tang(x_stm, y_stm)

# Crear subplots
fig, axs = plt.subplots(7, 3, figsize=(15, 20))

# Graficar la función Rastrigin
cp = axs[0, 0].contourf(x_rm, y_rm, Z_rastrigin, cmap='viridis')
fig.colorbar(cp, ax=axs[0, 0])
axs[0, 0].set_title('Función de Rastrigin')
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('Y')

# Graficar la función Ackley
cp = axs[0, 1].contourf(x_am, y_am, Z_ackley, cmap='viridis')
fig.colorbar(cp, ax=axs[0, 1])
axs[0, 1].set_title('Función de Ackley')
axs[0, 1].set_xlabel('X')
axs[0, 1].set_ylabel('Y')

# Graficar la función Sphere
cp = axs[0, 2].contourf(x_sm, y_sm, Z_sphere, cmap='viridis')
fig.colorbar(cp, ax=axs[0, 2])
axs[0, 2].set_title('Función de Sphere')
axs[0, 2].set_xlabel('X')
axs[0, 2].set_ylabel('Y')

# Graficar la función Rosenbrock
cp = axs[1, 0].contourf(x_rsm, y_rsm, Z_rosenbrock, cmap='viridis')
fig.colorbar(cp, ax=axs[1, 0])
axs[1, 0].set_title('Función de Rosenbrock')
axs[1, 0].set_xlabel('X')
axs[1, 0].set_ylabel('Y')

# Graficar la función Beale
cp = axs[1, 1].contourf(x_bm, y_bm, Z_beale, cmap='viridis')
fig.colorbar(cp, ax=axs[1, 1])
axs[1, 1].set_title('Función de Beale')
axs[1, 1].set_xlabel('X')
axs[1, 1].set_ylabel('Y')

# Graficar la función Goldstein-Price
cp = axs[1, 2].contourf(x_gm, y_gm, Z_goldstein, cmap='viridis')
fig.colorbar(cp, ax=axs[1, 2])
axs[1, 2].set_title('Función de Goldstein-Price')
axs[1, 2].set_xlabel('X')
axs[1, 2].set_ylabel('Y')

# Graficar la función Booth
cp = axs[2, 0].contourf(x_bom, y_bom, Z_booth, cmap='viridis')
fig.colorbar(cp, ax=axs[2, 0])
axs[2, 0].set_title('Función de Booth')
axs[2, 0].set_xlabel('X')
axs[2, 0].set_ylabel('Y')

# Graficar la función Bukin N.6
cp = axs[2, 1].contourf(x_bnm, y_bnm, Z_bukin_n6, cmap='viridis')
fig.colorbar(cp, ax=axs[2, 1])
axs[2, 1].set_title('Función de Bukin N.6')
axs[2, 1].set_xlabel('X')
axs[2, 1].set_ylabel('Y')

# Graficar la función Matyas
cp = axs[2, 2].contourf(x_mm, y_mm, Z_matyas, cmap='viridis')
fig.colorbar(cp, ax=axs[2, 2])
axs[2, 2].set_title('Función de Matyas')
axs[2, 2].set_xlabel('X')
axs[2, 2].set_ylabel('Y')

# Graficar la función Lévi N.13
cp = axs[3, 0].contourf(x_lm, y_lm, Z_levi_n13, cmap='viridis')
fig.colorbar(cp, ax=axs[3, 0])
axs[3, 0].set_title('Función de Lévi N.13')
axs[3, 0].set_xlabel('X')
axs[3, 0].set_ylabel('Y')

# Graficar la función Himmelblau
cp = axs[3, 1].contourf(x_hm, y_hm, Z_himmelblau, cmap='viridis')
fig.colorbar(cp, ax=axs[3, 1])
axs[3, 1].set_title('Función de Himmelblau')
axs[3, 1].set_xlabel('X')
axs[3, 1].set_ylabel('Y')

# Graficar la función Three-hump Camel
cp = axs[3, 2].contourf(x_thm, y_thm, Z_three_hump_camel, cmap='viridis')
fig.colorbar(cp, ax=axs[3, 2])
axs[3, 2].set_title('Función de Three-hump Camel')
axs[3, 2].set_xlabel('X')
axs[3, 2].set_ylabel('Y')

# Graficar la función Easom
cp = axs[4, 0].contourf(x_em, y_em, Z_easom, cmap='viridis')
fig.colorbar(cp, ax=axs[4, 0])
axs[4, 0].set_title('Función de Easom')
axs[4, 0].set_xlabel('X')
axs[4, 0].set_ylabel('Y')

# Graficar la función Cross-in-tray
cp = axs[4, 1].contourf(x_cm, y_cm, Z_cross_in_tray, cmap='viridis')
fig.colorbar(cp, ax=axs[4, 1])
axs[4, 1].set_title('Función de Cross-in-tray')
axs[4, 1].set_xlabel('X')
axs[4, 1].set_ylabel('Y')

# Graficar la función Eggholder
cp = axs[4, 2].contourf(x_egm, y_egm, Z_eggholder, cmap='viridis')
fig.colorbar(cp, ax=axs[4, 2])
axs[4, 2].set_title('Función de Eggholder')
axs[4, 2].set_xlabel('X')
axs[4, 2].set_ylabel('Y')

# Graficar la función Hölder Table
cp = axs[5, 0].contourf(x_hom, y_hom, Z_holder_table, cmap='viridis')
fig.colorbar(cp, ax=axs[5, 0])
axs[5, 0].set_title('Función de Hölder Table')
axs[5, 0].set_xlabel('X')
axs[5, 0].set_ylabel('Y')

# Graficar la función McCormick
cp = axs[5, 1].contourf(x_mcm, y_mcm, Z_mccormick, cmap='viridis')
fig.colorbar(cp, ax=axs[5, 1])
axs[5, 1].set_title('Función de McCormick')
axs[5, 1].set_xlabel('X')
axs[5, 1].set_ylabel('Y')

# Graficar la función Schaffer N.2
cp = axs[5, 2].contourf(x_s2m, y_s2m, Z_schaffer_n2, cmap='viridis')
fig.colorbar(cp, ax=axs[5, 2])
axs[5, 2].set_title('Función de Schaffer N.2')
axs[5, 2].set_xlabel('X')
axs[5, 2].set_ylabel('Y')

# Graficar la función Schaffer N.4
cp = axs[6, 0].contourf(x_s4m, y_s4m, Z_schaffer_n4, cmap='viridis')
fig.colorbar(cp, ax=axs[6, 0])
axs[6, 0].set_title('Función de Schaffer N.4')
axs[6, 0].set_xlabel('X')
axs[6, 0].set_ylabel('Y')

# Graficar la función Styblinski-Tang
cp = axs[6, 1].contourf(x_stm, y_stm, Z_styblinski_tang, cmap='viridis')
fig.colorbar(cp, ax=axs[6, 1])
axs[6, 1].set_title('Función de Styblinski-Tang')
axs[6, 1].set_xlabel('X')
axs[6, 1].set_ylabel('Y')


# Ajustar el layout
plt.tight_layout()
plt.show()
