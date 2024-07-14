import numpy as np
import math 

def gradiente(f, x, deltaX=0.00001):
    grad = []
    for i in range(0, len(x)):
        xp = x.copy()
        xn = x.copy()
        xp[i] = xp[i]+deltaX
        xn[i] = xn[i]-deltaX
        print(x, xp, xn, deltaX)
        grad.append((f(xp)-f(xn))/(2*deltaX))
    return np.array(grad)

def regla_eliminacion(x1, x2, fx1, fx2, a, b)->tuple[float, float]:
    if fx1 > fx2:
        return x1, b

    if fx1 < fx2:
        return a, x2
    
    return x1, x2

def w_to_x(w:float, a, b) -> float:
    return w * (b-a) + a

def busquedaDorada(funcion, epsilon:float, a:float=None, b:float=None) -> float:
    PHI = ( 1 + math.sqrt(5) ) / 2 - 1
    aw, bw = 0, 1
    Lw = 1
    k = 1

    while Lw > epsilon:
        w2 = aw + PHI*Lw
        w1 = bw - PHI*Lw
        aw, bw = regla_eliminacion(w1, w2, funcion(w_to_x(w1, a, b)), 
                                    funcion(w_to_x(w2, a, b)), aw, bw)
        k+=1
        Lw = bw - aw

    return (w_to_x(aw, a, b)+w_to_x(bw, a, b))/2


def gradiente_conjugado(funcion, x0, epsilon1, epsilon2, M):
    terminar = False
    xk = x0
    k=0
    s = -1*(gradiente(funcion, xk))
    while not terminar:
        
        
        if np.linalg.norm(s) < epsilon1 or k >= M:
            terminar = True
        else:
            
            def alpha_funcion(alpha):
                return funcion(xk + alpha*s)
            
            alpha = busquedaDorada(alpha_funcion, epsilon=epsilon2, a=0.0, b=1.0)
            x_k1 = xk + alpha*s

            print("Numerito", (np.linalg.norm(gradiente(funcion, x_k1)) ** 2 / np.linalg.norm(gradiente(funcion, xk)) ** 2)*s)
            print("Primera parte",-1*(gradiente(funcion, x_k1)))
            #URGENTE AHORITA MISMO: :D Formula paso 4 D:...Debería funcionar
            s = -1*(gradiente(funcion, x_k1)) + (np.linalg.norm(gradiente(funcion, x_k1)) ** 2 / np.linalg.norm(gradiente(funcion, xk)) ** 2) * s
  
            #print(xk, alpha, s, x_k1)
            
            if np.linalg.norm(x_k1-xk)/(np.linalg.norm(xk)+0.00001) <= epsilon2 :
                terminar = True
            else:
                k = k + 1
                xk = x_k1
    return xk

himmenblau = lambda x: (((x[0]**2)+x[1]-11)**2) + ((x[0]+(x[1]**2)-7)**2)
print(gradiente_conjugado(himmenblau, np.array([1.0,1.0]), 0.001, 0.001, 100))
#cercano -> 3,2
