#Metodos de gradiente
from .cauchy import regla_eliminacion, w_to_x, busquedaDorada, gradiente, cauchy
from diferenciacentralgradiente import f, primera_derivada, segunda_derivada
from .gradiente_conjugado import gradiente, regla_eliminacion, w_to_x, busquedaDorada, gradiente_conjugado
from .newton import regla_eliminacion, w_to_x, busquedaDorada, gradiente, hessian_matrix, newton