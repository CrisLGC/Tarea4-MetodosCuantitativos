# Cristian Guevara 31.567.525. seccion: 208C1
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.optimize import linprog

class PlanoCorte:
    def __init__(self):
        self.restricciones = []
        self.funcion_objetivo = None
        self.variables = None
        self.soluciones = []
        
    def ingresar_datos(self):
        print("=== Método de Planos de Corte ===")
        num_vars = int(input("Ingrese el número de variables: "))
        self.variables = [f'x{i+1}' for i in range(num_vars)]
        
        print("\nIngrese la función objetivo (maximizar o minimizar):")
        coeficientes = []
        for i in range(num_vars):
            coef = float(input(f"Coeficiente para {self.variables[i]}: "))
            coeficientes.append(coef)
        tipo = input("¿Maximizar (max) o Minimizar (min)? ").lower()
        self.funcion_objetivo = {'coeficientes': coeficientes, 'tipo': tipo}
        
        num_restricciones = int(input("\nIngrese el número de restricciones: "))
        for i in range(num_restricciones):
            print(f"\nRestricción {i+1}:")
            coef_restriccion = []
            for j in range(num_vars):
                coef = float(input(f"Coeficiente para {self.variables[j]}: "))
                coef_restriccion.append(coef)
            desigualdad = input("Desigualdad (<=, >=, =): ")
            termino_indep = float(input("Término independiente: "))
            self.restricciones.append({
                'coeficientes': coef_restriccion,
                'desigualdad': desigualdad,
                'termino_indep': termino_indep
            })
    
    def resolver_relajacion_lineal(self):
        print("\nResolviendo relajación lineal con scipy.optimize.linprog...")
        num_vars = len(self.variables)
        c = np.array(self.funcion_objetivo['coeficientes'], dtype=float)
        if self.funcion_objetivo['tipo'] == 'max':
            c = -c  # linprog minimiza, así que cambiamos el signo para maximizar

        A_ub = []
        b_ub = []
        A_eq = []
        b_eq = []
        for r in self.restricciones:
            if r['desigualdad'] == '<=':
                A_ub.append(r['coeficientes'])
                b_ub.append(r['termino_indep'])
            elif r['desigualdad'] == '>=':
                # Multiplicamos por -1 para convertir a <=
                A_ub.append([-coef for coef in r['coeficientes']])
                b_ub.append(-r['termino_indep'])
            elif r['desigualdad'] == '=':
                A_eq.append(r['coeficientes'])
                b_eq.append(r['termino_indep'])

        bounds = [(0, None)] * num_vars  # Variables >= 0

        res = linprog(c,
                      A_ub=np.array(A_ub) if A_ub else None,
                      b_ub=np.array(b_ub) if b_ub else None,
                      A_eq=np.array(A_eq) if A_eq else None,
                      b_eq=np.array(b_eq) if b_eq else None,
                      bounds=bounds,
                      method='highs')

        if not res.success:
            print("No se encontró solución factible.")
            return None

        solucion = {self.variables[i]: res.x[i] for i in range(num_vars)}
        print("Solución relajada:", solucion)

        # Verificar si la solución es entera
        es_entera = all(abs(val - int(round(val))) < 1e-6 for val in solucion.values())

        if es_entera:
            print("¡Solución óptima encontrada!")
            return solucion
        else:
            print("Solución no entera. Generando plano de corte...")
            return solucion
    
    def generar_plano_corte(self, solucion):
        # Seleccionar una variable con valor fraccionario
        vars_frac = [var for var, val in solucion.items() if val != int(val)]
        if not vars_frac:
            return None
        
        var_corte = vars_frac[0]
        val_corte = solucion[var_corte]
        
        # Generar un corte simple (Gomory) - esto es una simplificación
        parte_entera = int(val_corte)
        parte_frac = val_corte - parte_entera
        
        # Crear nueva restricción: x_i <= parte_entera
        nueva_restriccion = {
            'coeficientes': [1 if v == var_corte else 0 for v in solucion.keys()],
            'desigualdad': '<=',
            'termino_indep': parte_entera
        }
        
        print(f"\nNuevo plano de corte: {var_corte} <= {parte_entera}")
        self.restricciones.append(nueva_restriccion)
        return nueva_restriccion
    
    def resolver(self, max_iter=10):
        iteracion = 0
        while iteracion < max_iter:
            iteracion += 1
            print(f"\n--- Iteración {iteracion} ---")
            
            solucion = self.resolver_relajacion_lineal()
            if not solucion:
                print("Problema no factible.")
                break
                
            es_entera = all(val == int(val) for val in solucion.values())
            if es_entera:
                print("\nSolución óptima entera encontrada:")
                for var, val in solucion.items():
                    print(f"{var} = {val}")
                
                # Calcular valor objetivo
                valor_obj = sum(c*val for c, val in zip(
                    self.funcion_objetivo['coeficientes'], 
                    solucion.values()
                ))
                print(f"Valor de la función objetivo: {valor_obj}")
                return solucion
                
            self.generar_plano_corte(solucion)
        
        print("Máximo de iteraciones alcanzado sin solución entera.")
        return None
    
    def graficar(self, solucion=None):
        if len(self.variables) != 2:
            print("La visualización solo está disponible para 2 variables.")
            return
            
        print("\nGenerando gráfico...")
        fig, ax = plt.subplots()
        
        # Configurar ejes
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.grid(True)
        
        # Dibujar restricciones
        for r in self.restricciones:
            a, b = r['coeficientes']
            c = r['termino_indep']
            
            if b != 0:
                # y = (c - a*x)/b
                x_vals = np.linspace(0, 10, 100)
                y_vals = (c - a * x_vals) / b
                ax.plot(x_vals, y_vals, label=f"{a}x1 + {b}x2 {r['desigualdad']} {c}")
            else:
                # x = c/a
                x_val = c / a
                ax.axvline(x=x_val, label=f"{a}x1 {r['desigualdad']} {c}")
        
        # Dibujar solución si existe
        if solucion:
            x, y = solucion.values()
            ax.plot(x, y, 'ro', markersize=10)
            ax.annotate(f'Solución: ({x:.2f}, {y:.2f})', (x, y), 
                         textcoords="offset points", xytext=(10,10), ha='center')
        
        ax.legend()
        ax.set_xlabel(self.variables[0])
        ax.set_ylabel(self.variables[1])
        plt.title("Método de Planos de Corte")
        plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    pc = PlanoCorte()
    pc.ingresar_datos()
    solucion = pc.resolver()
    pc.graficar(solucion)