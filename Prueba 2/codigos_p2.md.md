# OPT0-Prueba 2

```python
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
import time

(x, y, z, v, w, t) = sp.var("x y z v w t")
```

## Metodo Newton Multivariable

```python
def newton_raphson_method(
    function: any,
    variables: list[any] | tuple[any],
    point: list[float] | tuple[float],
    tol: float = 0.01,
    min_value: float = 1e9,
    max_iter: int = 100,
    print_solutions: bool = False,
) -> None:

    expression = function(*variables)
    grad = [sp.diff(expression, var) for var in variables]
    H = sp.hessian(expression, variables)

    for i in range(max_iter):

        J = [g.subs(zip(variables, point)) for g in grad]
        H_eval = H.subs(zip(variables, point))
        current_value = function(*point)

        if abs(min_value - current_value) < tol:
            solution = (current_value, tuple(point))

            if print_solutions:
                print(f"Solucion {solution}\nIteraciones: {i}")

            return solution

        min_value = current_value

        H_eval = H.subs(zip(variables, point))
        H_eval = np.array(H_eval, dtype="float")
        H_inv = np.linalg.inv(H_eval)
        point = point - H_inv @ J

```

Para agregar puntos creados por numeros random

```python
def newton_random(
    function: any,
    variables: list[any] | tuple[any],
    initial_bounds: tuple[float] | list[float],
    tol: float = 0.01,
    min_value: float = 1e9,
    max_iters: list[int] = [100],
    num_random_points: int = 10,
    get_time: bool = False,
    print_solutions: bool = False,
) -> tuple:

    info_collected = {}
    time_dict = {}

    if isinstance(max_iters, int):
        max_iters = [max_iters]

    for i in max_iters:
        start_time = time.perf_counter()
        solutions = []
        for _ in range(num_random_points):
            random_point = np.random.uniform(*initial_bounds, len(variables))

            solution = newton_raphson_method(
                function, variables, random_point, tol, min_value, i, print_solutions
            )
            if solution:
                solutions.append(solution)

        info_collected[i] = solutions

        end_time = time.perf_counter() - start_time
        time_dict[i] = end_time

    if get_time:
        return info_collected, time_dict

    return info_collected
```

- Para graficar

```python
x = np.linspace(-20, 20, 100)
y = np.linspace(-20, 20, 100)
X, Y = np.meshgrid(x, y)

Z = f(X, Y) # Funcion objetivo

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='plasma')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X, Y)')
ax.set_title('Gráfico de F(x, y)')

plt.show()
```

![[Pasted image 20240609191831.png]]

## Metodo del Descenso del Gradiente

```python
def gradient_descent_method(
    function: callable,
    variables: list[any] | tuple[any],
    point: list[float] | tuple[float],
    alpha: float = 0.01,
    tol: float = 0.01,
    min_value: float = 1e9,
    max_iter: int = 100,
    print_solutions: bool = False,
) -> tuple[float, tuple]:

    expression = function(*variables)
    grad = [sp.diff(expression, var) for var in variables]

    for i in range(max_iter):

        # Se ve feo, pero es solo para estandarizar el formato
        J = np.array([float(g.subs(dict(zip(variables, point)))) for g in grad])
        current_value = float(expression.subs(dict(zip(variables, point))))

        if np.isnan(current_value) or np.isinf(current_value):
            break

        if abs(min_value - current_value) < tol:
            solution = (current_value, tuple(point))

            if print_solutions:
                print(f"Solucion {solution}\nIteraciones: {i}")

            return solution

        min_value = current_value

        point = (
            np.array(point, dtype="float") - alpha * J
        )

    return None
```

### Usar numeros aleatorios

```python
def gradient_method_random(
    function: callable,
    variables: list[any] | tuple[any],
    initial_bounds: tuple[float] | list[float],
    alpha: float = 0.01,
    tol: float = 0.001,
    min_value: float = 1e9,
    max_iters: list[int] = [100],
    num_random_points: int = 10,
    get_time: bool = False,
    print_solutions: bool = False
) -> tuple:
    info_collected = {}
    time_dict = {}

    if isinstance(max_iters, int):
        max_iters = [max_iters]

    for i in max_iters:
        start_time = time.perf_counter()
        solutions = []

        for _ in range(num_random_points):
            random_point = np.random.uniform(*initial_bounds, len(variables))

            solution = gradient_descent_method(
                function, variables, random_point, alpha, tol, min_value, i, print_solutions
            )
            if solution:
                solutions.append(solution)

        info_collected[i] = solutions

        end_time = time.perf_counter() - start_time
        time_dict[i] = end_time
    if get_time:
        return info_collected, time_dict

    return info_collected
```

### Graficar tiempos

```python
plt.plot(times.keys(), times.values(), "o-", color = "darkblue")
plt.xlabel("Maximo de Iteraciones")
plt.ylabel("Tiempo de Cómputo")
```

### Limpiar datos

```python
def clean_solutions(
    solutions: dict[int, list[tuple]], decimal_places: int = 3
) -> dict[int, list[tuple]]:
    cleaned_solutions = {}

    for key, value_list in solutions.items():
        unique_values = []
        seen_values = set()

        for value, point in value_list:
            rounded_value = round(value, decimal_places)
            if rounded_value not in seen_values:
                seen_values.add(rounded_value)
                unique_values.append((value, point))

        cleaned_solutions[key] = unique_values

    return cleaned_solutions
```

### Encontrar minimo y maximo de soluciones

```python
def find_min_max(solutions: dict[float, list[tuple]]) -> tuple[tuple]:
    min_solution = None
    max_solution = None

    for values in solutions.values():
        for value in values:
            if min_solution is None or value[0] < min_solution[0]:
                min_solution = value
            if max_solution is None or value[0] > max_solution[0]:
                max_solution = value

    return min_solution, max_solution

```

## Programacion Lineal con PuLP

```python
import pulp as p
```

### Problema simple

```python
prob = p.LpProblem("Maximize_profit", p.LpMaximize)

A = p.LpVariable("A", lowBound=0, cat="Integer")
B = p.LpVariable("B", lowBound=0, cat="Integer")

prob += A * 20 + B * 30, "Profit"

prob += A + B * 2 <= 100, "time_limit1"
prob += A * 2 + B <= 100, "time_limit2"

prob.solve()
print(p.LpStatus[prob.status])

print(f"Production of chairs: {A.varValue}")
print(f"Production of tables: {B.varValue}")

print(p.value(prob.objective))

```

### Planificacion de la produccion con PuLP

#### Problema 1

```python
t = [0, 1, 2, 3, 4, 5, 6]

demand = {1:100, 2:100, 3:150, 4:200, 5:150, 6:100}
UPC = {1:7, 2:8, 3:8, 4:8, 5:7, 6:8}
UHC = {1:3, 2:4, 3:4, 4:4, 5:3, 6:2}
URLC = {1:15, 2:15, 3:18, 4:18, 5:15, 6:15}
UOLC = {1:22.5, 2:22.5, 3:27, 4:27, 5:22.5, 6:22.5}
R_MH = {1:120, 2:130, 3:120, 4:150, 5:100, 6:100}
O_MH = {1:30, 2:40, 3:40, 4:30, 5:30, 6:30}

prob = p.LpProblem("Aggregate_Production", p.LpMinimize)

Xt = p.LpVariable.dicts("Quantity_Produced", t, 0)
It = p.LpVariable.dicts("Inventory", t, 0)
Rt = p.LpVariable.dicts("R_Labor_Used", t, 0)
Ot = p.LpVariable.dicts("O_Labor_Used", t, 0)

prob += p.lpSum(
    UPC[i] * Xt[i] +
    UHC[i] * It[i] +
    URLC[i] * Rt[i] +
    UOLC[i] * Ot[i]
    for i in t[1:]
)

initial_inventory = 3
prob += It[0] == initial_inventory

for i in t[1:]:
    prob += (Xt[i] + It[i-1] - It[i]) == demand[i]
    prob += Xt[i] - Rt[i] - Ot[i] == 0
    prob += Rt[i] <= R_MH[i]
    prob += Ot[i] <= O_MH[i]

prob.solve()
print("Solution Status = ", p.LpStatus[prob.status])

for v in prob.variables():
    if v.varValue>0:
        print(v.name, "=", v.varValue)

```

#### Problema 2

```python
factories = pd.read_csv("factory_variables.csv", index_col=["Month", "Factory"])
demand = pd.read_csv("monthly_demand.csv", index_col=["Month"])

model = p.LpProblem("Minimizacion_costos", p.LpMinimize)

production = p.LpVariable.dicts(
    "production",
    ((month, factory) for month, factory in factories.index),
    lowBound=0,
    cat="integer",
)

factory_status = p.LpVariable.dicts(
    "factory_status",
    ((month, factory) for month, factory in factories.index),
    cat="binary",
)


model += p.lpSum(
    [
        production[month, factory] * factories.loc[(month, factory), "Variable_Costs"]
        for month, factory in factories.index
    ]
    + [
        factory_status[month, factory] * factories.loc[(month, factory), "Fixed_Costs"]
        for month, factory in factories.index
    ]
)

months = demand.index
for month in months:
    model += (
        production[(month, "A")] + production[(month, "B")]
        == demand.loc[month, "Demand"]
    )

for month, factory in factories.index:
    min_production = factories.loc[(month, factory), "Min_Capacity"]
    max_production = factories.loc[(month, factory), "Min_Capacity"]

    model += (
        production[(month, factory)] >= min_production * factory_status[month, factory]
    )

    model += (
        production[(month, factory)] <= max_production * factory_status[month, factory]
    )

model.solve()

print("Status", p.LpStatus[model.status])


output = []
for month, factory in production:
    var_output = {
        "Month": month,
        "Factory": factory,
        "Production": production[(month, factory)].varValue,
        "Factory Status": factory_status[(month, factory)].varValue,
    }
    output.append(var_output)

output_df = pd.DataFrame(output)

output_df
```

### Programacion MultiObjetivo con PuLP

```python

x1 = p.LpVariable("A", lowBound=1, cat="Continous")
x2 = p.LpVariable("B", lowBound=0, upBound=5, cat="Continous")

Z_1 = []
Z_2 = []

alpha_values = []
obj_value = []
alphas = np.linspace(0, 1, 100)

for alpha in alphas:
    linearProblem = p.LpProblem("Multi_objetive_minimizatio", p.LpMinimize)
    linearProblem += alpha * (4 * x1 - x2) + (1 - alpha) * (-0.5 * x1 + x2)

    linearProblem += 2 * x1 + x2 <= 8
    linearProblem += x1 - x2 <= 4

    solution = linearProblem.solve()

    z1 = 4 * p.value(x1) - p.value(x2)
    z2 = -0.5 * p.value(x1) + p.value(x2)

    Z_1.append(z1)
    Z_2.append(z2)
    alpha_values.append(alpha)
    obj_value.append(p.value(linearProblem.objective))

```

### Graficar pareto

```python
plt.figure(figsize=(10, 5))
plt.plot(Z_1, Z_2, color="red")
plt.xlabel("z1", size=12)
plt.ylabel("z2", size=12)
plt.title("Pareto graph", size=14)
plt.grid(True)
plt.show()
```

### Graficar combinacion optima

```python
plt.figure(figsize=(10, 5))
plt.plot(alpha_values, obj_value, color="red")
plt.xlabel("alpha", size=12)
plt.ylabel("Objective Value", size=12)
plt.title("Optimal combined objective function value as a function of alpha", size=14)
plt.grid(True)
plt.show()
```
