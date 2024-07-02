# Codigo para incertidumbres en la demanda y en los costos

from pulp import *
import numpy as np
import time


start_time = time.time()

nofsim = 10
# Lsit (TimePeriods)
t = [0, 1, 2, 3, 4, 5, 6]

# Parameters and Data
demand = {1: 100, 2: 100, 3: 150, 4: 200, 5: 150, 6: 100}  # Demand data
UPC = {1: 7, 2: 8, 3: 8, 4: 8, 5: 7, 6: 8}  # Unit Production Cost (Excluding Labor)
UHC = {1: 3, 2: 4, 3: 4, 4: 4, 5: 3, 6: 2}  # Unit Holding Cost
URLC = {1: 15, 2: 15, 3: 18, 4: 18, 5: 15, 6: 15}  # Unit Regular Labor Cost
UOLC = {1: 22.5, 2: 22.5, 3: 27, 4: 27, 5: 22.5, 6: 22.5}  # Unit Overtime Labor Cost
R_MH = {
    1: 120,
    2: 130,
    3: 120,
    4: 150,
    5: 100,
    6: 100,
}  # Available Man-hours R (Regular time) Labor
O_MH = {
    1: 30,
    2: 40,
    3: 40,
    4: 30,
    5: 30,
    6: 30,
}  # Available Man-hours O (Overtime) Labor


cost_list = []


for iter in range(nofsim):
    ran_dem = np.random.uniform(-0.12, 0.12)
    ran_unitcost = np.random.uniform(-0.08, 0.08)
    # Setting the Problem
    prob = LpProblem(
        "Aggregate_Production:Planning:_Fixed_Work_Force_Model", LpMinimize
    )

    # Desicion Variables
    Xt = LpVariable.dicts("Quantity Produced", t, 0)
    It = LpVariable.dicts("Inventory", t, 0)
    Rt = LpVariable.dicts("R_Labor Used", t, 0)
    Ot = LpVariable.dicts("O_Labor Used", t, 0)

    # Objective Function
    prob += (
        lpSum(UPC[i] * (1 + ran_unitcost) * Xt[i] for i in t[1:])
        + lpSum(UHC[i] * It[i] for i in t[1:])
        + lpSum(URLC[i] * Rt[i] for i in t[1:])
        + lpSum(UOLC[i] * Ot[i] for i in t[1:])
    )

    # Constraints
    It[0] = 3
    for i in t[1:]:
        prob += (Xt[i] + It[i - 1] - It[i]) == demand[i] * (
            1 + ran_dem
        )  # Inventory-Balancing Constraints
    for i in t[1:]:
        prob += Xt[i] - Rt[i] - Ot[i] == 0  # Time Required to produce products
    for i in t[1:]:
        prob += Rt[i] <= R_MH[i]  # Regular Time Required
    for i in t[1:]:
        prob += Ot[i] <= O_MH[i]  # Over Time Required

    # Para evitar mensajes en la salida
    # prob.solve()
    solver = getSolver("PULP_CBC_CMD", msg=False)
    prob.solve(solver)

    # Almacenar solo los factibles
    if LpStatus[prob.status] == "Optimal":
        cost_list.append(pulp.value(prob.objective))


end_time = time.time()
# Resultados

desviacion_estandar = np.std(cost_list)
media = np.mean(cost_list)

print("iteraciones", nofsim)
print("Desviación estandar = ", round(desviacion_estandar, 0))
print("Media = ", round(media, 0))
print("tiempo de ejecución", end_time - start_time)
