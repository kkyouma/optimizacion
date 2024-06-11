import matplotlib.pyplot as plt
import numpy as np
import pulp

x1 = pulp.LpVariable("A", lowBound=1, cat="Continuous")
x2 = pulp.LpVariable("B", lowBound=0, upBound=5, cat="Continuous")

# plotting lists for:
# Pareto graph
Z_1 = []
Z_2 = []
# objective function value as a function of alpha
alpha_values = []
obj_value = []
alphas = np.linspace(0, 1, 100)

# iterate through alpha values from 0 to 1 and write PuLP solutions into solutionTable
for alpha in alphas:
    # Define the problem
    linearProblem = pulp.LpProblem("Multi_objective_minimization", pulp.LpMinimize)
    # Define the objetive funtion
    linearProblem += alpha * (4 * x1 - x2) + (1 - alpha) * (-0.5 * x1 + x2)
    # Add constraints
    linearProblem += 2 * x1 + x2 <= 8
    linearProblem += x1 - x2 <= 4
    # Solve the problem
    solution = linearProblem.solve()
    z1 = 4 * pulp.value(x1) - pulp.value(x2)
    z2 = -0.5 * pulp.value(x1) + pulp.value(x2)
    Z_1.append(z1)
    Z_2.append(z2)
    alpha_values.append(alpha)
    obj_value.append(pulp.value(linearProblem.objective))


# Pareto graph
plt.plot(Z_1, Z_2, color="red")
plt.xlabel("z1", size=12)
plt.ylabel("z2", size=12)
plt.title("Pareto graph", size=12)
plt.show()

# Combined objective function value as a function of alpha
plt.plot(alpha_values, obj_value, color="red")
plt.xlabel("alpha", size=12)
plt.ylabel("obj_value", size=12)
plt.title("Optimal combined objective function value as a function of alpha", size=12)
plt.show()
