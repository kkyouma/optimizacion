import numpy as np
import matplotlib.pyplot as plt


def obj_function(x, y):
    return 25 - x**2 - y**2


start = -10
stop = 10
num = 100

x = np.linspace(start, stop, num)
y = x

cost = -np.inf

for air in x:
    for fuel in y:
        cost_fun = obj_function(air, fuel)
        if cost_fun > cost:
            cost = cost_fun
            parameters = [air, fuel]

print(cost, parameters)

X, Y = np.meshgrid(x, y)
Z = obj_function(X, Y)

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection="3d")

ax.set_zlabel("Eficiencia")

ax.plot_wireframe(X, Y, Z, alpha=0.1)
ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.4)
ax.scatter(parameters[0], parameters[1], cost, color="red", s=100)

ax.text(
    parameters[0],
    parameters[1],
    cost,
    f"Máx: ({parameters[0]:.2f}, {parameters[1]:.2f}, {cost:.2f})",
    color="black",
    fontsize=12,
    horizontalalignment="left",
    verticalalignment="bottom",
)
plt.show()
