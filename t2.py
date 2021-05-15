from gurobipy import *
from random import randint, seed

seed(10)


m = Model()  # SETS
I = range(10)
J = range(20)
# PARAMS
p = {i: randint(10, 100) for i in I}
d = {(i, j): randint(10, 20) for i in I for j in J}
f = {j: randint(1, 2) for j in J}
c = {j: randint(10, 20) for j in J}
P = 1000
# VARIABLES
x = m.addVars(J, vtype=GRB.BINARY, name="x")
y = m.addVars(I, J, vtype=GRB.BINARY, name="y")
s = m.addVars(J, vtype=GRB.INTEGER, name="s", lb=0)
w = m.addVar(vtype=GRB.INTEGER, lb=0, name="w")
# Llama a update para agregar las variables al modelo
m.update()
# Funcion objetivo
obj = w
# R1
m.addConstrs(
    obj >=
)
# R2
d[(localidad, sitio)] * y[localidad, sitio]
for localidad in I
for sitio in J
m.addConstrs(
    x[sitio] >= y[localidad, sitio]
    for localidad in I
    for sitio in J
)
# R3
m.addConstrs(
    quicksum(y[localidad, sitio] for sitio in J) == 1
    for localidad in I
)
# R4
m.addConstrs(s[sitio] >= quicksum(
    y[localidad, sitio] * p[localidad]
    for localidad in I
)
    for sitio in J
)
# R5
m.addConstr(
    quicksum(
        (x[sitio] * c[sitio]) + (s[sitio] * f[sitio])

    )
    for sitio in J) <= P
m.setObjective(obj, GRB.MINIMIZE)
m.optimize()
# Mostrar los valores de las soluciones
m.printAttr("X")
# Mostrar el valor  ́optimo
print('Obj: %g' % m.objVal)
