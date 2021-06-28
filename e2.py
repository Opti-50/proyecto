import sys
from gurobipy import Model, GRB, quicksum
import numpy as np
from typing import List
from datos import *


m = Model()

sys.stdout = open('output-slack.txt', 'w+', encoding='utf-8')
print('\nCONJUNTOS')
print(f'\tPacientes: {I}')
print(f'\tPacientes UCI: {IStar}')
print(f'\tCamas {J}')
print(f'\tCamas UCI {JStar}')
print(f'\tHoras: {H}')
print(f'\tMédicos: {K}')
print(f'\tPersonal de Aseo: {M}')
print('----------------------------\n')
print('PARÁMETROS')
print(f'\tTiempos de Llegada r: {r}')
print(f'\tTiempos de Estadia t: {t}')
print(f'\tDisponibilidad de Médicos: {len(f)}x{len(f[0])}')
print(f'\tDisponibilidad de Aseo: {len(q)}x{len(q[0])}')
print('----------------------------\n')
sys.stdout = sys.__stdout__


# Q = m.addVar(vtype=GRB.INTEGER, lb=0, name="Q")
X = m.addVars(I, J, H, vtype=GRB.BINARY, name="X")
A = m.addVars(I, J, K, H, vtype=GRB.BINARY, name="A")
O = m.addVars(I, J, H, vtype=GRB.BINARY, name="O")
S = m.addVars(M, J, H, vtype=GRB.BINARY, name="S")
Z = m.addVars(J, H, vtype=GRB.BINARY, name="Z")
C = m.addVars(J, H, vtype=GRB.BINARY, name="C")
B = m.addVars(I, J, H, vtype=GRB.BINARY, name="B")


m.update()
m.setObjective(quicksum(A[i, j, k, h]
                          for i in I
                          for j in J
                          for k in K
                          for h in H
                          ), GRB.MAXIMIZE)

# m.addConstr(Q == quicksum(A[i, j, k, h]
#                           for i in I
#                           for j in J
#                           for k in K
#                           for h in H
#                           ),
#             name="R1 - Construcción Variable Q")

m.addConstrs(
    (quicksum(
        O[i, j, h] for j in J) <= 1
     for i in I
     for h in H),
    name="R2 - Paciente utiliza máximo una cama en una hora determinada"
)

m.addConstrs(
    (t[i] * A[i, j, k, h] <= quicksum(O[i, j, h_] for h_ in H[h:(h + t[i] - 1 + 1)])
     for i in I
     for j in J
     for k in K
     for h in H[:(H[-1] - t[i] + 1 + 1)]),
    name="R3 - Continuidad de variable O, desde que es instalado en h hasta h + ti"
)

m.addConstrs(
    (O[i, j, h] <= quicksum(A[i, j, k, h_] for h_ in range(h - t[i], h) for k in K)
     for i in I
     for j in J
     for h in H[t[i]+1:]),
    name="R4 - Continuidad de variable O, si se esta ocupando en un tiempo previo se realizó la instalación"
)

m.addConstrs(
    (q[m][h] * MAX_CAMAS_SANITIZACION_TRAMO >= quicksum(S[m, j, h] for j in J)
     for m in M
     for h in H),
    name="R5 - Personal de Aseo debe estar disponible para ser capaz de sanitizar"
)

m.addConstrs(
    (quicksum(A[i, j, k, h] for i in I) <= Z[j, h] * f[k][h]
     for k in K
     for j in J
     for h in H),
    name="R6 - La cama y el doctor deben estar disponbles para instalar a un paciente"
)

m.addConstrs(
    (quicksum(X[i, j, h] for j in np.setdiff1d(J, JStar) for h in H) == 0
     for i in IStar),
    name="R7 - Paciente UCI no puede ser asignado a cama NO UCI"
)

m.addConstrs(
    (quicksum(X[i, j, h] for j in JStar for h in H) == 0
     for i in np.setdiff1d(I, IStar)),
    name="R8 - Paciente NO UCI no puede ser asignado a cama UCI"
)

m.addConstrs(
    (
        quicksum(A[i, j, k, h_] for h_ in range(1, h+1) for k in K) <=
        quicksum(X[i, j, h_] for h_ in range(1, h+1))
        for i in I
        for j in J
        for h in H),
    name="R9 - No se puede instalar paciente a cama si no fue previamente asignado"
)

m.addConstrs(
    (quicksum(A[i, j, k, h] for j in J for i in I) <= MAX_PACIENTES_INSTALACION_TRAMO
     for k in K
     for h in H),
    name="R10 - Doctor solo puede instalar máximo a un paciente por hora"
)

m.addConstrs(
    (quicksum(A[i, j, k, h] for j in J for k in K for h in H) <= 1
     for i in I),
    name="R11 - Un paciente puede ser instalado máximo una vez"
)

m.addConstrs(
    (quicksum(X[i, j, h] for j in J for h in H) <= 1
     for i in I),
    name="R12 - Un paciente puede ser asignado máximo una vez"
)

m.addConstrs(
    (1 - (quicksum(O[i, j, h]
     for i in I)) >= quicksum(S[m, j, h] for m in M)
     for j in J
     for h in H),
    name="R13 - Una cama solo puede ser sanitizada si está desocupada"
)

m.addConstrs(
    (quicksum(X[i, j, h] for i in I) <= 1
     for j in J
     for h in H),
    name="R14 - En un momento dado una cama solo puede ser asignada máximo a un paciente"
)

m.addConstrs(
    (quicksum(X[i, j, h_] for j in J for h_ in range(1, h+1))
     <= (h/r[i])
     for i in I
     for h in H
     ),
    name="R15 - Solo se puede realizar una asignación a un paciente, si este llegó"
)

m.addConstrs(
    (Z[j, h] == (1/2) * (C[j, h] + 1 - quicksum(O[i, j, h] for i in I))
     for j in J
     for h in H),
    name="R16 - Construcción de la Variable Z (Disponibilidad de Cama)"
)

m.addConstrs(
    (quicksum(O[i, j, h] for i in I) <= 1
     for j in J
     for h in H),
    name="R17 - Una cama solo puede ser ocupada por una persona a la vez"
)

m.addConstrs(
    (quicksum(O[i, j, h] for j in J for h in H) <= t[i]
     for i in I),
    name="R18 - La máxima cantidad de veces que un paciente ocupa alguna cama es igual ti"
)

m.addConstrs(
    (quicksum(A[i, j, k, h] for k in K) <= B[i, j, h + t[i]]
     for j in J
     for i in I
     for h in H[:-t[i]]),
    name="R19 - El paciente deja de usar la cama j, ti horas despues de ser instalado"
)

m.addConstrs(
    (quicksum(B[i, j, h] for j in J for h in H) <= 1
     for i in I),
    name="R20 - Cada paciente puede abandonar una cama máximo una vez"
)

m.addConstrs(
    (quicksum(A[i, j, k, h-t[i]] for k in K) >= B[i, j, h]
     for j in J
     for i in I
     for h in H[t[i]+1:]),
    name="R21 - Si el paciente deja la cama en una hora h, debió ser instalado en ti horas anteriores"
)


m.addConstrs(
    (
        quicksum(S[m, j, h_] for h_ in range(1, h+1) for m in M) <=
        quicksum(B[i, j, h_] for h_ in range(1, h+1) for i in I)
        for h in H
        for j in J
    ),
    name="R22 - Para todo momento el número de veces que se sanitiza una cama debe ser menor o igual a la cantidad de veces que la dejan de ocupar"
)

m.addConstrs(
    (
        1 - C[j, h] >=
        quicksum(B[i, j, h_p] for i in I for h_p in range(1, h + 1)) -
        quicksum(S[m, j, h_p] for m in M for h_p in range(1, h + 1))
        for h in H
        for j in J
    ),
    name="R23 Construcción de C - La cama j no esta sanitizada si se ha dejado de ocupar más veces que la cantidad de sanitizaciones realizadas"
)

m.addConstrs(
    (
        quicksum(X[i, j, h_p] for i in I for h_p in range(1, h+1)) -
        quicksum(B[i, j, h_p]
                 for i in I for h_p in range(1, h+1))
        <= 1
        for j in J
        for h in H
    ),
    name="R24 - En todo momento la diferencia entre asignaciones y retiros de una cama debe ser de a los más 1"
)

m.addConstrs(
    (quicksum(A[i, j, k, h] for k in K for i in I) <= 1
     for h in H
     for j in J
     ),
    name="R25 - Para toda cama solo se permite máximo una instalación por hora"
)

m.addConstrs(
    (quicksum(A[i, j, k, h_] for k in K for h_ in range(1, h+1)) >= quicksum(B[i, j, h_] for h_ in range(1, h+1))
     for j in J
     for i in I
     for h in H),
    name="R26 - El paciente no puede desocupar la cama antes de ser instalado"
)

m.optimize()

sys.stdout = open('output-slack.txt', 'a', encoding='utf-8')
try:
    print('SOLUCIÓN')
    m.printAttr("X")

    # for c in m.getConstrs():
    #     print('%s : %g' % (c.constrName, c.slack))

except Exception as e:
    m.computeIIS()
sys.stdout = sys.__stdout__
