import sys
from gurobipy import Model, GRB, quicksum
import numpy as np
from typing import List

rng = np.random.default_rng(69699)
m = Model()

PACIENTES_TOTAL = 5
PACIENTES_UCI_TOTAL = 2
CAMAS_TOTAL = 5
CAMAS_UCI_TOTAL = 2
HORAS_TOTAL = 100
PERSONAL_TOTAL = 5
MEDICO_TOTAL = 5
TIEMPO_CAMA_PROMEDIO = 40
DESVIACION_CAMA_PROMEDIO = 4
HORAS_TRABAJO_PERSONAL_MIN = 7
HORAS_TRABAJO_PERSONAL_MAX = 9
HORAS_TRABAJO_MEDICO_MIN = 7
HORAS_TRABAJO_MEDICO_MAX = 9

I = [i for i in range(1, PACIENTES_TOTAL)]
IStar = rng.choice(I, size=PACIENTES_UCI_TOTAL, replace=False)
J = [i for i in range(1, CAMAS_TOTAL)]
JStar = rng.choice(J, size=CAMAS_UCI_TOTAL, replace=False)
H = [i for i in range(1, HORAS_TOTAL)]
K = [i for i in range(1, PERSONAL_TOTAL)]
M = [i for i in range(1, MEDICO_TOTAL)]


def generate_shift_times_continuum(max_hours: int, work_hours: int) -> List[int]:
    return [1 if i % 24 < work_hours else 0 for i in range(0, max_hours)]


def generate_shift_times(max_hours: int, work_hours: int) -> List[int]:
    shift_starting_hour = rng.integers(0, 24)
    shift_list = [0] * shift_starting_hour
    shift_list += generate_shift_times_continuum(
        max_hours-shift_starting_hour,
        work_hours,
    )
    return shift_list


r = rng.integers(low=1, high=HORAS_TOTAL, size=PACIENTES_TOTAL-1)
t = np.rint(rng.normal(TIEMPO_CAMA_PROMEDIO,
            DESVIACION_CAMA_PROMEDIO, PACIENTES_TOTAL-1)).astype(int)
f = [
    generate_shift_times(
        HORAS_TOTAL,
        rng.integers(
            HORAS_TRABAJO_PERSONAL_MIN,
            HORAS_TRABAJO_PERSONAL_MAX,
        )
    )
    for _ in range(PERSONAL_TOTAL)
]
q = [
    generate_shift_times(
        HORAS_TOTAL,
        rng.integers(
            HORAS_TRABAJO_MEDICO_MIN,
            HORAS_TRABAJO_MEDICO_MAX
        )
    )
    for _ in range(MEDICO_TOTAL)
]


# print(r)
print("CHECK")
print(H)
print(f"Pacientes : {I}")
print(f"Tiempos de llegada : {r}")
print(f"Tiempos de estadia : {t}")


Q = m.addVar(vtype=GRB.INTEGER, lb=0, name="Q")
X = m.addVars(I, J, H, vtype=GRB.BINARY, name="X")
O = m.addVars(I, J, H, vtype=GRB.BINARY, name="O")
A = m.addVars(I, J, K, H, vtype=GRB.BINARY, name="A")
S = m.addVars(M, J, H, vtype=GRB.BINARY, name="S")
Z = m.addVars(J, H, vtype=GRB.BINARY, name="Z")
C = m.addVars(J, H, vtype=GRB.BINARY, name="C")
B = m.addVars(I, J, H, vtype=GRB.BINARY, name="B")

print("CHECK2")
print(f"Pacientes : {I}")

m.update()

m.setObjective(Q, GRB.MAXIMIZE)
print("CHECK3")
print(f"Pacientes : {I}")
m.addConstr(Q == quicksum(A[i, j, k, h]
                          for i in I
                          for j in J
                          for k in K
                          for h in H
                          ),
            name="R1")

m.addConstrs(
    (quicksum(
        O[i, j, h] for j in J) <= 1
     for i in I
     for h in H),
    name="R2"
)
print(f'I: {I}')
m.addConstrs(
    (t[i-1] * A[i, j, k, h] <= quicksum(O[i, j, h_]
                                        for h_ in H[:(H[-1] - t[i-1] + 1)])
     for i in I
     for j in J
     for k in K
     for h in H[:(H[-1] - t[i-1] + 2)]),
    name="R3"
)


m.addConstrs(
    (q[m][h] >= S[m, j, h]
     for m in M
     for j in J
     for h in H),
    name="R4"
)


m.addConstrs(
    (quicksum(S[m, j, h] for j in J) <= 1
     for m in M
     for h in H),
    name="R5"
)

m.addConstrs(
    (quicksum(A[i, j, k, h] for i in I) <= Z[j, h] * f[k][h]
     for k in K
     for j in J
     for h in H),
    name="R6"
)

m.addConstrs(
    (quicksum(X[i, j, h] for j in np.setdiff1d(J, JStar) for h in H) == 0
     for i in IStar),
    name="R7"
)

print("CHECK2")

m.addConstrs(
    (quicksum(X[i, j, h] for j in JStar for h in H) == 0
     for i in np.setdiff1d(I, IStar)),
    name="R8"
)


m.addConstrs(
    (quicksum(A[i, j, k, h] for k in K) <= quicksum(X[i, j, h_] for h_ in H)
     for i in I
     for j in J
     for h in H),
    name="R9"
)

m.addConstrs(
    (quicksum(A[i, j, k, h] for j in J for i in I) <= 1
     for k in K
     for h in H),
    name="R10"
)

m.addConstrs(
    (quicksum(A[i, j, k, h] for k in K) <= 1
     for j in J
     for i in I
     for h in H),
    name="R11"
)

m.addConstrs(
    (quicksum(A[i, j, k, h] for j in J for k in K for h in H) <= 1
     for i in I),
    name="R12"
)

m.addConstrs(
    ((quicksum((1 - O[i, j, h])
     for i in I)/len(I)) >= quicksum(S[m, j, h] for m in M)
     for j in J
     for h in H),
    name="R13"
)

m.addConstrs(
    (C[j, h+1] >= quicksum(S[m, j, h] for m in M)
     for j in J
     for h in H[:-1]),
    name="R14"
)

# TODO REVISAR EVENTUALMENTE
m.addConstrs(
    (Z[j, h] >= quicksum(X[s, j, h] for s in list(filter(lambda x: r[x-1] <= h, I)))
     for j in J
     for h in H),
    name="R15"
)

m.addConstrs(
    (quicksum(X[i, j, h] for j in J for h in H[:r[i-1]]) == 0
     for i in I),
    name="R69420xxX"
)

# m.addConstrs(
#     (quicksum(O[i, j, h] for j in J for h in H[:r[i-1]]) == 0
#      for i in I),
#     name="R420xxX69"
# )

m.addConstrs(
    (Z[j, h] >= C[j, h] - quicksum(O[i, j, h] for i in I)
     for j in J
     for h in H),
    name="R16"
)

# TODO REVISAR EVENTUALMENTE
m.addConstrs(
    (quicksum(A[i, j, k, h] for k in K) <= B[i, j, h + t[i-1]]
     for j in J
     for i in I
     for h in H[:-t[i-1]]),
    name="R17"
)

m.addConstrs(
    (quicksum(B[i, j, h] for j in J for h in H) <= 1
     for i in I),
    name="R18"
)

m.addConstrs(
    (quicksum(X[i, j, h] for j in J for h in H) <= 1
     for i in I),
    name="xXxPussy69xXx"
)


# # ARREGLAR
# m.addConstrs(
#     name="R18"
# )

# # ARREGLAR
# m.addConstrs(
#     name="R19"
# )


m.optimize()

sys.stdout = open('output.txt', 'w+')
m.printAttr("X")
