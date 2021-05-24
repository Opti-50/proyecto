import numpy as np
from typing import List


# Seed
rng = np.random.default_rng(6547)


PACIENTES_TOTAL = 50
PACIENTES_UCI_TOTAL = 35
CAMAS_TOTAL = 30
CAMAS_UCI_TOTAL = 10
HORAS_TOTAL = 3 * 21
PERSONAL_TOTAL = 10
MEDICO_TOTAL = 10
TIEMPO_CAMA_PROMEDIO = 3 * 12
DESVIACION_CAMA_PROMEDIO = 3 * 2
MAX_CAMAS_SANITIZACION_TRAMO = 8
MAX_PACIENTES_INSTALACION_TRAMO = 12

# Conjuntos
I = [i for i in range(1, PACIENTES_TOTAL+1)]
IStar = rng.choice(I, size=PACIENTES_UCI_TOTAL, replace=False)
J = [i for i in range(1, CAMAS_TOTAL+1)]
JStar = rng.choice(J, size=CAMAS_UCI_TOTAL, replace=False)
H = [i for i in range(1, HORAS_TOTAL+1)]
K = [i for i in range(1, MEDICO_TOTAL+1)]
M = [i for i in range(1, PERSONAL_TOTAL+1)]


def generate_shift_times_continuum(max_hours: int, work_hours: int) -> List[int]:
    return [1 if i % 3 < work_hours else 0 for i in range(0, max_hours+1)]


def generate_shift_times(max_hours: int, work_hours: int) -> List[int]:
    shift_starting_hour = rng.integers(0, 3)
    shift_list = [0] * shift_starting_hour
    shift_list += generate_shift_times_continuum(
        max_hours-shift_starting_hour,
        work_hours,
    )
    return shift_list


# Parametros
r = rng.integers(low=1, high=HORAS_TOTAL, size=PACIENTES_TOTAL+1)
t = np.rint(rng.normal(TIEMPO_CAMA_PROMEDIO,
                       DESVIACION_CAMA_PROMEDIO, PACIENTES_TOTAL+1)).astype(int)

f = [
    generate_shift_times(
        HORAS_TOTAL,
        1
    )
    for _ in range(MEDICO_TOTAL+1)
]
q = [
    generate_shift_times(
        HORAS_TOTAL,
        1
    )
    for _ in range(PERSONAL_TOTAL+1)
]
