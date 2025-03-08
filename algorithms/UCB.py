"""
Module: algorithms/epsilon_greedy.py
Description: Implementación del algoritmo epsilon-greedy para el problema de los k-brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np

from algorithms.algorithm import Algorithm

import numpy as np

class UCB1(Algorithm):
    def __init__(self, k: int, c: float = 1.0):
        """
        Inicializa el algoritmo UCB1 con la Desigualdad de Hoeffding.

        :param k: Número de brazos.
        :param c: Parámetro de exploración (generalmente 1).
        """
        super().__init__(k)
        self.counts = np.zeros(k)  # Número de veces que cada brazo ha sido seleccionado
        self.values = np.zeros(k)  # Estimaciones de recompensa promedio de cada brazo
        self.total_pulls = 0       # Total de selecciones realizadas
        self.c = c                 # Parámetro de exploración

    def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política UCB1 con la Desigualdad de Hoeffding.

        :return: índice del brazo seleccionado.
        """
        if self.total_pulls < self.k:
            # Asegurar que cada brazo ha sido jugado al menos una vez
            return self.total_pulls
        
        # Aplicación de la fórmula UCB1 basada en la Desigualdad de Hoeffding
        ucb_values = self.values + self.c * np.sqrt((2 * np.log(self.total_pulls)) / (self.counts + 1e-5))

        # Seleccionamos el brazo con el valor UCB más alto
        return np.argmax(ucb_values)

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza la información del brazo seleccionado después de recibir la recompensa.

        :param chosen_arm: Brazo seleccionado.
        :param reward: Recompensa obtenida.
        """
        self.counts[chosen_arm] += 1
        self.total_pulls += 1

        # Actualización del valor estimado de la recompensa usando la media incremental
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / self.counts[chosen_arm]


import numpy as np

class UCB2:
    def __init__(self, k: int, alpha: float = 0.1):
        """
        Inicializa el algoritmo UCB2.

        :param k: Número de brazos.
        :param alpha: Parámetro de ajuste para exploración-explotación (0 < alpha < 1).
        """
        assert 0 < alpha < 1, "El parámetro alpha debe estar en el rango (0,1)."
        
        self.k = k
        self.alpha = alpha
        
        # Número de veces que cada brazo ha sido seleccionado
        self.counts = np.zeros(k, dtype=int)
        
        # Estimaciones actuales de la recompensa promedio para cada brazo
        self.values = np.zeros(k, dtype=float)
        
        # Contador global de selecciones (t)
        self.total_pulls = 0
        
        # Época actual de cada brazo
        self.epochs = np.zeros(k, dtype=int)
        
        # Veces restantes para jugar un brazo en su época (sin recalcular el índice)
        self.remaining_pulls = np.zeros(k, dtype=int)

    def reset(self):
        """
        Opcional: Restablece todo para repetir experimentos desde cero.
        """
        self.counts[:] = 0
        self.values[:] = 0
        self.total_pulls = 0
        self.epochs[:] = 0
        self.remaining_pulls[:] = 0

    def tau(self, epoch: int) -> int:
        """
        Función de época. Indica cuántas veces se juega un brazo
        dentro de la época 'epoch' (s_i en la literatura).

        :param epoch: Índice de época de un brazo.
        :return: Número de tiradas en esa época.
        """
        return int(np.ceil((1 + self.alpha) ** float(epoch)))

    def select_arm(self) -> int:
        """
        Selecciona un brazo siguiendo la política UCB2.

        :return: Índice del brazo seleccionado.
        """
        # 1) Asegurar que cada brazo se juega al menos una vez antes de aplicar la fórmula
        if self.total_pulls < self.k:
            arm = self.total_pulls
            self.total_pulls += 1
            self.counts[arm] += 1
            return arm

        # 2) Revisar si alguno de los brazos aún tiene 'remaining_pulls'
        for arm in range(self.k):
            if self.remaining_pulls[arm] > 0:
                self.remaining_pulls[arm] -= 1
                self.total_pulls += 1
                self.counts[arm] += 1
                return arm

        # 3) Calcular los índices UCB2 de todos los brazos y elegir el de mayor valor
        ucb_values = np.zeros(self.k)
        for arm in range(self.k):
            # La "escala" de la época actual se define por tau(self.epochs[arm])
            epoch_arm = self.epochs[arm]
            t_arm = self.tau(epoch_arm)

            # Para evitar log de valores muy bajos o no válidos:
            # -> argumento del log: max(1, self.total_pulls / t_arm)
            argument = max(1e-12, float(self.total_pulls) / max(1, t_arm))

            # Término de confianza (bonus)
            bonus = np.sqrt(
                (1 + self.alpha)
                * max(0, np.log(np.e * argument))
                / (2 * max(1, t_arm))  # evitar división por 0
            )
            ucb_values[arm] = self.values[arm] + bonus

        chosen_arm = np.argmax(ucb_values)

        # 4) Al seleccionar un brazo, incrementar su época y determinar cuántas veces se usará
        self.epochs[chosen_arm] += 1
        new_epoch = self.epochs[chosen_arm]
        
        # Cantidad de tiradas en la nueva época
        pulls_in_new_epoch = self.tau(new_epoch)
        # Cantidad de tiradas en la época anterior
        pulls_in_old_epoch = self.tau(new_epoch - 1)
        
        # Asignar la diferencia (cuántas veces se jugará este brazo en la nueva época)
        self.remaining_pulls[chosen_arm] = pulls_in_new_epoch - pulls_in_old_epoch
        
        # Finalmente, restar 1 porque ya vamos a seleccionar este brazo "ahora mismo"
        self.remaining_pulls[chosen_arm] -= 1
        self.total_pulls += 1
        self.counts[chosen_arm] += 1

        return chosen_arm

    def update(self, chosen_arm: int, reward: float):
        """
        Actualiza la estimación de la recompensa promedio del brazo elegido.

        :param chosen_arm: Índice del brazo seleccionado.
        :param reward: Recompensa obtenida.
        """
        # Media incremental
        n = self.counts[chosen_arm]
        current_value = self.values[chosen_arm]
        self.values[chosen_arm] = current_value + (reward - current_value) / n