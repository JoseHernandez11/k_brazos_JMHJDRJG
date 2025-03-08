"""
Module: arms/armbinomial.py
Description: Contains the implementation of the ArmBinomial class for the binomial distribution arm.
"""


import numpy as np

from arms import Arm



class ArmBinomial(Arm):
    def __init__(self, n: int, p: float):
        """
        Inicializa el brazo con distribución Binomial.
        :param n: Número de ensayos.
        :param p: Probabilidad de éxito en cada ensayo (valor entre 0 y 1).
        """
        assert n > 0, "El número de ensayos n debe ser mayor que 0."
        assert 0 <= p <= 1, "La probabilidad p debe estar en el rango [0, 1]."
        self.n = n
        self.p = p

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución Binomial.
        :return: Número de éxitos en n ensayos.
        """
        return np.random.binomial(self.n, self.p)

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución Binomial.
        :return: Valor esperado de la distribución (n * p).
        """
        return self.n * self.p

    def __str__(self):
        """
        Representación en cadena del brazo Binomial.
        :return: Descripción detallada del brazo Binomial.
        """
        return f"ArmBinomial(n={self.n}, p={self.p})"

    @classmethod
    def generate_arms(cls, k: int, n: int, p_min: float = 0.1, p_max: float = 0.9):
        """
        Genera k brazos con probabilidades únicas en el rango [p_min, p_max].
        :param k: Número de brazos a generar.
        :param n: Número de ensayos para cada brazo.
        :param p_min: Valor mínimo de la probabilidad.
        :param p_max: Valor máximo de la probabilidad.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."
        assert n > 0, "El número de ensayos n debe ser mayor que 0."
        assert 0 <= p_min < p_max <= 1, "Los valores de p_min y p_max deben estar en el rango [0,1]."
        
        # Generar k valores únicos de p
        p_values = set()
        while len(p_values) < k:
            p = np.random.uniform(p_min, p_max)
            p = round(p, 2)
            p_values.add(p)
        
        return [ArmBinomial(n, p) for p in p_values]

