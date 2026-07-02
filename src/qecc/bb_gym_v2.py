import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.integrate import trapezoid
from qecc.polynomialCodes import generateBicycleCode, generateABmatrices
from qecc.logicals import calculateCodeDimension

INT_DATA_TYPE = np.int16
NEGATIVE_REWARD = -1


class bicycleBivariateCodeEnvironmentV2(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, l, m, max_ax, max_ay, max_bx, max_by,
                 evaluationDecoderFunction,
                 errorRange=None,
                 minimumNumberOfLogicalQubits=6,
                 render_mode=None):
        if errorRange is None:
            errorRange = np.linspace(0.0001,0.1,10)
        self.render_mode = render_mode
        self._l = l
        self._m = m
        self._max_ax = max_ax
        self._max_ay = max_ay
        self._max_bx = max_bx
        self._max_by = max_by
        self.decoder = evaluationDecoderFunction
        self.errorRange = errorRange
        if any(a >= b for a, b in zip(errorRange, errorRange[1:])):
            raise ValueError(
                f"errorRange must be strictly increasing (e.g. [0.001, 0.01, 0.1]); got {list(errorRange)}"
            )
        self.minimumNumberOfLogicalQubits = minimumNumberOfLogicalQubits

        self.action_space = spaces.MultiDiscrete(
            [max_ax + 1, max_ay + 1, max_bx + 1, max_by + 1]
        )

        self.aX = np.zeros(max_ax, INT_DATA_TYPE)
        self.aY = np.zeros(max_ay, INT_DATA_TYPE)
        self.bX = np.zeros(max_bx, INT_DATA_TYPE)
        self.bY = np.zeros(max_by, INT_DATA_TYPE)

        self.A, self.B = generateABmatrices(
            l, m,
            np.where(self.aX != 0)[0],
            np.where(self.aY != 0)[0],
            np.where(self.bX != 0)[0],
            np.where(self.bY != 0)[0],
        )
        self.Hx, self.Hz = generateBicycleCode(
            l, m,
            np.where(self.aX != 0)[0],
            np.where(self.aY != 0)[0],
            np.where(self.bX != 0)[0],
            np.where(self.bY != 0)[0],
        )

        self.flatObservationSize = (
            2 * (l * m) ** 2
            + max_ax + max_ay + max_bx + max_by
        )
        self.observation_space = spaces.MultiBinary(self.flatObservationSize)

    def render(self):
        pass

    def close(self):
        pass

    def _getObservation(self):
        return np.concatenate([
            np.vstack((self.A, self.B)).flatten(),
            self.aX, self.aY, self.bX, self.bY,
        ]).astype(np.int8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.aX = np.zeros(self._max_ax, INT_DATA_TYPE)
        self.aY = np.zeros(self._max_ay, INT_DATA_TYPE)
        self.bX = np.zeros(self._max_bx, INT_DATA_TYPE)
        self.bY = np.zeros(self._max_by, INT_DATA_TYPE)
        self.A, self.B = generateABmatrices(
            self._l, self._m,
            np.where(self.aX != 0)[0],
            np.where(self.aY != 0)[0],
            np.where(self.bX != 0)[0],
            np.where(self.bY != 0)[0],
        )
        self.Hx, self.Hz = generateBicycleCode(
            self._l, self._m,
            np.where(self.aX != 0)[0],
            np.where(self.aY != 0)[0],
            np.where(self.bX != 0)[0],
            np.where(self.bY != 0)[0],
        )
        return self._getObservation(), {}

    def step(self, action):
        idx_ax, idx_ay, idx_bx, idx_by = action

        if idx_ax < self._max_ax:
            self.aX[idx_ax] ^= 1
        if idx_ay < self._max_ay:
            self.aY[idx_ay] ^= 1
        if idx_bx < self._max_bx:
            self.bX[idx_bx] ^= 1
        if idx_by < self._max_by:
            self.bY[idx_by] ^= 1

        self.A, self.B = generateABmatrices(
            self._l, self._m,
            np.where(self.aX != 0)[0],
            np.where(self.aY != 0)[0],
            np.where(self.bX != 0)[0],
            np.where(self.bY != 0)[0],
        )
        self.Hx, self.Hz = generateBicycleCode(
            self._l, self._m,
            np.where(self.aX != 0)[0],
            np.where(self.aY != 0)[0],
            np.where(self.bX != 0)[0],
            np.where(self.bY != 0)[0],
        )

        if calculateCodeDimension(self.Hx, self.Hz) >= self.minimumNumberOfLogicalQubits:
            logicalErrorRate, decoderFailureRate = self.decoder(
                self.Hx, self.Hz, self.errorRange, seed=None
            )
            reward = float(self._calculateReward(logicalErrorRate, decoderFailureRate))
        else:
            reward = float(NEGATIVE_REWARD)

        return self._getObservation(), reward, False, False, {}

    def _calculateReward(self, logicalErrorRate, decoderFailureRate):
        outputBER = logicalErrorRate + decoderFailureRate
        return trapezoid(1 - outputBER, self.errorRange)

