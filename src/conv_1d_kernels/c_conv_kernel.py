import numpy as np
from abc import ABC, abstractmethod


class CConvKernel(ABC):

    def __init__(self, kernel_size=3):
        self._kernel_size = kernel_size
        self._mask = None

    @property
    def kernel_size(self):
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, kernel_value):
        if kernel_value % 2 == 0:
            raise ValueError("Kernel size must be odd !")
        # else:
        self._kernel_size = kernel_value
        # regenerate kernel mask in the child classes after kernel size update
        self.kernel_mask()

    @property
    def mask(self):
        return self._mask

    @abstractmethod
    def kernel_mask(self):
        raise NotImplementedError("Method not implemented yet !")

    def kernel(self, x):
        xp = x.copy()
        k = (self._kernel_size - 1) // 2

        for i in range(k, x.size-k):
            xp[i] = np.dot(x[i-k: i+k+1], self._mask)
        return xp
