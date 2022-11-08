import numpy as np
from conv_1d_kernels import CConvKernel


class CConvKernelTriangle(CConvKernel):

    def kernel_mask(self):
        self._mask = np.ones(self._kernel_size)
        for i in range((self._kernel_size//2)):
            print(i)
            self._mask[i] += i
            self._mask[self._kernel_size-i-1] += i

        self._mask[(self._kernel_size//2)] += (self._kernel_size//2)
        self._mask /= np.sum(self._mask)
