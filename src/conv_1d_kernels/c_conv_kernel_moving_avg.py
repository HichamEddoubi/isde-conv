import numpy as np
from conv_1d_kernels import CConvKernel


class CConvKernelMovingAverage(CConvKernel):

    def kernel_mask(self):
        self._mask = np.ones(self._kernel_size) / self._kernel_size
