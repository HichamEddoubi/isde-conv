class CConvKernelCombo:
    def __init__(self, kernels):
        self._kernels = []
        self.kernels = kernels

    @property
    def kernels(self):
        return self._kernels

    @kernels.setter
    def kernels(self, kernels):
        self._kernels = kernels

    def kernel(self, x):
        xp = x.copy()

        for k in range(len(self.kernels)):
            xp = self.kernels[k].kernel(xp)
        return xp
