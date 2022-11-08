import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from conv_1d_kernels import CConvKernelMovingAverage
from conv_1d_kernels import CConvKernelTriangle
from conv_1d_kernels import CConvKernelCombo

data = pd.read_csv('../data/mnist_data.csv')
data = np.array(data)

labels = data[:, 0]
data = data[:, 1:] / 255

# print(data.shape, labels.shape, np.unique(labels))

x = data[0, :]

plt.subplot(1, 4, 1)
plt.imshow(x.reshape(28, 28))

kernel_filter_tr = CConvKernelTriangle(9)
kernel_filter_tr.kernel_mask()
# print(kernel_filter_tr.mask)
xp_tr = kernel_filter_tr.kernel(x)

kernel_filter_avg = CConvKernelMovingAverage(9)
kernel_filter_avg.kernel_mask()
# print(kernel_filter_avg.mask)
xp_avg = kernel_filter_avg.kernel(x)

kernel_filter_avg = CConvKernelCombo([kernel_filter_tr, kernel_filter_avg])
# print(kernel_filter_avg.mask)
xp_c = kernel_filter_avg.kernel(x)

plt.subplot(1, 4, 2)
plt.imshow(xp_avg.reshape(28, 28))

plt.subplot(1, 4, 3)
plt.imshow(xp_tr.reshape(28, 28))

plt.subplot(1, 4, 4)
plt.imshow(xp_c.reshape(28, 28))
plt.show()
