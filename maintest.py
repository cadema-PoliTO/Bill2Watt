import numpy as np
from bill2watt.methods.slp_predictor import SLPPredictor
from bill2watt.methods.flat_predictor import FlatPredictor
from bill2watt.scaling import flat

import matplotlib.pyplot as plt

x = np.array([1000, 200, 800])
type_ = 'dom'
nd = np.array([21, 5, 5])
m = 10

y_flat = FlatPredictor().predict(x, nd)
y_slp = SLPPredictor().predict(x, nd, key=(type_, m), scaler=flat.evaluate)

plt.plot(y_flat, label='Flat')
plt.plot(y_slp, label='SLP')
plt.legend()
plt.show()