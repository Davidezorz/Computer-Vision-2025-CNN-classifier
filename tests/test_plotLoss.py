from utils import models_eval 
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    a, b, n = 0, 10, 100
    y1 = np.linspace(a, b, n)
    y2 = np.linspace(a, b, n)*0.9

    models_eval.plotLoss(y1, y2)
    plt.show()
