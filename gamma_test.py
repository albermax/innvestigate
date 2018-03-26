from innvestigate.utils.visualizations import gamma

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    center = 0
    x1 = np.linspace(-0.6,0.8,200)
    x1 += center
    y1 = gamma(x1, minamp = center)


    center = 0.5
    x2 = np.linspace(-0.6,0.8,200)
    x2 += center
    y2 = gamma(x2, minamp = center)

    center = -0.2
    x3 = np.linspace(-0.6,0.8,200)
    x3 += center
    y3 = gamma(x3, minamp = center)

    center = -2
    x3 = np.linspace(-0.6,0.8,200)*3
    x3 += center
    x3 *= 2
    y3 = gamma(x3, minamp = center)

    plt.plot(x1,y1)
    plt.plot(x2,y2)
    plt.plot(x3,y3)
    plt.show()
