
from bergsf import example_setup
import matplotlib.pyplot as plt
import numpy as np

def main(): 

    b: BerGSF = example_setup()
    
    x = np.linspace(-6, 6, 1000)
    x = [np.array([xx]) for xx in x]
    
    # b.update([np.array([1.])])
    #b.update([np.array([1.])])
    b.predict()
    b.predict()
    
    gaussians = b.density_gaussians(x)
    
    for g in gaussians: 
        plt.plot(x, g)
    plt.show()


    print(b)

if __name__ == "__main__":

    main()
