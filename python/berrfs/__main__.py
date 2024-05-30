
from berrsf import example_setup_gc, example_setup_uc
import matplotlib.pyplot as plt
import numpy as np

def main(): 

    b = example_setup_uc()
    
    x = np.linspace(-6, 6, 1000)
    x = [np.array([xx]) for xx in x]
    
    # b.update([np.array([1.])])

    for _ in range(10): 
        b.update([])
    
    gaussians = b.density_gaussians(x)
    
    for g in gaussians: 
        plt.plot(x, g)
    plt.show()


    print(b)

if __name__ == "__main__":

    main()
