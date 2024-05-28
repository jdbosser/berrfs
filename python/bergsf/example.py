import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import math

num_time_steps = 1000

llambda = 0.0001
pd = 0.5
pb = 0.01
ps = 0.99

area = [(-100.0, 100.0), (-100.0, 100.0)]
tot_area= 200 * 200

def asfloat(*args):

    return [a.astype(float) for a in args]

f = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
g = np.array([[1/2, 0], [1, 0], [0, 1/2], [0, 1]])
q = np.diag([1,1]) * 0.1**2
h = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
r = np.diag([1, 1]) * 2.0**2
# r = np.diag([0, 0])

f,g,q,h,r = asfloat(f, g, q, h, r)
qf = g @ q @ g.T
print(qf)

birth_mean = np.array([0.0, 0.0, 0.0, 0.0])
birth_cov = np.diag([10**2, 0.1, 10**2, 0.1]).astype(float)

def generate_clutter(): 

    num = np.random.poisson(llambda * tot_area)
    # print(llambda * tot_area)
    return [
        [np.random.uniform(mi, ma) for (mi, ma) in area ]
        for _ in range(num)
    ]

def simulation(): 

    target_state: np.ndarray | None = None
    
    detections_over_time = []
    for ii in range(num_time_steps):
        
        detections = []
        clutter = generate_clutter()
        detections.extend(clutter)

        if target_state is None: 
            # Will the target be born? 
            if np.random.rand() < pb: 
                print("birth ", ii)
                target_state = np.random.multivariate_normal(birth_mean, birth_cov)

        else: 
            # Will the target die?
            if np.random.rand() > ps: 

                print("death", ii)
                target_state = None

            else: 
                # Target survived
                target_state = f @ target_state + g @ np.random.multivariate_normal(np.zeros(len(q)), q)
                
                # Will it generate a detection? 
                if np.random.rand() < pd: 

                    # Generate a detection
                    detection = np.random.multivariate_normal(h @ target_state, r)
                    detections.append(detection)

        detections_over_time.append(detections)

    return detections_over_time

def plot_detections(detections_over_time): 

    num_time_steps = len(detections_over_time)
    for k, detections in enumerate(detections_over_time): 
        
        dtarr = np.atleast_2d(np.array(detections))
        if dtarr.shape[1] > 0:
            plt.plot(dtarr[:, 0], dtarr[:, 1], marker = ".", linewidth = 0, c = cm.viridis(k/num_time_steps))
    
    plt.show()

def animate_detections(detections_over_time): 

    fig, ax = plt.subplots()
    from matplotlib.animation import ArtistAnimation
    
    detections_over_time = [np.atleast_2d(detections) for detections in detections_over_time]
    num_time_steps = len(detections_over_time)
    artists = [
        ax.plot(dtarr[:, 0], dtarr[:, 1], marker = ".", linewidth = 0, c = cm.viridis(k/num_time_steps)) 
        for (k, dtarr) in enumerate(detections_over_time)
        if dtarr.shape[1] > 0 
    ]
    anim = ArtistAnimation(fig, artists)
    plt.show()

def setup_filter(): 

    from bergsf import BerPFDetections
    tracker = BerPFDetections(f, g, q, h, r, llambda, 1000, 1000, [(-100.0, 100.0), (-math.sqrt(0.2), math.sqrt(0.2)), (-100.0, 100.), (-math.sqrt(0.2), math.sqrt(0.2))], pb, ps, pd)
    return tracker 

def track(detections, filter): 

    counter = 0
    
    for k, detection in enumerate(detections): 
        det = [np.array(d) for d in detection]
        if filter.prob() > 1:
            breakpoint()
        filter.update(det)
        if filter.prob() > 1:
            breakpoint()
        print("py", k, filter.prob())
        # print(filter.weights())

        yield det, filter.particles(), filter.weights()

def confidence_ellipse(mean, cov, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ellipse

def dets_and_gauss_to_plot(ax, dets_and_gauss): 
    
    det, gausss, weights = dets_and_gauss
    det = np.array(det)
    artists = []
    if len(det) > 0 and det[0].shape[0]>0:
        artists.append(ax.plot(det[:, 0], det[:, 1], marker = ".", linewidth = 0)[0])
    
    # print(gausss) 
    artists.append(ax.scatter(gausss[:, 0], gausss[:, 2], marker = ".", c = weights, cmap = "viridis"))


    
    return artists

if __name__ == "__main__":

    detections = simulation()
    track_over_time = list(track(detections, setup_filter()))

    fig, ax = plt.subplots()
    import itertools
    from matplotlib.animation import ArtistAnimation

    probs = [t[2] for t in track_over_time]
    breakpoint()

    artists = [dets_and_gauss_to_plot(ax, d) for d in track_over_time]
    anim = ArtistAnimation(fig, artists)
    plt.show()
    
    fig, ax = plt.subplots()
    # dets_and_gauss_to_plot(ax, track_over_time)
    animate_detections(simulation())
        

