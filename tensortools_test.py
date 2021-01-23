import tensortools
import numpy as np
import matplotlib.pyplot as plt


def cp_als_test():
    # Create synthetic dataset.
    I, J, K, R = 25, 25, 25, 3  # dimensions and rank parameters
    # Create a random tensor consisting of a low-rank component and noise.
    X = tensortools.randn_ktensor((I, J, K), rank=R).full()
    X += np.random.randn(I, J, K)  # add some random noise

    # Perform CP tensor decomposition.
    U = tensortools.cp_als(X, rank=R, verbose=True)
    V = tensortools.cp_als(X, rank=R, verbose=True)

    # Compare the low-dimensional factors from the two fits.
    fig, _, _ = tensortools.plot_factors(U.factors)
    tensortools.plot_factors(V.factors, fig=fig)

    # Align the two fits and print a similarity score.
    similarity_score = tensortools.kruskal_align(U.factors, V.factors, permute_U=True, permute_V=True)
    print(similarity_score)

    # Plot the results to see alignment.
    fig, ax, po = tensortools.plot_factors(U.factors)
    tensortools.plot_factors(V.factors, fig=fig)

    plt.show()


if __name__ == '__main__':
    cp_als_test()
