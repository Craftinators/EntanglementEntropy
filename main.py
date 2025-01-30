import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc

# Enable LaTeX rendering
rc('text', usetex=True)
rc('font', family='serif')

def build_hamiltonian_matrix(matrix_size, periodic_boundaries=True, normalized=True):
    if matrix_size % 2 != 0:
        raise ValueError("Matrix size must be even")

    matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    for i in range(matrix_size):
        if i > 0:
            matrix[i, i - 1] = -1
        if i < matrix_size - 1:
            matrix[i, i + 1] = -1

    if periodic_boundaries:
        matrix[0, matrix_size - 1] = -1
        matrix[matrix_size - 1, 0] = -1

    if normalized:
        return 0.5 * matrix
    return matrix

def sort_periodic_boundary_eigenvalues(eigen_results):
    sorted_eigen_results = []
    for i in range(len(eigen_results)):
        if i % 2 == 0:
            sorted_eigen_results.append(eigen_results[i])
        else:
            sorted_eigen_results.insert(0, eigen_results[i])
    return sorted_eigen_results

matrix_size = 250
hamiltonian = build_hamiltonian_matrix(matrix_size)
eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

# Create a list of eigenvalue, eigenvector tuples
eigen_results = [(eigenvalues[i], eigenvectors[:, i]) for i in range(len(eigenvalues))]
sorted_eigen_results = sort_periodic_boundary_eigenvalues(eigen_results)

# Plot cos(2*pi*x) for comparison
x = np.linspace(0, len(sorted_eigen_results), 100)
y = np.cos(2 * np.pi * x / matrix_size)

plt.scatter(range(len(sorted_eigen_results)), [eigen_result[0] for eigen_result in sorted_eigen_results], label='Eigenvalues', zorder=2)
plt.axhline(y=0, color='k', linestyle='--', zorder=0)
plt.plot(x, y, label=r'$f(x) = \cos\left(\frac{2\pi x}{%d}\right)$' % matrix_size, zorder=1, color="red", alpha=0.3)

# Set the corner of the graph to (0, -1.5)
plt.xlim(0, len(sorted_eigen_results))
plt.ylim(-1.5, 1.5)

# Add labels and legend
plt.xlabel(r'Index $k$')
plt.ylabel(r'Energy eigenvalue $E_k$')
plt.legend()

plt.savefig("Energy_Eigenvalues.png", dpi=1200)

lower_bound = int(matrix_size / 4 if matrix_size % 4 == 0 else (matrix_size - 2) / 4 + 1)
upper_bound = int(3 * matrix_size / 4 if matrix_size % 4 == 0 else 3 * (matrix_size - 2) / 4 + 1) + 1

non_positive_eigen_results = sorted_eigen_results[lower_bound : upper_bound]

def build_c_matrix(non_positive_eigen_results):
    c_matrix = np.zeros((matrix_size, matrix_size), dtype=float)
    for x in range(matrix_size):
        for y in range(matrix_size):
            for k in range(len(non_positive_eigen_results)):
                c_matrix[x, y] += non_positive_eigen_results[k][1][x] * non_positive_eigen_results[k][1][y]
    return c_matrix

c_matrix = build_c_matrix(non_positive_eigen_results)
plt.matshow(c_matrix)
plt.savefig("c_matrix.png", dpi=1200)