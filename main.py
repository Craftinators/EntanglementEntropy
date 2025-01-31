import matplotlib.pyplot as plt
import numpy as np
from rich.progress import Progress
from rich.prompt import Prompt

def build_hamiltonian_matrix(matrix_size, with_periodic_boundary, normalized, rich_progress=None, task=None):
    if matrix_size % 2 != 0 or matrix_size < 0:
        raise ValueError("Matrix size must be an even number greater than 0")
    h = np.zeros((matrix_size, matrix_size), dtype=np.int64)
    for i in range(matrix_size):
        if i > 0:
            h[i, i - 1] = -1
        if i < matrix_size - 1:
            h[i, i + 1] = -1

        # Update progress bar
        if rich_progress is not None and task is not None:
            rich_progress.update(task, advance=1)

    if with_periodic_boundary:
        h[0, matrix_size - 1] = -1
        h[matrix_size - 1, 0] = -1

    return 0.5 * h if normalized else h

def build_correlation_matrix(valid_eigen_results, rich_progress=None, task=None):
    c_size = len(valid_eigen_results)

    c = np.zeros((c_size, c_size), dtype=float)
    for x in range(c_size):
        for y in range(c_size):
            for k in range(c_size):
                c[x, y] += valid_eigen_results[k][1][x] * valid_eigen_results[k][1][y]

                # Update progress bar
                if rich_progress is not None and task is not None:
                    rich_progress.update(task, advance=1)
    return c

def build_region_from_correlation_matrix(c_matrix, r_size):
    # Get inner matrix that is r_size by r_size big
    return c_matrix[:r_size, :r_size]

def compute_region_entanglement_entropy(r_eigenvalues):
    entropy = 0
    for value in np.abs(r_eigenvalues):
        value = np.clip(value, 1e-10, 1 - 1e-10)  # Ensures proper range
        entropy -= value * np.log(value) + (1 - value) * np.log(1 - value)
    return entropy

if __name__ == '__main__':
    hamiltonian_size = size = int(Prompt.ask("Enter the size of the Hamiltonian matrix", default="8"))
    has_periodic_boundary = Prompt.ask("Do you want to use periodic boundary conditions?", choices=["yes", "no"], default="yes") == "yes"
    is_normalized = Prompt.ask("Do you want to normalize the Hamiltonian matrix?", choices=["yes", "no"], default="yes") == "yes"

    with Progress() as progress:
        # Create Hamiltonian matrix
        task_create_hamiltonian_matrix = progress.add_task(f"Creating {hamiltonian_size}x{hamiltonian_size} Hamiltonian matrix...", total=hamiltonian_size)
        hamiltonian_matrix = build_hamiltonian_matrix(hamiltonian_size, has_periodic_boundary, is_normalized, progress, task_create_hamiltonian_matrix)

        # Get Hamiltonian matrix eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)

        # Create a list of tuple pair (eigenvalue, eigenvector) and sort it by eigenvalue
        eigen_results = sorted([(eigenvalues[i], eigenvectors[:, i]) for i in range(len(eigenvalues))], key=lambda x: x[0])

        # Only keep the eigen results with negative eigenvalues
        negative_eigen_results = [eigen_result for eigen_result in eigen_results if eigen_result[0] < 0]

        correlation_size = len(negative_eigen_results)

        task_create_correlation_matrix = progress.add_task(f"Creating {correlation_size}x{correlation_size} correlation matrix...", total=correlation_size**3)
        correlation_matrix = build_correlation_matrix(negative_eigen_results, progress, task_create_correlation_matrix)

        task_compute_region_entropies = progress.add_task(f"Computing entropies for regions of size 1 to {correlation_size}...", total=correlation_size * (correlation_size + 1) // 2)
        entropies = []
        for region_size in range(1, correlation_size + 1):
            region_matrix = build_region_from_correlation_matrix(correlation_matrix, region_size)
            region_eigenvalues = np.linalg.eigvals(region_matrix)
            entropies.append(compute_region_entanglement_entropy(region_eigenvalues))
            progress.update(task_compute_region_entropies, advance=region_size)

        # Compute the logarithm of the region sizes
        log_region_sizes = np.log(range(1, correlation_size + 1))

        # Create a figure with four subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Plot the Hamiltonian matrix
        cax1 = axs[0, 0].matshow(hamiltonian_matrix, interpolation="none")
        fig.colorbar(cax1, ax=axs[0, 0])
        axs[0, 0].set_title("Hamiltonian Matrix")

        # Plot the Correlation matrix
        cax2 = axs[0, 1].matshow(correlation_matrix, interpolation="none")
        fig.colorbar(cax2, ax=axs[0, 1])
        axs[0, 1].set_title("Correlation Matrix")

        # Plot eigenvalues with their index
        axs[1, 0].scatter(range(len(eigenvalues)), eigenvalues)
        axs[1, 0].set_title("Energy Eigenvalues")
        axs[1, 0].set_xlabel("i")
        axs[1, 0].set_ylabel("E_i")

        # Plot entropies against the logarithm of the region sizes
        axs[1, 1].scatter(log_region_sizes, entropies)
        axs[1, 1].set_title("Entanglement Entropy vs Log of Region Size")
        axs[1, 1].set_xlabel("Log of Region Size")
        axs[1, 1].set_ylabel("Entanglement Entropy")

        # Adjust layout and show the plot
        plt.tight_layout()
        plt.savefig("entanglement_entropy.png", dpi=2 ** 10)