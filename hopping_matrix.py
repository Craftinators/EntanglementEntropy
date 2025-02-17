from typing import List, Dict, Tuple
import itertools
import numpy as np
from matplotlib import pyplot as plt


def generate_states(n: int, m: int) -> List[str]:
    """
    Generate all binary strings of length n with exactly m ones.
    Each state is represented as a string of '0's and '1's.
    """
    states: List[str] = []
    for ones in itertools.combinations(range(n), m):
        state: List[str] = ['0'] * n
        for i in ones:
            state[i] = '1'
        states.append(''.join(state))
    return states


def compute_magnitude(state: str) -> int:
    """
    Compute the magnitude of a state.
    The magnitude is defined as the decimal interpretation of the binary string.
    """
    return int(state, 2)


def generate_neighbors(state: str) -> List[str]:
    """
    Generate the neighbor states for a given state.
    A neighbor is obtained by shifting a '1' one position to the left
    or right (provided that the target position exists and is '0').
    """
    n: int = len(state)
    neighbors: List[str] = []
    s: List[str] = list(state)

    for i in range(n):
        if s[i] == '1':
            # Attempt a left shift
            if i > 0 and s[i - 1] == '0':
                neighbor: List[str] = s.copy()
                neighbor[i - 1], neighbor[i] = neighbor[i], neighbor[i - 1]
                neighbors.append(''.join(neighbor))
            # Attempt a right shift
            if i < n - 1 and s[i + 1] == '0':
                neighbor: List[str] = s.copy()
                neighbor[i], neighbor[i + 1] = neighbor[i + 1], neighbor[i]
                neighbors.append(''.join(neighbor))
    return neighbors


def build_hamiltonian(states: List[str], state_to_index: Dict[str, int]) -> np.ndarray:
    """
    Build the Hamiltonian matrix H such that:
      H_{ij} = -1 if state j is a neighbor of state i, 0 otherwise.
    The matrix is symmetric and has zeros on its diagonal.
    """
    N: int = len(states)
    H: np.ndarray = np.zeros((N, N), dtype=int)
    for i, state in enumerate(states):
        for neighbor in generate_neighbors(state):
            j: int = state_to_index[neighbor]  # look up index of neighbor state
            H[i, j] = -1
    return H


def diagonalize_hamiltonian(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Diagonalize the Hamiltonian matrix H using np.linalg.eigh,
    which is optimized for symmetric matrices.

    Returns:
        eigenvalues: A 1D numpy array of eigenvalues (sorted in ascending order).
        eigenvectors: A 2D numpy array where each column is an eigenvector.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    return eigenvalues, eigenvectors

def plot_hamiltonian_analysis(H: np.ndarray,
                              eigenvalues: np.ndarray,
                              eigenvectors: np.ndarray,
                              rho: np.ndarray) -> None:
    """
    Plots the Hamiltonian matrix, its diagonalized form, the eigenvalue spectrum,
    and the density matrix in a single figure.

    Args:
        H (np.ndarray): The Hamiltonian matrix.
        eigenvalues (np.ndarray): Eigenvalues of H.
        eigenvectors (np.ndarray): Eigenvectors of H.
        rho (np.ndarray): The density matrix.
    """
    H_diag: np.ndarray = np.diag(eigenvalues)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Plot the original Hamiltonian matrix
    im1 = axes[0, 0].imshow(H, cmap='viridis', interpolation='none')
    axes[0, 0].set_title(r"Original Hamiltonian $\mathbf{H}$", fontsize=14)
    fig.colorbar(im1, ax=axes[0, 0])

    # Plot the diagonalized (eigenvalue) matrix
    im2 = axes[0, 1].imshow(H_diag, cmap='viridis', interpolation='none')
    axes[0, 1].set_title(r"Diagonalized $\mathbf{H}$", fontsize=14)
    fig.colorbar(im2, ax=axes[0, 1])

    # Plot the eigenvalue spectrum with markers only (no lines) and no grid
    axes[1, 0].plot(eigenvalues, 'o', color='lightblue', label='Eigenvalues')
    axes[1, 0].set_title(r"Eigenvalue Spectrum", fontsize=14)
    axes[1, 0].set_xlabel("Index", fontsize=12)
    axes[1, 0].set_ylabel(r"$\lambda_i$", fontsize=12)
    axes[1, 0].legend()

    # Plot the density matrix
    im3 = axes[1, 1].imshow(rho, cmap='viridis', interpolation='none')
    axes[1, 1].set_title(r"Density Matrix $\mathbf{\rho}$", fontsize=14)
    fig.colorbar(im3, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()

def construct_ground_state(ground_state_vector: np.ndarray) -> str:
    """
    Construct a string representation of the ground state |psi_0> in terms of the basis states |theta_k>,
    where ground_state_vector contains the coefficients alpha_{0,k}.

    Args:
        ground_state_vector (np.ndarray): The eigenvector corresponding to the smallest eigenvalue.

    Returns:
        A LaTeX string representing \ket{\psi_0} = \sum_k alpha_{0,k} \ket{\theta_k}.
    """
    terms: List[str] = []
    for k, coeff in enumerate(ground_state_vector):
        formatted_coeff = f"{coeff:.3f}"
        term = rf"{formatted_coeff} \ket{{\theta_{{{k}}}}}"
        terms.append(term)

    # Join the terms with ' + ' and format LaTeX-style output
    psi0_str = r"$\ket{\psi_0} = " + " + ".join(terms) + "$"
    return psi0_str

def main() -> None:
    # Define system parameters
    n: int = 8  # total number of sites
    m: int = 4  # number of occupied sites

    # Generate all states with exactly m ones and sort them by magnitude
    states: List[str] = generate_states(n, m)
    states.sort(key=lambda s: compute_magnitude(s))

    # Create a mapping from state to its index for quick lookup
    state_to_index: Dict[str, int] = {state: k for k, state in enumerate(states)}

    # Build the Hamiltonian matrix
    H: np.ndarray = build_hamiltonian(states, state_to_index)

    # Diagonalize H (eigenvalues are sorted in ascending order)
    eigenvalues, eigenvectors = diagonalize_hamiltonian(H)

    # Print the ground state information (smallest eigenvalue)
    ground_state_energy = eigenvalues[0]

    ground_state_vector: np.ndarray = eigenvectors[:, 0]  # first column is the ground state eigenvector

    rho = np.outer(ground_state_vector, ground_state_vector.conj())
    # Display rho as matrix in matplotlib

    is_hermitian = np.allclose(rho, rho.conj().T)
    print("Hermitian:", is_hermitian)
    trace_rho = np.trace(rho)
    print("Trace:", trace_rho)
    is_trace_one = np.isclose(trace_rho, 1.0)
    print("Trace equals 1:", is_trace_one)
    eigenvalues = np.linalg.eigvals(rho)
    print("Eigenvalues:", eigenvalues)
    is_psd = np.all(eigenvalues >= -1e-10)  # tolerance to account for numerical error
    print("Positive semidefinite:", is_psd)
    is_idempotent = np.allclose(rho @ rho, rho)
    print("Idempotency (rho^2 = rho):", is_idempotent)


    # Plot the Hamiltonian analysis (original H, diagonalized H, and eigenvalue spectrum)
    plot_hamiltonian_analysis(H, eigenvalues, eigenvectors, rho)


if __name__ == "__main__":
    main()
