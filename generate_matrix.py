import numpy as np
import os

def generate_matrices(N):
    folder = "data_matrix"
    os.makedirs(folder, exist_ok=True)
    
    print(f"Generating random matrices and an identity matrix of size {N}x{N}")
    print(f"and saving to '{folder}/'...\n")
    
    # 1. Generate random matrices A and B (Values between 0.0 and 1.0)
    matrix_A = np.random.rand(N, N).astype(np.float64)
    matrix_B = np.random.rand(N, N).astype(np.float64)
    
    # 2. Generate the Identity Matrix (I)
    # np.eye(N) creates an NxN matrix with 1s on the diagonal and 0s elsewhere
    matrix_I = np.eye(N).astype(np.float64)
    
    # 3. Save as Binary (For your C++ MPI-IO reader)
    matrix_A.tofile(f"{folder}/matrix_A_{N}.bin")
    matrix_B.tofile(f"{folder}/matrix_B_{N}.bin")
    matrix_I.tofile(f"{folder}/matrix_I_{N}.bin")
    print(f"Saved binary files: \n -{folder}/matrix_A_{N}.bin, \n -{folder}/matrix_B_{N}.bin, \n -{folder}/matrix_I_{N}.bin")
    
    # 4. Save as CSV (For your visual inspection)
    np.savetxt(f"{folder}/matrix_A_{N}.csv", matrix_A, delimiter=",", fmt="%.4f")
    np.savetxt(f"{folder}/matrix_B_{N}.csv", matrix_B, delimiter=",", fmt="%.4f")
    # Using %.0f for the identity matrix CSV since it only contains whole 1s and 0s
    np.savetxt(f"{folder}/matrix_I_{N}.csv", matrix_I, delimiter=",", fmt="%.0f") 
    print(f"Saved CSV files: \n -{folder}/matrix_A_{N}.csv, \n -{folder}/matrix_B_{N}.csv, \n -{folder}/matrix_I_{N}.csv")

if __name__ == "__main__":
    # Change this to whatever size you are testing with
    MATRIX_SIZE = 1000 
    generate_matrices(MATRIX_SIZE)