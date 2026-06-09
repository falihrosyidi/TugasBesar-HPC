#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>

// Helper to map 2D coordinates to 1D array index
inline int IDX(int i, int j, int cols) { return i * cols + j; }

// Parallel Data Generation: Each process calculates its global coordinates
// and initializes its local block independently. No Rank 0 bottleneck.
void initialize_local_matrices(int my_row, int my_col, int block_size, 
                               std::vector<double>& myA, std::vector<double>& myB) {
    int global_row_start = my_row * block_size;
    int global_col_start = my_col * block_size;

    for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
            // Replicating the previous checkerboard pattern, but generated fully in parallel
            int global_i = global_row_start + i;
            int global_j = global_col_start + j;
            
            myA[IDX(i, j, block_size)] = (global_i + global_j) % 2;
            myB[IDX(i, j, block_size)] = (global_i + global_j) % 2;
        }
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int numP, myId;
    MPI_Comm_size(MPI_COMM_WORLD, &numP);
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);

    // Ensure we have arguments for matrix size N
    if (argc < 2) {
        if (!myId) std::cout << "Usage: ./summa N (where N is the matrix size N x N)" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int N = atoi(argv[1]);

    // Calculate grid dimensions
    int gridDim = std::round(std::sqrt(numP));
    if (gridDim * gridDim != numP) {
        if (!myId) std::cout << "ERROR: The number of processes must be a perfect square." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (N % gridDim != 0) {
        if (!myId) std::cout << "ERROR: Matrix size N must be divisible by sqrt(numP)." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 1. Create Hardware-Aware Cartesian Topology
    int dims[2] = {gridDim, gridDim};
    int periods[2] = {0, 0}; // Non-periodic grid
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, myId, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];

    // 2. Extract Row and Column Sub-Communicators
    MPI_Comm row_comm, col_comm;
    int remain_dims[2];

    remain_dims[0] = 0; remain_dims[1] = 1; // Keep row, vary col
    MPI_Cart_sub(cart_comm, remain_dims, &row_comm);

    remain_dims[0] = 1; remain_dims[1] = 0; // Keep col, vary row
    MPI_Cart_sub(cart_comm, remain_dims, &col_comm);

    // 3. Fully Distributed Memory Allocation
    int block_size = N / gridDim;
    int elements_per_block = block_size * block_size;

    std::vector<double> myA(elements_per_block);
    std::vector<double> myB(elements_per_block);
    std::vector<double> myC(elements_per_block, 0.0); // Accumulator, starts at 0

    std::vector<double> buffA(elements_per_block);
    std::vector<double> buffB(elements_per_block);

    initialize_local_matrices(my_row, my_col, block_size, myA, myB);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // 4. The SUMMA Main Loop
    for (int step = 0; step < gridDim; ++step) {
        
        // --- Row Broadcast of A ---
        if (my_col == step) {
            buffA = myA; // I am the source for this step, copy my data to the broadcast buffer
        }
        MPI_Bcast(buffA.data(), elements_per_block, MPI_DOUBLE, step, row_comm);

        // --- Column Broadcast of B ---
        if (my_row == step) {
            buffB = myB;
        }
        MPI_Bcast(buffB.data(), elements_per_block, MPI_DOUBLE, step, col_comm);

        // --- Local Computation (Retained as requested) ---
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                for (int l = 0; l < block_size; l++) {
                    myC[IDX(i, j, block_size)] += buffA[IDX(i, l, block_size)] * buffB[IDX(l, j, block_size)];
                }
            }
        }
    }

    double end_time = MPI_Wtime();

    if (!myId) {
        std::cout << "SUMMA completed for " << N << "x" << N << " matrix." << std::endl;
        std::cout << "Time with " << numP << " processes: " << end_time - start_time << " seconds." << std::endl;
    }

    // Clean up
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();

    return 0;
}