#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <mpi.h>
#include <omp.h> // <-- MENAMBAHKAN LINE BARU: Header resmi OpenMP untuk mengaktifkan fungsi omp

// Helper to map 2D coordinates to 1D array index
inline int IDX(int i, int j, int cols) { return i * cols + j; }

// Parallel MPI-IO File Reader
void read_matrix_block(MPI_Comm comm, const char *filename, int N, int gridDim, int my_row, int my_col, std::vector<double> &local_matrix)
{
    MPI_File fh;
    // Open the binary file in Read-Only mode
    int err = MPI_File_open(comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (err != MPI_SUCCESS)
    {
        int myId;
        MPI_Comm_rank(comm, &myId);
        if (!myId)
            std::cout << "ERROR: Could not open file " << filename << std::endl;
        MPI_Abort(comm, 1);
    }

    int block_size = N / gridDim;

    // A 2D block is not contiguous in a 1D file. We must read it row by row.
    for (int i = 0; i < block_size; ++i)
    {
        int global_row = my_row * block_size + i;
        int global_col_start = my_col * block_size;

        // Calculate the exact byte offset for this row chunk
        MPI_Offset offset = (global_row * N + global_col_start) * sizeof(double);

        // Jump to the offset and read 'block_size' elements into our local array
        MPI_File_read_at(fh, offset, &local_matrix[i * block_size], block_size, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }

    MPI_File_close(&fh);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int numP, myId;
    MPI_Comm_size(MPI_COMM_WORLD, &numP);
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);

    // --- UPDATED: Require 4 arguments (Program, N, Path_A, Path_B) ---
    if (argc < 4)
    {
        if (!myId)
        {
            std::cout << "Usage: mpiexec -n <cores> summa.exe <N> <path_to_matrix_A> <path_to_matrix_B>\n";
            std::cout << "Example: mpiexec -n 4 summa.exe 1000 data_matrix/matrix_A_1000.bin data_matrix/matrix_B_1000.bin\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int N = atoi(argv[1]);
    std::string filenameA = argv[2]; // Path to first matrix
    std::string filenameB = argv[3]; // Path to second matrix

    // Topology checks
    int gridDim = std::round(std::sqrt(numP));
    if (gridDim * gridDim != numP)
    {
        if (!myId)
            std::cout << "ERROR: The number of processes must be a perfect square.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (N % gridDim != 0)
    {
        if (!myId)
            std::cout << "ERROR: Matrix size N must be divisible by sqrt(numP).\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Create Hardware-Aware Cartesian Topology
    int dims[2] = {gridDim, gridDim};
    int periods[2] = {0, 0};
    MPI_Comm cart_comm, row_comm, col_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, myId, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];

    // Extract Row and Column Sub-Communicators
    int remain_dims[2];
    remain_dims[0] = 0;
    remain_dims[1] = 1;
    MPI_Cart_sub(cart_comm, remain_dims, &row_comm);

    remain_dims[0] = 1;
    remain_dims[1] = 0;
    MPI_Cart_sub(cart_comm, remain_dims, &col_comm);

    // Memory Allocation
    int block_size = N / gridDim;
    int elements_per_block = block_size * block_size;

    std::vector<double> myA(elements_per_block);
    std::vector<double> myB(elements_per_block);
    std::vector<double> myC(elements_per_block, 0.0);

    std::vector<double> buffA(elements_per_block);
    std::vector<double> buffB(elements_per_block);

    // --- Read data directly from the binary files using the provided paths ---
    read_matrix_block(MPI_COMM_WORLD, filenameA.c_str(), N, gridDim, my_row, my_col, myA);
    read_matrix_block(MPI_COMM_WORLD, filenameB.c_str(), N, gridDim, my_row, my_col, myB);

    // Synchronize before timing the main math loop
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // The SUMMA Main Loop
    for (int step = 0; step < gridDim; ++step)
    {

        // Broadcast A along rows
        if (my_col == step)
            buffA = myA;
        MPI_Bcast(buffA.data(), elements_per_block, MPI_DOUBLE, step, row_comm);

        // Broadcast B along columns
        if (my_row == step)
            buffB = myB;
        MPI_Bcast(buffB.data(), elements_per_block, MPI_DOUBLE, step, col_comm);

        // --- MENAMBAHKAN LINE BARU: SUNTIKAN DIREKTIF PARALELISASI OPENMP ---
        // #pragma omp parallel for akan memparalelkan loop terluar 'i' ke dalam thread-thread CPU.
        // collapse(2) menggabungkan loop 'i' dan 'j' menjadi satu ruang iterasi besar agar pembagian beban kerja ke semua core CPU merata.
        // schedule(dynamic) mendistribusikan beban secara dinamis, sangat bagus jika performa tiap core sedikit berbeda.
        #pragma omp parallel for collapse(2) schedule(dynamic)
        // Local Computation Block
        for (int i = 0; i < block_size; i++)
        {
            for (int j = 0; j < block_size; j++)
            {
                for (int l = 0; l < block_size; l++)
                {
                    myC[IDX(i, j, block_size)] += buffA[IDX(i, l, block_size)] * buffB[IDX(l, j, block_size)];
                }
            }
        }
    }

    double end_time = MPI_Wtime();

    if (!myId)
    {
        std::cout << "SUMMA completed for " << N << "x" << N << " matrix.\n";
        // --- MENAMBAHKAN LINE BARU: Menampilkan informasi jumlah thread OpenMP yang aktif di terminal ---
        std::cout << "Each MPI process utilized " << omp_get_max_threads() << " OpenMP threads for local computation.\n";
        std::cout << "Time with " << numP << " processes: " << end_time - start_time << " seconds.\n";
    }

    // Clean up
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();

    return 0;
}
