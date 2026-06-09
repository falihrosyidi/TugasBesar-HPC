
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cstdlib>

inline int IDX(int i, int j, int cols)
{
    return i * cols + j;
}

// Membaca matrix penuh dari file .bin
void read_matrix(const char *filename, int N, std::vector<double> &matrix)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file)
    {
        std::cerr << "ERROR: Could not open file " << filename << std::endl;
        std::exit(1);
    }

    file.read(reinterpret_cast<char *>(matrix.data()), N * N * sizeof(double));

    if (!file)
    {
        std::cerr << "ERROR: Failed to read full matrix from " << filename << std::endl;
        std::exit(1);
    }

    file.close();
}

// Perkalian matrix serial biasa
void matrix_multiply_serial(
    const std::vector<double> &A,
    const std::vector<double> &B,
    std::vector<double> &C,
    int N)
{
    for (int i = 0; i < N; ++i)
    {
        for (int k = 0; k < N; ++k)
        {
            double rA = A[IDX(i, k, N)];

            for (int j = 0; j < N; ++j)
            {
                C[IDX(i, j, N)] += rA * B[IDX(k, j, N)];
            }
        }
    }
}

// Checksum sederhana untuk validasi ringan
double compute_checksum(const std::vector<double> &C)
{
    double sum = 0.0;

    for (double value : C)
    {
        sum += value;
    }

    return sum;
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cout << "Usage: ./serial_baseline <N> <path_to_matrix_A> <path_to_matrix_B>\n";
        std::cout << "Example: ./serial_baseline 1000 data_matrix/matrix_A_1000.bin data_matrix/matrix_B_1000.bin\n";
        return 1;
    }

    int N = std::atoi(argv[1]);
    std::string filenameA = argv[2];
    std::string filenameB = argv[3];

    std::cout << "Serial baseline matrix multiplication\n";
    std::cout << "Matrix size: " << N << " x " << N << std::endl;

    // Alokasi memori
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N, 0.0);

    read_matrix(filenameA.c_str(), N, A);
    read_matrix(filenameB.c_str(), N, B);

    // Mulai timing hanya untuk proses komputasi
    auto start_time = std::chrono::high_resolution_clock::now();

    matrix_multiply_serial(A, B, C, N);

    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end_time - start_time;

    double checksum = compute_checksum(C);

    std::cout << "Computation completed.\n";
    std::cout << "Execution time: " << elapsed.count() << " seconds.\n";
    std::cout << "Checksum: " << checksum << std::endl;

    return 0;
}