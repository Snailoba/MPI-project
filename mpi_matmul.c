#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

/*
Distributed Matrix Multiplication using block-row partitioning.
- Reads matrices A and B from text files where first line is "rows cols",
  followed by rows*cols numbers in row-major order.
- Uses MPI_Bcast to distribute matrix B (and metadata) to all ranks.
- Uses MPI_Scatterv to distribute contiguous row blocks of A.
- Each rank computes its block of C = A x B.
- Uses MPI_Gatherv to assemble the final C on rank 0.
- Measures parallel runtime; optionally runs serial multiply on rank 0 to report speedup.

Usage:
  mpirun -np <P> ./mpi_matmul <matrix_a.txt> <matrix_b.txt> [--serial]

Notes:
- Matrices must be compatible: A[rA x cA] * B[rB x cB] with cA == rB.
- Output C is written to stdout on rank 0; suppressable via a flag if desired.
*/

static int read_matrix(const char *path, int *rows, int *cols, double **data) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Error: cannot open %s\n", path); return 0; }
    if (fscanf(f, "%d %d", rows, cols) != 2) {
        fprintf(stderr, "Error: invalid header in %s\n", path); fclose(f); return 0;
    }
    size_t n = (size_t)(*rows) * (size_t)(*cols);
    *data = (double*)malloc(n * sizeof(double));
    if (!*data) { fprintf(stderr, "Error: malloc failed\n"); fclose(f); return 0; }
    for (size_t i = 0; i < n; ++i) {
        if (fscanf(f, "%lf", &((*data)[i])) != 1) {
            fprintf(stderr, "Error: not enough elements in %s\n", path);
            free(*data); *data = NULL; fclose(f); return 0;
        }
    }
    fclose(f);
    return 1;
}

static void write_matrix_stdout(const char *name, int rows, int cols, const double *data) {
    printf("%s %d %d\n", name, rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%g%s", data[(size_t)i * cols + j], (j + 1 == cols) ? "" : " ");
        }
        printf("\n");
    }
}

static void serial_matmul(int rA, int cA, int rB, int cB, const double *A, const double *B, double *C) {
    // Assume cA == rB
    for (int i = 0; i < rA; ++i) {
        for (int j = 0; j < cB; ++j) {
            double sum = 0.0;
            for (int k = 0; k < cA; ++k) {
                sum += A[(size_t)i * cA + k] * B[(size_t)k * cB + j];
            }
            C[(size_t)i * cB + j] = sum;
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc < 3) {
        if (world_rank == 0) {
            fprintf(stderr, "Usage: mpirun -np P ./mpi_matmul <matrix_a.txt> <matrix_b.txt> [--serial]\n");
        }
        MPI_Finalize();
        return 1;
    }

    const char *pathA = argv[1];
    const char *pathB = argv[2];
    int run_serial = 0;
    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--serial") == 0) run_serial = 1;
    }

    int rA = 0, cA = 0, rB = 0, cB = 0;
    double *A = NULL; // only fully loaded on rank 0
    double *B = NULL; // broadcast to all ranks

    double *C_full = NULL; // only on rank 0

    // Rank 0 reads matrices
    if (world_rank == 0) {
        if (!read_matrix(pathA, &rA, &cA, &A)) {
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        if (!read_matrix(pathB, &rB, &cB, &B)) {
            free(A);
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
        if (cA != rB) {
            fprintf(stderr, "Error: dimension mismatch A[%d x %d] * B[%d x %d]\n", rA, cA, rB, cB);
            free(A); free(B);
            MPI_Abort(MPI_COMM_WORLD, 4);
        }
    }

    // Broadcast dimensions
    MPI_Bcast(&rA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rB, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cB, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate B on non-root and broadcast data
    if (world_rank != 0) {
        B = (double*)malloc((size_t)rB * (size_t)cB * sizeof(double));
        if (!B) { fprintf(stderr, "Rank %d: malloc failed for B\n", world_rank); MPI_Abort(MPI_COMM_WORLD, 5); }
    }
    MPI_Bcast(B, rB * cB, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute row distribution for A
    int *sendcounts = (int*)malloc(world_size * sizeof(int));
    int *displs = (int*)malloc(world_size * sizeof(int));
    int base = rA / world_size;
    int rem = rA % world_size;
    int disp = 0;
    for (int p = 0; p < world_size; ++p) {
        int rows_p = base + (p < rem ? 1 : 0);
        sendcounts[p] = rows_p * cA; // number of elements
        displs[p] = disp;
        disp += sendcounts[p];
    }

    // Allocate local A block and C block
    int local_elems_A = sendcounts[world_rank];
    int local_rows = local_elems_A / cA;
    double *A_local = (double*)malloc((size_t)local_elems_A * sizeof(double));
    double *C_local = (double*)malloc((size_t)local_rows * (size_t)cB * sizeof(double));
    if (!A_local || !C_local) {
        fprintf(stderr, "Rank %d: malloc failed for local matrices\n", world_rank);
        free(sendcounts); free(displs);
        if (A_local) free(A_local);
        if (C_local) free(C_local);
        free(B);
        if (world_rank == 0) { if (A) free(A); if (C_full) free(C_full); }
        MPI_Abort(MPI_COMM_WORLD, 6);
    }

    // Scatter rows of A
    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
                 A_local, local_elems_A, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Timing parallel section
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Compute local C = A_local x B
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < cB; ++j) {
            double sum = 0.0;
            for (int k = 0; k < cA; ++k) {
                sum += A_local[(size_t)i * cA + k] * B[(size_t)k * cB + j];
            }
            C_local[(size_t)i * cB + j] = sum;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double local_time = t1 - t0;

    // Gather C blocks to root
    if (world_rank == 0) {
        C_full = (double*)malloc((size_t)rA * (size_t)cB * sizeof(double));
        if (!C_full) { fprintf(stderr, "Rank 0: malloc failed for C_full\n"); MPI_Abort(MPI_COMM_WORLD, 7); }
    }

    // counts/displs for C are analogous to A but scaled by cB
    int *recvcountsC = (int*)malloc(world_size * sizeof(int));
    int *displsC = (int*)malloc(world_size * sizeof(int));
    disp = 0;
    for (int p = 0; p < world_size; ++p) {
        int rows_p = sendcounts[p] / cA;
        recvcountsC[p] = rows_p * cB;
        displsC[p] = disp;
        disp += recvcountsC[p];
    }

    MPI_Gatherv(C_local, local_rows * cB, MPI_DOUBLE,
                C_full, recvcountsC, displsC, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Reduce max(local_time) to get parallel runtime (dominant rank)
    double parallel_runtime = 0.0;
    MPI_Reduce(&local_time, &parallel_runtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Serial run and speedup
    double serial_runtime = 0.0;
    if (world_rank == 0 && run_serial) {
        double *C_serial = (double*)malloc((size_t)rA * (size_t)cB * sizeof(double));
        double tS0 = MPI_Wtime();
        serial_matmul(rA, cA, rB, cB, A, B, C_serial);
        double tS1 = MPI_Wtime();
        serial_runtime = tS1 - tS0;
        // Optionally verify correctness by comparing C_full and C_serial
        if (C_full) {
            int mismatch = 0;
            for (size_t i = 0; i < (size_t)rA * (size_t)cB; ++i) {
                double diff = C_full[i] - C_serial[i];
                if (diff < -1e-9 || diff > 1e-9) { mismatch = 1; break; }
            }
            if (mismatch) {
                fprintf(stderr, "Warning: parallel result differs from serial result beyond tolerance.\n");
            }
        }
        free(C_serial);
    }

    if (world_rank == 0) {
        // Print runtimes and speedup
        printf("Parallel runtime (max rank time): %.6f s\n", parallel_runtime);
        if (run_serial) {
            printf("Serial runtime: %.6f s\n", serial_runtime);
            if (parallel_runtime > 0.0) {
                printf("Speedup (serial / parallel): %.3f\n", serial_runtime / parallel_runtime);
            }
        }
        // Output matrix C
        if (C_full) {
            write_matrix_stdout("C", rA, cB, C_full);
        }
    }

    // Cleanup
    free(sendcounts); free(displs);
    free(recvcountsC); free(displsC);
    free(A_local); free(C_local);
    free(B);
    if (world_rank == 0) { free(A); free(C_full); }

    MPI_Finalize();
    return 0;
}
