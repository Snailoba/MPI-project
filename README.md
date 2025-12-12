# Distributed Matrix Multiplication (MPI, C)

This project implements parallel matrix multiplication `C = A × B` in C using MPI with block-row partitioning. It uses collective operations (`MPI_Bcast`, `MPI_Scatterv`, `MPI_Gatherv`) and reports execution time and speedup versus serial.

## Matrix File Format

Each matrix file is plain text:

- First line: `rows cols`
- Following lines: `rows*cols` numbers in row-major order (whitespace-separated). Example for a 2×3 matrix:

```
2 3
1 2 3 4 5 6
```

Place your matrices as `matrix_a.txt` and `matrix_b.txt` in the workspace.

## Build

You need an MPI implementation and a C compiler.

### Windows Options

- MS-MPI (native Windows): Install MS-MPI and use `mpiexec.exe` with `cl` or `gcc`.
- WSL2 + OpenMPI (recommended): Install Ubuntu in WSL2, install `build-essential` and `openmpi-bin`.

### WSL2 + OpenMPI (recommended)

Inside Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential openmpi-bin libopenmpi-dev
mpicc -O2 -o mpi_matmul mpi_matmul.c
```

### MSYS2 / MinGW + OpenMPI (alternative)

Install MSYS2, `mingw-w64-x86_64-openmpi` and build with `mpicc` similarly.

## Run

Parallel run (P processes):

```bash
mpirun -np 4 ./mpi_matmul matrix_a.txt matrix_b.txt --serial
```

- `--serial` also runs a serial multiply on rank 0 and prints speedup.
- Output includes parallel runtime, optional serial runtime, speedup, and the resulting `C` matrix.

Example output:

```
Parallel runtime (max rank time): 0.012345 s
Serial runtime: 0.045678 s
Speedup (serial / parallel): 3.701
C 4 4
...values...
```

## Notes

- Ensure dimensions match: `A[rA×cA]` times `B[rB×cB]` requires `cA == rB`.
- For large matrices, consider redirecting `stdout` to a file.
- Change process count (`-np`) to match your CPU cores.
