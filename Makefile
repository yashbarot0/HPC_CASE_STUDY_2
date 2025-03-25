# Compiler settings
CXX = g++
MPI_CXX = mpic++
CXXFLAGS = -Wall -std=c++11 -O2

# Targets
all: arnoldi serial parallel

arnoldi: arnoldi.cpp
    $(CXX) $(CXXFLAGS) -o arnoldi arnoldi.cpp

serial: serial_gmres.cpp
    $(CXX) $(CXXFLAGS) -o serial_gmres serial_gmres.cpp

parallel: parallel_gmres.cpp
    $(MPI_CXX) $(CXXFLAGS) -o parallel_gmres parallel_gmres.cpp

plot: serial_gmres
    ./serial_gmres && python3 plot_convergence.py

run_parallel: parallel_gmres
    mpirun -n 4 ./parallel_gmres 256 128 1e-6

clean:
    rm -f arnoldi serial_gmres parallel_gmres
    rm -f Q9.txt residual_norms.txt gmres_convergence.png
    rm -f *.o *~