# Compiler settings and targets for the Arnoldi and GMRES implementations in C++. 
CXX = g++  # Serial compiler
MPI_CXX = mpic++  # MPI compiler
CXXFLAGS = -Wall -std=c++11 -O2

# Targets 
all: arnoldi serial parallel

arnoldi: arnoldi.cpp  # Arnoldi implementation
    $(CXX) $(CXXFLAGS) -o arnoldi arnoldi.cpp

serial: serial_GMRES.cpp  # Serial GMRES implementation
    $(CXX) $(CXXFLAGS) -o serial_GMRES serial_GMRES.cpp

parallel: parallel_gmres.cpp # Parallel GMRES implementation
    $(MPI_CXX) -o parallel_gmres parallel_gmres.cpp

run_parallel: parallel_gmres # Run parallel GMRES with 4 processes and a 256x128 matrix with a tolerance of 1e-6 
    mpirun -n 4 ./parallel_gmres 256 128 1e-6  

clean: 
    rm -f arnoldi serial_GMRES parallel_gmres  
    rm -f arnoldi_Q9.txt residual_norms.txt
    rm -f *.o *~   