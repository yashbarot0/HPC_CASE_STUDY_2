#include <iostream>
#include <vector>
#include <cmath>    // for abs, sqrt
#include <mpi.h>    // for MPI

using namespace std;

// Parallel matrix-vector multiplication for tridiagonal matrix 
void parallel_matvec(const vector<double>& x_local, vector<double>& Ax_local, int local_n, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double left_val = 0.0, right_val = 0.0; // Boundary values from neighbors

    // Exchange boundary values with neighbors (if they exist) 
    if (rank > 0) {
        MPI_Send(&x_local[0], 1, MPI_DOUBLE, rank-1, 0, comm); // Send left boundary value
        MPI_Recv(&left_val, 1, MPI_DOUBLE, rank-1, 0, comm, MPI_STATUS_IGNORE);  // Receive left boundary value
    }
    if (rank < size-1) {
        MPI_Send(&x_local.back(), 1, MPI_DOUBLE, rank+1, 0, comm);  // Send right boundary value
        MPI_Recv(&right_val, 1, MPI_DOUBLE, rank+1, 0, comm, MPI_STATUS_IGNORE);  // Receive right boundary value
    }

    // Compute local part of A*x
    Ax_local.resize(local_n);  // Resize output vector
    for (int i = 0; i < local_n; ++i) {  
        Ax_local[i] = -4.0 * x_local[i];  // Diagonal term
        if (i > 0 || (i == 0 && rank > 0)) {   
            Ax_local[i] += (i > 0) ? x_local[i-1] : left_val;  // Lower diagonal term
        }
        if (i < local_n-1 || (i == local_n-1 && rank < size-1)) {
            Ax_local[i] += (i < local_n-1) ? x_local[i+1] : right_val;  // Upper diagonal term
        }
    }
}

// Parallel GMRES implementation
tuple<vector<double>, vector<double>> parallel_gmres(int n, const vector<double>& b_local, int local_n, int m, double tol, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    vector<double> x(local_n, 0.0); // Initial guess x0 = 0 
    vector<double> residual_norms;

    // Compute initial residual r0 = b - A*x (x is zero, so r0 = b)
    vector<double> r_local = b_local;
    double beta_local = 0.0;
    for (double val : r_local) beta_local += val * val;
    double beta;
    MPI_Allreduce(&beta_local, &beta, 1, MPI_DOUBLE, MPI_SUM, comm);
    beta = sqrt(beta);
    double norm_b = beta;
    if (beta == 0.0) {
        residual_norms.push_back(0.0);
        return make_tuple(x, residual_norms);
    }

    // Initialize Q and H (H is replicated across all processes)
    vector<vector<double>> Q(m + 1, vector<double>(local_n));
    vector<vector<double>> H(m + 1, vector<double>(m, 0.0));

    // Q[0] = r0 / beta
    for (int i = 0; i < local_n; ++i) {
        Q[0][i] = r_local[i] / beta;
    }

    // Givens rotations and residual tracking
    vector<double> cs(m), sn(m), g(m + 1);
    g[0] = beta;
    residual_norms.push_back(beta / norm_b);

    for (int j = 0; j < m; ++j) {
        // Arnoldi iteration: Compute v = A*Q[j]
        vector<double> v_local;
        parallel_matvec(Q[j], v_local, local_n, comm);

        // Modified Gram-Schmidt orthogonalization
        for (int i = 0; i <= j; ++i) {
            // Compute H[i][j] = dot(Q[i], v)
            double local_dot = 0.0;
            for (int k = 0; k < local_n; ++k) {
                local_dot += Q[i][k] * v_local[k];
            }
            MPI_Allreduce(MPI_IN_PLACE, &local_dot, 1, MPI_DOUBLE, MPI_SUM, comm); // Reduce dot product
            H[i][j] = local_dot;

            // Orthogonalize v_local
            for (int k = 0; k < local_n; ++k) {
                v_local[k] -= H[i][j] * Q[i][k];
            }
        }

        // Compute H[j+1][j] = norm(v_local)
        double local_norm_sq = 0.0;
        for (double val : v_local) local_norm_sq += val * val;
        double norm_v;
        MPI_Allreduce(&local_norm_sq, &norm_v, 1, MPI_DOUBLE, MPI_SUM, comm);  // Reduce norm squared
        H[j+1][j] = sqrt(norm_v);

        // Normalize v_local to get Q[j+1]
        for (int k = 0; k < local_n; ++k) {
            Q[j+1][k] = v_local[k] / H[j+1][j];
        }

        // Apply previous Givens rotations to column j of H
        for (int k = 0; k < j; ++k) {
            double temp = cs[k] * H[k][j] - sn[k] * H[k+1][j];
            H[k+1][j] = sn[k] * H[k][j] + cs[k] * H[k+1][j];
            H[k][j] = temp;
        }

        // Compute new Givens rotation for H[j][j] and H[j+1][j]
        double c, s;
        if (H[j+1][j] == 0.0) {
            c = 1.0;
            s = 0.0;
        } else if (abs(H[j+1][j]) > abs(H[j][j])) {
            double tau = -H[j][j] / H[j+1][j];
            s = 1.0 / sqrt(1.0 + tau*tau);
            c = s * tau;
        } else {
            double tau = -H[j+1][j] / H[j][j];
            c = 1.0 / sqrt(1.0 + tau*tau);
            s = c * tau;
        }
        cs[j] = c;
        sn[j] = s;

        // Apply Givens rotation to H and g
        double temp = c * H[j][j] - s * H[j+1][j];
        H[j+1][j] = s * H[j][j] + c * H[j+1][j];
        H[j][j] = temp;

        temp = c * g[j] - s * g[j+1];
        g[j+1] = s * g[j] + c * g[j+1];
        g[j] = temp;

        // Update residual norm
        double residual_norm = abs(g[j+1]); // Last entry in g is the residual norm
        residual_norms.push_back(residual_norm / norm_b);  // Store relative residual norm 

        // Check stopping criterion
        if (residual_norm / norm_b < tol) {
            m = j + 1; // Adjust m to exit loop
            break;
        }
    }

    // Solve upper triangular system R*y = g[0..m-1]
    vector<double> y(m, 0.0);
    for (int i = m-1; i >= 0; --i) {
        y[i] = g[i];
        for (int k = i+1; k < m; ++k) {
            y[i] -= H[i][k] * y[k];
        }
        y[i] /= H[i][i];
    }

    // Compute x = Q*y
    x.assign(local_n, 0.0);
    for (int i = 0; i < m; ++i) {
        for (int k = 0; k < local_n; ++k) {
            x[k] += y[i] * Q[i][k];
        }
    }

    return make_tuple(x, residual_norms);
}

int main(int argc, char** argv) {  
    MPI_Init(&argc, &argv); // Initialize MPI 
    int rank, size;        // Process rank and total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get total number of processes

    int n = 256;    // Global matrix size
    int m = 128;    // Max iterations (n/2)
    double tol = 1e-6;  // Tolerance for stopping criterion (relative residual) 

    // Parse command line arguments
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) m = atoi(argv[2]);
    if (argc > 3) tol = atof(argv[3]);

    // Compute local size
    int local_n = n / size;
    if (rank == size-1) local_n = n - local_n*(size-1); // Handle remainder if n is not divisible by size

    // Initialize local b vector
    vector<double> b_local(local_n);
    for (int i = 0; i < local_n; ++i) {
        int global_i = rank * (n/size) + i;
        if (global_i < n-1) {
            b_local[i] = (global_i + 1) / static_cast<double>(n);
        } else {
            b_local[i] = 1.0;
        }
    }

    // Run GMRES solver in parallel 
    vector<double> x_local; // Local part of solution
    vector<double> res_norms;
    tie(x_local, res_norms) = parallel_gmres(n, b_local, local_n, m, tol, MPI_COMM_WORLD);  // Call parallel GMRES

    // Output results (only root process)
    if (rank == 0) {
        cout << "Residual norms:\n";
        for (double rn : res_norms) {
            cout << rn << endl;
        }
    }

    MPI_Finalize();
    return 0;
}