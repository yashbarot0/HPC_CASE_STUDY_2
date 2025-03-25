#include <iostream>
#include <vector>
#include <cmath>  // For std::abs, std::sqrt
#include <tuple>  // For returning multiple values from a function
#include <fstream>

// Function to compute A*x for the given tridiagonal matrix A
void matrix_vector_mult(const std::vector<double>& x, std::vector<double>& Ax, int n) {
    Ax.resize(n);
    for (int i = 0; i < n; ++i) {
        Ax[i] = -4.0 * x[i];
        if (i > 0) {
            Ax[i] += x[i-1];
        }
        if (i < n-1) {
            Ax[i] += x[i+1];
        }
    }
}

// Function to compute the Givens rotation for a and b
void generate_givens_rotation(double a, double b, double &c, double &s) {
    if (b == 0.0) {
        c = 1.0;
        s = 0.0;
    } else if (std::abs(b) > std::abs(a)) {
        double tau = -a / b;
        s = 1.0 / std::sqrt(1.0 + tau * tau);
        c = s * tau;
    } else {
        double tau = -b / a;
        c = 1.0 / std::sqrt(1.0 + tau * tau);
        s = c * tau;
    }
}

// Function to compute the right-hand side vector b
std::vector<double> compute_b(int n) {
    std::vector<double> b(n);
    for (int i = 0; i < n-1; ++i) {
        b[i] = (i+1) / static_cast<double>(n);
    }
    b[n-1] = 1.0;
    return b;
}

// Function to solve the system Ax = b using GMRES with m iterations 
std::tuple<std::vector<double>, std::vector<double>> gmres(int n, const std::vector<double>& b, int m) {
    std::vector<double> x(n, 0.0); // Initial guess x0 = 0 
    std::vector<double> residual_norms; 

    // Compute initial residual r0 = b - A*x (x is zero, so r0 = b)
    std::vector<double> r0 = b;
    double beta = 0.0;
    for (double val : r0) beta += val * val;
    beta = std::sqrt(beta);
    double norm_b = beta;
    if (beta == 0.0) {
        residual_norms.push_back(0.0);
        return std::make_tuple(x, residual_norms); // Return if b = 0 
    }

    // Initialize Q and H
    std::vector<std::vector<double>> Q(m + 1, std::vector<double>(n)); // Q[m+1][n]
    std::vector<std::vector<double>> H(m + 1, std::vector<double>(m)); // H[m+1][m] 

    // Q[0] = r0 / beta
    for (int i = 0; i < n; ++i) {
        Q[0][i] = r0[i] / beta;
    }

    // Initialize Givens rotations
    std::vector<double> cs(m);
    std::vector<double> sn(m);
    std::vector<double> g(m + 1, 0.0);
    g[0] = beta;

    residual_norms.push_back(beta / norm_b); // Initial residual norm

    for (int j = 0; j < m; ++j) {
        // Arnoldi iteration
        std::vector<double> v(n);
        matrix_vector_mult(Q[j], v, n); // Compute A * Q[j]

        for (int i = 0; i <= j; ++i) {
            // Compute H[i][j] = dot product of Q[i] and v
            H[i][j] = 0.0;
            for (int k = 0; k < n; ++k) {
                H[i][j] += Q[i][k] * v[k];
            }
            // Orthogonalize v
            for (int k = 0; k < n; ++k) {
                v[k] -= H[i][j] * Q[i][k];
            }
        }
        // Compute H[j+1][j] = norm(v)
        double norm_v = 0.0;
        for (double val : v) norm_v += val * val;
        H[j+1][j] = std::sqrt(norm_v);
        // Normalize v to get Q[j+1]
        for (int k = 0; k < n; ++k) {
            Q[j+1][k] = v[k] / H[j+1][j];
        }

        // Apply previous Givens rotations to column j of H
        for (int k = 0; k < j; ++k) {
            double temp = cs[k] * H[k][j] - sn[k] * H[k+1][j];
            H[k+1][j] = sn[k] * H[k][j] + cs[k] * H[k+1][j];
            H[k][j] = temp;
        }

        // Compute new Givens rotation for H[j][j] and H[j+1][j]
        double c, s;
        generate_givens_rotation(H[j][j], H[j+1][j], c, s);     // Compute Givens rotation for H[j][j] and H[j+1][j]
        cs[j] = c;
        sn[j] = s;

        // Apply the new Givens rotation to H[j][j] and H[j+1][j]
        double temp = c * H[j][j] - s * H[j+1][j];
        H[j+1][j] = s * H[j][j] + c * H[j+1][j];
        H[j][j] = temp;

        // Apply the new Givens rotation to g[j] and g[j+1]
        temp = c * g[j] - s * g[j+1];
        g[j+1] = s * g[j] + c * g[j+1];
        g[j] = temp;

        // Update the residual norm
        double residual_norm = std::abs(g[j+1]); // residual_norm = |g[j+1]|
        residual_norms.push_back(residual_norm / norm_b); // residual_norm / norm_b
    }

    // Solve the upper triangular system R y = g[0..m-1]
    std::vector<double> y(m, 0.0);
    for (int i = m - 1; i >= 0; --i) {
        y[i] = g[i];
        for (int k = i + 1; k < m; ++k) {
            y[i] -= H[i][k] * y[k];
        }
        y[i] /= H[i][i];
    }

    // Compute x = Q * y
    x.assign(n, 0.0);
    for (int i = 0; i < m; ++i) {
        for (int k = 0; k < n; ++k) {
            x[k] += y[i] * Q[i][k];
        }
    }

    return std::make_tuple(x, residual_norms); // Return the solution x and residual norms 
}

int main() {
    std::vector<int> sizes = {8, 16, 32, 64, 128, 256}; // Problem sizes to test GMRES 
    std::ofstream outfile("residual_norms.txt"); 

    for (int n : sizes) {
        int m = n / 2; // Number of iterations for GMRES
        std::vector<double> b = compute_b(n); 
        auto [x, res_norms] = gmres(n, b, m);
        
        outfile << "n = " << n << "\n"; // save residual norms to file for plotting.
        for (double rn : res_norms) {
            outfile << rn << " ";
        }
        outfile << "\n";
    }

    outfile.close();
    return 0;
}