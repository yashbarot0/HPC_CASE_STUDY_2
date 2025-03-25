#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace Eigen;
using namespace std;

void arnoldi(const MatrixXd& A, const VectorXd& u, int m, MatrixXd& Q, MatrixXd& H) {
    int n = A.rows();
    Q = MatrixXd::Zero(n, m + 1);
    H = MatrixXd::Zero(m + 1, m);
    
    // Normalize the initial vector
    VectorXd q = u.normalized();
    Q.col(0) = q;
    
    for (int j = 0; j < m; ++j) {
        // Apply matrix A to the current basis vector
        VectorXd v = A * Q.col(j);
        
        // Orthogonalize v against all previous q's
        for (int i = 0; i <= j; ++i) {
            H(i, j) = Q.col(i).dot(v);
            v -= H(i, j) * Q.col(i);
        }
        
        // Compute the next Hessenberg element
        H(j + 1, j) = v.norm();
        
        // Check for breakdown (H(j+1,j) is zero)
        if (H(j + 1, j) == 0) {
            cout << "Arnoldi iteration broke down at step " << j + 1 << endl;
            return;
        }
        
        // Normalize to get the next basis vector
        Q.col(j + 1) = v / H(j + 1, j);
    }
}

int main() {
    // Define matrix A from the case study
    MatrixXd A(10, 10);
    A << 3, 8, 7, 3, 3, 7, 2, 3, 4, 8,
         5, 4, 1, 6, 9, 8, 3, 7, 1, 9,
         3, 6, 9, 4, 8, 6, 5, 6, 6, 6,
         5, 3, 4, 7, 4, 9, 2, 3, 5, 1,
         4, 4, 2, 1, 7, 4, 2, 2, 4, 5,
         4, 2, 8, 6, 6, 5, 2, 1, 1, 2,
         2, 8, 9, 5, 2, 9, 4, 7, 3, 3,
         9, 3, 2, 2, 7, 3, 4, 8, 7, 7,
         9, 1, 9, 3, 3, 1, 2, 7, 7, 1,
         9, 3, 2, 2, 6, 4, 4, 7, 3, 5;
    
    // Define vector u (from x in the case study)
    VectorXd u(10);
    u << +0.757516242460009,
          +2.734057963614329,
          -0.555605907443403,
          +1.144284746786790,
          +0.645280108318073,
          -0.085488474462339,
          -0.623679022063185,
          -0.465240896342741,
          +2.382909057772335,
          -0.120465395885881;
    
    int m = 9;  // We need Q9
    MatrixXd Q, H;
    arnoldi(A, u, m, Q, H);
    
    // Output Q9 (the first 10 columns of Q, since m=9 gives Q with 10 columns)
    cout << "Q9 matrix:" << endl;
    cout << Q << endl;
    
    // Save Q9 to a file for verification
    ofstream outfile("Q9.txt");
    if (outfile.is_open()) {
        outfile << Q << endl;
        outfile.close();
    } else {
        cout << "Unable to open file for writing Q9" << endl;
    }
    
    return 0;
}