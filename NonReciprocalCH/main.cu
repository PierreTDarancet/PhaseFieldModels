#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cufft.h>
#include <cuda_runtime.h>
#include <hdf5.h>
#include <map>
#include <string>
#include <sstream>
#include <iomanip>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// --- CUDA Kernels ---
__global__ void init_fourier(double* q2, double* E, double* E1, int nx, int ny, double dx, double dt, double kappa) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < nx && iy < ny) {
        int idx = iy * nx + ix;
        double kx = (ix < nx / 2 ? ix : ix - nx) * 2.0 * M_PI / (nx * dx);
        double ky = (iy < ny / 2 ? iy : iy - ny) * 2.0 * M_PI / (ny * dx);
        double q2_val = kx * kx + ky * ky;
        double L = -kappa * q2_val * q2_val;
        
        q2[idx] = q2_val;
        E[idx] = exp(L * dt);
        E1[idx] = (L != 0.0) ? (exp(L * dt) - 1.0) / L : dt;
    }
}

__global__ void calc_nonlinear(cufftDoubleComplex* phi1, cufftDoubleComplex* phi2,
                               cufftDoubleComplex* N1, cufftDoubleComplex* N2,
                               int nx, int ny, double scale, double chi, double chi_prime, 
                               double alpha, double c11, double c12, double c21, double c22) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < nx && iy < ny) {
        int i = iy * nx + ix;
        
        double p1 = phi1[i].x * scale;
        double p2 = phi2[i].x * scale;
        phi1[i].x = p1; phi1[i].y = 0.0;
        phi2[i].x = p2; phi2[i].y = 0.0;

        double bulk1 = 2.0 * (p1 - c11) * (p1 - c12) * (2.0 * p1 - c11 - c12);
        double int1 = chi * p2 + 2.0 * chi_prime * p1 * (p2 * p2);
        
        double bulk2 = 2.0 * (p2 - c21) * (p2 - c22) * (2.0 * p2 - c21 - c22);
        double int2 = chi * p1 + 2.0 * chi_prime * p2 * (p1 * p1);

        N1[i].x = bulk1 + int1 + (alpha * p2); N1[i].y = 0.0;
        N2[i].x = bulk2 + int2 - (alpha * p1); N2[i].y = 0.0;
    }
}

__global__ void etd_step(cufftDoubleComplex* phi1_hat, cufftDoubleComplex* phi2_hat,
                         cufftDoubleComplex* N1_hat, cufftDoubleComplex* N2_hat,
                         double* q2, double* E, double* E1, int nx, int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < nx && iy < ny) {
        int i = iy * nx + ix;
        
        double q2_val = q2[i];
        double n1_r = -q2_val * N1_hat[i].x; double n1_i = -q2_val * N1_hat[i].y;
        double n2_r = -q2_val * N2_hat[i].x; double n2_i = -q2_val * N2_hat[i].y;

        phi1_hat[i].x = phi1_hat[i].x * E[i] + n1_r * E1[i];
        phi1_hat[i].y = phi1_hat[i].y * E[i] + n1_i * E1[i];
        phi2_hat[i].x = phi2_hat[i].x * E[i] + n2_r * E1[i];
        phi2_hat[i].y = phi2_hat[i].y * E[i] + n2_i * E1[i];
    }
}

void get_flux(int x, int y, int nx, int ny, double dx, cufftDoubleComplex* h_phi1, cufftDoubleComplex* h_phi2, double &jx, double &jy) {
    int xp = (x + 1) % nx; int xm = (x - 1 + nx) % nx;
    int yp = (y + 1) % ny; int ym = (y - 1 + ny) % ny;
    
    double p1 = h_phi1[y * nx + x].x; double p2 = h_phi2[y * nx + x].x;
    double d1x = (h_phi1[y * nx + xp].x - h_phi1[y * nx + xm].x) / (2.0 * dx);
    double d1y = (h_phi1[yp * nx + x].x - h_phi1[ym * nx + x].x) / (2.0 * dx);
    double d2x = (h_phi2[y * nx + xp].x - h_phi2[y * nx + xm].x) / (2.0 * dx);
    double d2y = (h_phi2[yp * nx + x].x - h_phi2[ym * nx + x].x) / (2.0 * dx);
    
    jx = p1 * d2x - p2 * d1x;
    jy = p1 * d2y - p2 * d1y;
}

// --- Main Program ---
int main() {
    // 1. String-Based Parser
    std::map<std::string, std::string> p_str;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty() || line[0] == '#') continue; 
        std::istringstream iss(line);
        std::string key, val;
        if (iss >> key >> val) p_str[key] = val;
    }

    auto getD = [&](const std::string& k, double def) { return p_str.count(k) ? std::stod(p_str[k]) : def; };
    auto getI = [&](const std::string& k, int def) { return p_str.count(k) ? std::stoi(p_str[k]) : def; };
    auto getS = [&](const std::string& k, std::string def) { return p_str.count(k) ? p_str[k] : def; };

    const int nx = getI("nx", 128);
    const int ny = getI("ny", 128);
    const double dx = getD("dx", 0.5);
    const double dt = getD("dt", 0.001);
    const int n_steps = getI("n_steps", 50000);
    const int seed = getI("seed", 42);

    const int log_interval = getI("log_interval", 100);
    const int field_interval = getI("field_interval", 5000);
    const int save_fields = getI("save_fields", 1);
    const std::string prefix = getS("file_prefix", "output_");

    double kappa = getD("kappa", 0.01), chi = getD("chi", -0.2), chi_prime = getD("chi_prime", 0.2);
    const int N = nx * ny;

    srand(seed);
    cufftDoubleComplex *h_phi1 = new cufftDoubleComplex[N], *h_phi2 = new cufftDoubleComplex[N];
    cufftDoubleComplex *h_phi1_hat = new cufftDoubleComplex[N], *h_phi2_hat = new cufftDoubleComplex[N];

    for (int i = 0; i < N; i++) {
        h_phi1[i].x = 0.35 + 0.01 * ((double)rand() / RAND_MAX - 0.5); h_phi1[i].y = 0.0;
        h_phi2[i].x = 0.30 + 0.01 * ((double)rand() / RAND_MAX - 0.5); h_phi2[i].y = 0.0;
    }

    cufftDoubleComplex *d_phi1, *d_phi2, *d_phi1_hat, *d_phi2_hat, *d_N1, *d_N2, *d_N1_hat, *d_N2_hat;
    double *d_q2, *d_E, *d_E1;
    
    CUDA_CHECK(cudaMalloc(&d_phi1, sizeof(cufftDoubleComplex) * N));
    CUDA_CHECK(cudaMalloc(&d_phi2, sizeof(cufftDoubleComplex) * N));
    CUDA_CHECK(cudaMalloc(&d_phi1_hat, sizeof(cufftDoubleComplex) * N));
    CUDA_CHECK(cudaMalloc(&d_phi2_hat, sizeof(cufftDoubleComplex) * N));
    CUDA_CHECK(cudaMalloc(&d_N1, sizeof(cufftDoubleComplex) * N));
    CUDA_CHECK(cudaMalloc(&d_N2, sizeof(cufftDoubleComplex) * N));
    CUDA_CHECK(cudaMalloc(&d_N1_hat, sizeof(cufftDoubleComplex) * N));
    CUDA_CHECK(cudaMalloc(&d_N2_hat, sizeof(cufftDoubleComplex) * N));
    CUDA_CHECK(cudaMalloc(&d_q2, sizeof(double) * N));
    CUDA_CHECK(cudaMalloc(&d_E, sizeof(double) * N));
    CUDA_CHECK(cudaMalloc(&d_E1, sizeof(double) * N));

    CUDA_CHECK(cudaMemcpy(d_phi1, h_phi1, sizeof(cufftDoubleComplex) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_phi2, h_phi2, sizeof(cufftDoubleComplex) * N, cudaMemcpyHostToDevice));

    cufftHandle plan;
    cufftPlan2d(&plan, ny, nx, CUFFT_Z2Z);
    dim3 threads(16, 16); dim3 blocks((nx + 15) / 16, (ny + 15) / 16);

    init_fourier<<<blocks, threads>>>(d_q2, d_E, d_E1, nx, ny, dx, dt, kappa);
    cufftExecZ2Z(plan, d_phi1, d_phi1_hat, CUFFT_FORWARD);
    cufftExecZ2Z(plan, d_phi2, d_phi2_hat, CUFFT_FORWARD);

    std::cout << "Step\tTime\tAlpha\tFree_Energy\tRMS_Vorticity\tDomain_Size_R\n";

    for (int step = 0; step <= n_steps; step++) {
        
        double prog = (double)step / n_steps;
        double alpha = getD("alpha_start", 0.0) + prog * (getD("alpha_end", 0.45) - getD("alpha_start", 0.0));
        double c11 = getD("c11_start", 0.2) + prog * (getD("c11_end", 0.2) - getD("c11_start", 0.2));
        double c12 = getD("c12_start", 0.5) + prog * (getD("c12_end", 0.5) - getD("c12_start", 0.5));
        double c21 = getD("c21_start", 0.1) + prog * (getD("c21_end", 0.1) - getD("c21_start", 0.1));
        double c22 = getD("c22_start", 0.5) + prog * (getD("c22_end", 0.5) - getD("c22_start", 0.5));

        cufftExecZ2Z(plan, d_phi1_hat, d_phi1, CUFFT_INVERSE);
        cufftExecZ2Z(plan, d_phi2_hat, d_phi2, CUFFT_INVERSE);

        double scale = 1.0 / N;
        calc_nonlinear<<<blocks, threads>>>(d_phi1, d_phi2, d_N1, d_N2, nx, ny, scale, chi, chi_prime, alpha, c11, c12, c21, c22);

        bool do_log = (step % log_interval == 0);
        bool do_field = (save_fields && step % field_interval == 0);

        if (do_log || do_field) {
            CUDA_CHECK(cudaMemcpy(h_phi1, d_phi1, sizeof(cufftDoubleComplex) * N, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_phi2, d_phi2, sizeof(cufftDoubleComplex) * N, cudaMemcpyDeviceToHost));
            
            if (do_log) {
                CUDA_CHECK(cudaMemcpy(h_phi1_hat, d_phi1_hat, sizeof(cufftDoubleComplex) * N, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_phi2_hat, d_phi2_hat, sizeof(cufftDoubleComplex) * N, cudaMemcpyDeviceToHost));

                double F_bulk = 0.0, sum_vorticity_sq = 0.0, sum_S = 0.0, sum_qS = 0.0, F_grad = 0.0;
                for (int y = 0; y < ny; y++) {
                    for (int x = 0; x < nx; x++) {
                        int i = y * nx + x;
                        double p1 = h_phi1[i].x, p2 = h_phi2[i].x;
                        
                        F_bulk += (pow(p1 - c11, 2) * pow(p1 - c12, 2) + pow(p2 - c21, 2) * pow(p2 - c22, 2) + chi * p1 * p2 + chi_prime * p1 * p1 * p2 * p2) * (dx * dx);

                        double jx_xp, jy_xp, jx_xm, jy_xm, jx_yp, jy_yp, jx_ym, jy_ym;
                        get_flux((x + 1) % nx, y, nx, ny, dx, h_phi1, h_phi2, jx_xp, jy_xp);
                        get_flux((x - 1 + nx) % nx, y, nx, ny, dx, h_phi1, h_phi2, jx_xm, jy_xm);
                        get_flux(x, (y + 1) % ny, nx, ny, dx, h_phi1, h_phi2, jx_yp, jy_yp);
                        get_flux(x, (y - 1 + ny) % ny, nx, ny, dx, h_phi1, h_phi2, jx_ym, jy_ym);
                        double omega = (jy_xp - jy_xm) / (2.0 * dx) - (jx_yp - jx_ym) / (2.0 * dx);
                        sum_vorticity_sq += omega * omega;

                        double kx = (x < nx / 2 ? x : x - nx) * 2.0 * M_PI / (nx * dx);
                        double ky = (y < ny / 2 ? y : y - ny) * 2.0 * M_PI / (ny * dx);
                        double q2 = kx * kx + ky * ky;
                        double S1 = pow(h_phi1_hat[i].x / N, 2) + pow(h_phi1_hat[i].y / N, 2);
                        double S2 = pow(h_phi2_hat[i].x / N, 2) + pow(h_phi2_hat[i].y / N, 2);

                        sum_S += S1; sum_qS += sqrt(q2) * S1;
                        F_grad += 0.5 * kappa * q2 * (S1 + S2) * (dx * dx);
                    }
                }
                
                std::cout << std::fixed << std::setprecision(6) << step << "\t" << step * dt << "\t" << alpha << "\t"
                          << (F_bulk + F_grad) << "\t" << sqrt(sum_vorticity_sq / N) << "\t" 
                          << ((sum_qS > 0) ? (2.0 * M_PI * sum_S / sum_qS) : 0.0) << "\n";
                
                std::cerr << "Step " << step << " | Alpha: " << alpha << " | F: " << (F_bulk + F_grad) << "\n";
            }

            if (do_field) {
                char filename[256];
                sprintf(filename, "%s%08d.h5", prefix.c_str(), step);
                
                hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
                hsize_t dims[2] = {(hsize_t)ny, (hsize_t)nx};
                hid_t dataspace_id = H5Screate_simple(2, dims, NULL);
                
                double* out_phi1 = new double[N];
                double* out_phi2 = new double[N];
                for (int i = 0; i < N; i++) {
                    out_phi1[i] = h_phi1[i].x;
                    out_phi2[i] = h_phi2[i].x;
                }
                
                hid_t ds1_id = H5Dcreate2(file_id, "phi1", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                hid_t ds2_id = H5Dcreate2(file_id, "phi2", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                
                H5Dwrite(ds1_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out_phi1);
                H5Dwrite(ds2_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out_phi2);
                
                H5Dclose(ds1_id);
                H5Dclose(ds2_id);
                H5Sclose(dataspace_id);
                H5Fclose(file_id);
                delete[] out_phi1;
                delete[] out_phi2;
            }
        }

        cufftExecZ2Z(plan, d_N1, d_N1_hat, CUFFT_FORWARD);
        cufftExecZ2Z(plan, d_N2, d_N2_hat, CUFFT_FORWARD);
        etd_step<<<blocks, threads>>>(d_phi1_hat, d_phi2_hat, d_N1_hat, d_N2_hat, d_q2, d_E, d_E1, nx, ny);
    }

    cufftDestroy(plan);
    cudaFree(d_phi1); cudaFree(d_phi2); cudaFree(d_phi1_hat); cudaFree(d_phi2_hat);
    cudaFree(d_N1); cudaFree(d_N2); cudaFree(d_N1_hat); cudaFree(d_N2_hat);
    cudaFree(d_q2); cudaFree(d_E); cudaFree(d_E1);
    delete[] h_phi1; delete[] h_phi2; delete[] h_phi1_hat; delete[] h_phi2_hat;

    return 0;
}
