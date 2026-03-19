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
#include <complex>
#include <vector>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// ============================================================
// CUDA Kernels (all 1D indexed)
// ============================================================

__global__ void calc_nonlinear(cufftDoubleComplex* phi1, cufftDoubleComplex* phi2,
                               cufftDoubleComplex* N1, cufftDoubleComplex* N2,
                               int N_total, double scale,
                               double chi, double chi_prime, double alpha,
                               double c11, double c12, double c21, double c22) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_total) {
        double p1 = phi1[i].x * scale;
        double p2 = phi2[i].x * scale;
        phi1[i].x = p1; phi1[i].y = 0.0;
        phi2[i].x = p2; phi2[i].y = 0.0;

        double bulk1 = 2.0 * (p1 - c11) * (p1 - c12) * (2.0 * p1 - c11 - c12);
        double int1  = chi * p2 + 2.0 * chi_prime * p1 * (p2 * p2);
        double bulk2 = 2.0 * (p2 - c21) * (p2 - c22) * (2.0 * p2 - c21 - c22);
        double int2  = chi * p1 + 2.0 * chi_prime * p2 * (p1 * p1);

        N1[i].x = bulk1 + int1 + alpha * p2; N1[i].y = 0.0;
        N2[i].x = bulk2 + int2 - alpha * p1; N2[i].y = 0.0;
    }
}

__global__ void calc_nonlinear_no_overwrite(
                               const cufftDoubleComplex* phi1,
                               const cufftDoubleComplex* phi2,
                               cufftDoubleComplex* N1, cufftDoubleComplex* N2,
                               int N_total, double scale,
                               double chi, double chi_prime, double alpha,
                               double c11, double c12, double c21, double c22) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_total) {
        double p1 = phi1[i].x * scale;
        double p2 = phi2[i].x * scale;

        double bulk1 = 2.0 * (p1 - c11) * (p1 - c12) * (2.0 * p1 - c11 - c12);
        double int1  = chi * p2 + 2.0 * chi_prime * p1 * (p2 * p2);
        double bulk2 = 2.0 * (p2 - c21) * (p2 - c22) * (2.0 * p2 - c21 - c22);
        double int2  = chi * p1 + 2.0 * chi_prime * p2 * (p1 * p1);

        N1[i].x = bulk1 + int1 + alpha * p2; N1[i].y = 0.0;
        N2[i].x = bulk2 + int2 - alpha * p1; N2[i].y = 0.0;
    }
}

__global__ void apply_laplacian(cufftDoubleComplex* N_hat, const double* q2,
                                int N_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_total) {
        double mq2 = -q2[i];
        N_hat[i].x *= mq2;
        N_hat[i].y *= mq2;
    }
}

__global__ void etdrk4_stage_a(
    cufftDoubleComplex* a_hat,
    const cufftDoubleComplex* u_hat,
    const cufftDoubleComplex* Na_hat,
    const double* E2, const double* E2_1,
    int N_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_total) {
        a_hat[i].x = E2[i] * u_hat[i].x + E2_1[i] * Na_hat[i].x;
        a_hat[i].y = E2[i] * u_hat[i].y + E2_1[i] * Na_hat[i].y;
    }
}

__global__ void etdrk4_stage_b(
    cufftDoubleComplex* b_hat,
    const cufftDoubleComplex* u_hat,
    const cufftDoubleComplex* Nb_hat,
    const double* E2, const double* E2_1,
    int N_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_total) {
        b_hat[i].x = E2[i] * u_hat[i].x + E2_1[i] * Nb_hat[i].x;
        b_hat[i].y = E2[i] * u_hat[i].y + E2_1[i] * Nb_hat[i].y;
    }
}

__global__ void etdrk4_stage_c(
    cufftDoubleComplex* c_hat,
    const cufftDoubleComplex* a_hat,
    const cufftDoubleComplex* Nc_hat,
    const cufftDoubleComplex* Na_hat,
    const double* E2, const double* E2_1,
    int N_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_total) {
        double rhs_r = 2.0 * Nc_hat[i].x - Na_hat[i].x;
        double rhs_i = 2.0 * Nc_hat[i].y - Na_hat[i].y;
        c_hat[i].x = E2[i] * a_hat[i].x + E2_1[i] * rhs_r;
        c_hat[i].y = E2[i] * a_hat[i].y + E2_1[i] * rhs_i;
    }
}

__global__ void etdrk4_final(
    cufftDoubleComplex* u_hat,
    const cufftDoubleComplex* Na_hat,
    const cufftDoubleComplex* Nb_hat,
    const cufftDoubleComplex* Nc_hat,
    const cufftDoubleComplex* Nd_hat,
    const double* E_full, const double* f1,
    const double* f2, const double* f3,
    int N_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N_total) {
        double ur = E_full[i] * u_hat[i].x
                  + f1[i] * Na_hat[i].x
                  + 2.0 * f2[i] * (Nb_hat[i].x + Nc_hat[i].x)
                  + f3[i] * Nd_hat[i].x;
        double ui = E_full[i] * u_hat[i].y
                  + f1[i] * Na_hat[i].y
                  + 2.0 * f2[i] * (Nb_hat[i].y + Nc_hat[i].y)
                  + f3[i] * Nd_hat[i].y;
        u_hat[i].x = ur;
        u_hat[i].y = ui;
    }
}

// ============================================================
// Host helper: compute ETDRK4 coefficients via contour integrals
// ============================================================
void compute_etdrk4_coefficients(
    const double* h_q2, int N_total, double dt, double kappa,
    double* h_E, double* h_E2, double* h_E2_1,
    double* h_f1, double* h_f2, double* h_f3)
{
    const int M = 64;
    const double r = 1.0;

    std::vector<std::complex<double>> contour_pts(M);
    for (int j = 0; j < M; j++) {
        double theta = M_PI * (2.0 * (j + 1) - 1.0) / (2.0 * M);
        contour_pts[j] = r * std::complex<double>(cos(theta), sin(theta));
    }

    for (int i = 0; i < N_total; i++) {
        double L = -kappa * h_q2[i] * h_q2[i];
        double Ldt = L * dt;
        double Ldt2 = L * dt * 0.5;

        h_E[i]  = exp(Ldt);
        h_E2[i] = exp(Ldt2);

        std::complex<double> sum_E2_1(0.0, 0.0);
        std::complex<double> sum_f1(0.0, 0.0);
        std::complex<double> sum_f2(0.0, 0.0);
        std::complex<double> sum_f3(0.0, 0.0);

        for (int j = 0; j < M; j++) {
            std::complex<double> z = Ldt + contour_pts[j];
            std::complex<double> z2 = Ldt2 + contour_pts[j] * 0.5;
            std::complex<double> ez = exp(z);
            std::complex<double> ez2 = exp(z2);

            sum_E2_1 += (ez2 - 1.0) / z2;

            std::complex<double> z3 = z * z * z;
            sum_f1 += (-4.0 - z + ez * (4.0 - 3.0 * z + z * z)) / z3;
            sum_f2 += (2.0 + z + ez * (-2.0 + z)) / z3;
            sum_f3 += (-4.0 - 3.0 * z - z * z + ez * (4.0 - z)) / z3;
        }

        h_E2_1[i] = (dt * 0.5) * (sum_E2_1 / (double)M).real();
        h_f1[i]   = dt * (sum_f1 / (double)M).real();
        h_f2[i]   = dt * (sum_f2 / (double)M).real();
        h_f3[i]   = dt * (sum_f3 / (double)M).real();
    }
}

// ============================================================
// Host helper for vorticity
// ============================================================
void get_flux(int x, int y, int N_grid, double dx,
              cufftDoubleComplex* h_phi1, cufftDoubleComplex* h_phi2,
              double &jx, double &jy) {
    int xp = (x + 1) % N_grid; int xm = (x - 1 + N_grid) % N_grid;
    int yp = (y + 1) % N_grid; int ym = (y - 1 + N_grid) % N_grid;

    double d1x = (h_phi1[y * N_grid + xp].x - h_phi1[y * N_grid + xm].x) / (2.0 * dx);
    double d1y = (h_phi1[yp * N_grid + x].x - h_phi1[ym * N_grid + x].x) / (2.0 * dx);
    double d2x = (h_phi2[y * N_grid + xp].x - h_phi2[y * N_grid + xm].x) / (2.0 * dx);
    double d2y = (h_phi2[yp * N_grid + x].x - h_phi2[ym * N_grid + x].x) / (2.0 * dx);

    double p1 = h_phi1[y * N_grid + x].x;
    double p2 = h_phi2[y * N_grid + x].x;

    jx = p1 * d2x - p2 * d1x;
    jy = p1 * d2y - p2 * d1y;
}

// ============================================================
// Helper: compute N_hat from phi_hat
// ============================================================
void compute_N_hat(
    cufftHandle plan,
    cufftDoubleComplex* d_phi1_hat_in,
    cufftDoubleComplex* d_phi2_hat_in,
    cufftDoubleComplex* d_phi1_work,
    cufftDoubleComplex* d_phi2_work,
    cufftDoubleComplex* d_N1, cufftDoubleComplex* d_N2,
    cufftDoubleComplex* d_N1_hat, cufftDoubleComplex* d_N2_hat,
    double* d_q2,
    int N_total, int threads_per_block,
    double chi, double chi_prime, double alpha,
    double c11, double c12, double c21, double c22,
    bool overwrite_phi,
    cufftDoubleComplex* d_phi1_out = nullptr,
    cufftDoubleComplex* d_phi2_out = nullptr)
{
    double scale = 1.0 / N_total;
    int blocks = (N_total + threads_per_block - 1) / threads_per_block;

    CUDA_CHECK(cudaMemcpy(d_phi1_work, d_phi1_hat_in,
                          sizeof(cufftDoubleComplex) * N_total, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_phi2_work, d_phi2_hat_in,
                          sizeof(cufftDoubleComplex) * N_total, cudaMemcpyDeviceToDevice));

    cufftExecZ2Z(plan, d_phi1_work, d_phi1_work, CUFFT_INVERSE);
    cufftExecZ2Z(plan, d_phi2_work, d_phi2_work, CUFFT_INVERSE);

    if (overwrite_phi && d_phi1_out && d_phi2_out) {
        calc_nonlinear<<<blocks, threads_per_block>>>(
            d_phi1_work, d_phi2_work, d_N1, d_N2,
            N_total, scale, chi, chi_prime, alpha, c11, c12, c21, c22);
        CUDA_CHECK(cudaMemcpy(d_phi1_out, d_phi1_work,
                              sizeof(cufftDoubleComplex) * N_total, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_phi2_out, d_phi2_work,
                              sizeof(cufftDoubleComplex) * N_total, cudaMemcpyDeviceToDevice));
    } else {
        calc_nonlinear_no_overwrite<<<blocks, threads_per_block>>>(
            d_phi1_work, d_phi2_work, d_N1, d_N2,
            N_total, scale, chi, chi_prime, alpha, c11, c12, c21, c22);
    }

    cufftExecZ2Z(plan, d_N1, d_N1_hat, CUFFT_FORWARD);
    cufftExecZ2Z(plan, d_N2, d_N2_hat, CUFFT_FORWARD);

    apply_laplacian<<<blocks, threads_per_block>>>(d_N1_hat, d_q2, N_total);
    apply_laplacian<<<blocks, threads_per_block>>>(d_N2_hat, d_q2, N_total);
}

// ============================================================
// Main Program
// ============================================================
int main() {
    // --- Parse input ---
    std::map<std::string, std::string> p_str;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string key, val;
        if (iss >> key >> val) p_str[key] = val;
    }

    auto getD = [&](const std::string& k, double def) {
        return p_str.count(k) ? std::stod(p_str[k]) : def; };
    auto getI = [&](const std::string& k, int def) {
        return p_str.count(k) ? std::stoi(p_str[k]) : def; };
    auto getS = [&](const std::string& k, std::string def) {
        return p_str.count(k) ? p_str[k] : def; };

    // --- Grid: single size parameter, square grid enforced ---
    const int N_grid      = getI("N_grid", 128);
    const double dx       = getD("dx", 0.5);
    const double dt       = getD("dt", 0.001);
    const int n_steps     = getI("n_steps", 50000);
    const int seed        = getI("seed", 42);
    const int log_interval   = getI("log_interval", 100);
    const int field_interval = getI("field_interval", 5000);
    const int save_fields    = getI("save_fields", 1);
    const std::string prefix = getS("file_prefix", "output_");

    double kappa     = getD("kappa", 0.01);
    double chi       = getD("chi", -0.2);
    double chi_prime = getD("chi_prime", 0.2);

    double alpha_start = getD("alpha_start", 0.0);
    double alpha_end   = getD("alpha_end", 0.45);
    double c11_start   = getD("c11_start", 0.2);
    double c11_end     = getD("c11_end", 0.2);
    double c12_start   = getD("c12_start", 0.5);
    double c12_end     = getD("c12_end", 0.5);
    double c21_start   = getD("c21_start", 0.1);
    double c21_end     = getD("c21_end", 0.1);
    double c22_start   = getD("c22_start", 0.5);
    double c22_end     = getD("c22_end", 0.5);

    const int N_total = N_grid * N_grid;
    const double L_box = N_grid * dx;
    const double T_final = n_steps * dt;

    // ============================================================
    // Print parameter header
    // ============================================================
    std::cout << "# ============================================================\n";
    std::cout << "# Cahn-Hilliard ETDRK4 Simulation (square grid enforced)\n";
    std::cout << "# ============================================================\n";
    std::cout << "#\n";
    std::cout << "# --- Grid Parameters ---\n";
    std::cout << "#   N_grid         = " << N_grid << " (nx = ny = N_grid)\n";
    std::cout << "#   dx             = " << dx << "\n";
    std::cout << "#   L_box          = " << L_box << "\n";
    std::cout << "#   N_total        = " << N_total << "\n";
    std::cout << "#\n";
    std::cout << "# --- Time Integration ---\n";
    std::cout << "#   dt             = " << dt << "\n";
    std::cout << "#   n_steps        = " << n_steps << "\n";
    std::cout << "#   T_final        = " << T_final << "\n";
    std::cout << "#   method         = ETDRK4 (Kassam-Trefethen)\n";
    std::cout << "#   contour_pts    = 64\n";
    std::cout << "#\n";
    std::cout << "# --- Model Parameters ---\n";
    std::cout << "#   kappa          = " << kappa << "\n";
    std::cout << "#   chi            = " << chi << "\n";
    std::cout << "#   chi_prime      = " << chi_prime << "\n";
    std::cout << "#\n";
    std::cout << "# --- Time-Dependent Parameters ---\n";
    std::cout << "#   alpha          = " << alpha_start << " -> " << alpha_end << "\n";
    std::cout << "#   c11            = " << c11_start << " -> " << c11_end << "\n";
    std::cout << "#   c12            = " << c12_start << " -> " << c12_end << "\n";
    std::cout << "#   c21            = " << c21_start << " -> " << c21_end << "\n";
    std::cout << "#   c22            = " << c22_start << " -> " << c22_end << "\n";
    std::cout << "#\n";
    std::cout << "# --- Initial Conditions ---\n";
    std::cout << "#   seed           = " << seed << "\n";
    std::cout << "#   phi1_init      = 0.35 + 0.01*U(-0.5,0.5)\n";
    std::cout << "#   phi2_init      = 0.30 + 0.01*U(-0.5,0.5)\n";
    std::cout << "#\n";
    std::cout << "# --- Output Controls ---\n";
    std::cout << "#   log_interval   = " << log_interval << "\n";
    std::cout << "#   field_interval = " << field_interval << "\n";
    std::cout << "#   save_fields    = " << save_fields << "\n";
    std::cout << "#   file_prefix    = " << prefix << "\n";
    std::cout << "#\n";

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "# --- GPU Info ---\n";
    std::cout << "#   device         = " << prop.name << "\n";
    std::cout << "#   compute_cap    = " << prop.major << "." << prop.minor << "\n";
    std::cout << "#   global_mem_MB  = " << prop.totalGlobalMem / (1024*1024) << "\n";
    std::cout << "#\n";

    double q_max = M_PI / dx;
    double L_max = kappa * pow(q_max, 4);
    double Ldt_max = L_max * dt;
    std::cout << "# --- Derived Quantities ---\n";
    std::cout << "#   q_max          = " << q_max << "\n";
    std::cout << "#   |L_max|        = " << L_max << "\n";
    std::cout << "#   |L_max|*dt     = " << Ldt_max << "\n";
    std::cout << "#   GPU_mem_est_MB = "
              << (22.0 * sizeof(cufftDoubleComplex) * N_total + 7.0 * sizeof(double) * N_total)
                 / (1024.0 * 1024.0) << "\n";
    std::cout << "#\n";
    std::cout << "# ============================================================\n";
    std::cout << "# Step\tTime\tAlpha\tFree_Energy\tRMS_Vorticity\tDomain_Size_R\n";
    std::cout << "# ============================================================\n";

    std::cerr << "=== Simulation Parameters ===" << std::endl;
    std::cerr << "Grid: " << N_grid << "x" << N_grid << " (dx=" << dx
              << ", L=" << L_box << ")" << std::endl;
    std::cerr << "Time: dt=" << dt << ", n_steps=" << n_steps
              << ", T_final=" << T_final << std::endl;
    std::cerr << "Model: kappa=" << kappa << ", chi=" << chi
              << ", chi_prime=" << chi_prime << std::endl;
    std::cerr << "Alpha: " << alpha_start << " -> " << alpha_end << std::endl;
    std::cerr << "c11: " << c11_start << " -> " << c11_end << std::endl;
    std::cerr << "c12: " << c12_start << " -> " << c12_end << std::endl;
    std::cerr << "c21: " << c21_start << " -> " << c21_end << std::endl;
    std::cerr << "c22: " << c22_start << " -> " << c22_end << std::endl;
    std::cerr << "GPU: " << prop.name << std::endl;
    std::cerr << "|L_max|*dt = " << Ldt_max << std::endl;
    std::cerr << "=============================" << std::endl;

    // --- Initialize fields ---
    srand(seed);
    cufftDoubleComplex* h_phi1     = new cufftDoubleComplex[N_total];
    cufftDoubleComplex* h_phi2     = new cufftDoubleComplex[N_total];
    cufftDoubleComplex* h_phi1_hat = new cufftDoubleComplex[N_total];
    cufftDoubleComplex* h_phi2_hat = new cufftDoubleComplex[N_total];

    for (int i = 0; i < N_total; i++) {
        h_phi1[i].x = 0.35 + 0.01 * ((double)rand() / RAND_MAX - 0.5);
        h_phi1[i].y = 0.0;
        h_phi2[i].x = 0.30 + 0.01 * ((double)rand() / RAND_MAX - 0.5);
        h_phi2[i].y = 0.0;
    }

    // --- Compute q^2 on host ---
    double* h_q2 = new double[N_total];
    for (int iy = 0; iy < N_grid; iy++) {
        for (int ix = 0; ix < N_grid; ix++) {
            double kx = (ix < N_grid / 2 ? ix : ix - N_grid) * 2.0 * M_PI / (N_grid * dx);
            double ky = (iy < N_grid / 2 ? iy : iy - N_grid) * 2.0 * M_PI / (N_grid * dx);
            h_q2[iy * N_grid + ix] = kx * kx + ky * ky;
        }
    }

    // --- Compute ETDRK4 coefficients ---
    double* h_E    = new double[N_total];
    double* h_E2   = new double[N_total];
    double* h_E2_1 = new double[N_total];
    double* h_f1   = new double[N_total];
    double* h_f2   = new double[N_total];
    double* h_f3   = new double[N_total];

    std::cerr << "Computing ETDRK4 coefficients via contour integrals (M=64)..."
              << std::endl;
    compute_etdrk4_coefficients(h_q2, N_total, dt, kappa,
                                h_E, h_E2, h_E2_1, h_f1, h_f2, h_f3);
    std::cerr << "Done." << std::endl;

    // --- Allocate device memory ---
    cufftDoubleComplex *d_phi1_hat, *d_phi2_hat;
    cufftDoubleComplex *d_phi1_work, *d_phi2_work;
    cufftDoubleComplex *d_a1_hat, *d_a2_hat;
    cufftDoubleComplex *d_b1_hat, *d_b2_hat;
    cufftDoubleComplex *d_c1_hat, *d_c2_hat;
    cufftDoubleComplex *d_N1, *d_N2;
    cufftDoubleComplex *d_N1_hat, *d_N2_hat;
    cufftDoubleComplex *d_Na1_hat, *d_Na2_hat;
    cufftDoubleComplex *d_Nb1_hat, *d_Nb2_hat;
    cufftDoubleComplex *d_Nc1_hat, *d_Nc2_hat;
    cufftDoubleComplex *d_Nd1_hat, *d_Nd2_hat;

    double *d_q2, *d_E, *d_E2, *d_E2_1, *d_f1, *d_f2, *d_f3;

    size_t csize = sizeof(cufftDoubleComplex) * N_total;
    size_t dsize = sizeof(double) * N_total;

    CUDA_CHECK(cudaMalloc(&d_phi1_hat, csize));
    CUDA_CHECK(cudaMalloc(&d_phi2_hat, csize));
    CUDA_CHECK(cudaMalloc(&d_phi1_work, csize));
    CUDA_CHECK(cudaMalloc(&d_phi2_work, csize));
    CUDA_CHECK(cudaMalloc(&d_a1_hat, csize));
    CUDA_CHECK(cudaMalloc(&d_a2_hat, csize));
    CUDA_CHECK(cudaMalloc(&d_b1_hat, csize));
    CUDA_CHECK(cudaMalloc(&d_b2_hat, csize));
    CUDA_CHECK(cudaMalloc(&d_c1_hat, csize));
    CUDA_CHECK(cudaMalloc(&d_c2_hat, csize));
    CUDA_CHECK(cudaMalloc(&d_N1, csize));
    CUDA_CHECK(cudaMalloc(&d_N2, csize));
    CUDA_CHECK(cudaMalloc(&d_N1_hat, csize));
    CUDA_CHECK(cudaMalloc(&d_N2_hat, csize));
    CUDA_CHECK(cudaMalloc(&d_Na1_hat, csize));
    CUDA_CHECK(cudaMalloc(&d_Na2_hat, csize));
    CUDA_CHECK(cudaMalloc(&d_Nb1_hat, csize));
    CUDA_CHECK(cudaMalloc(&d_Nb2_hat, csize));
    CUDA_CHECK(cudaMalloc(&d_Nc1_hat, csize));
    CUDA_CHECK(cudaMalloc(&d_Nc2_hat, csize));
    CUDA_CHECK(cudaMalloc(&d_Nd1_hat, csize));
    CUDA_CHECK(cudaMalloc(&d_Nd2_hat, csize));

    CUDA_CHECK(cudaMalloc(&d_q2, dsize));
    CUDA_CHECK(cudaMalloc(&d_E, dsize));
    CUDA_CHECK(cudaMalloc(&d_E2, dsize));
    CUDA_CHECK(cudaMalloc(&d_E2_1, dsize));
    CUDA_CHECK(cudaMalloc(&d_f1, dsize));
    CUDA_CHECK(cudaMalloc(&d_f2, dsize));
    CUDA_CHECK(cudaMalloc(&d_f3, dsize));

    CUDA_CHECK(cudaMemcpy(d_phi1_work, h_phi1, csize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_phi2_work, h_phi2, csize, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_q2, h_q2, dsize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E, h_E, dsize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E2, h_E2, dsize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E2_1, h_E2_1, dsize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f1, h_f1, dsize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f2, h_f2, dsize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f3, h_f3, dsize, cudaMemcpyHostToDevice));

    // --- Create FFT plan (square: N_grid x N_grid) ---
    cufftHandle plan;
    cufftPlan2d(&plan, N_grid, N_grid, CUFFT_Z2Z);

    // --- FFT initial conditions ---
    cufftExecZ2Z(plan, d_phi1_work, d_phi1_hat, CUFFT_FORWARD);
    cufftExecZ2Z(plan, d_phi2_work, d_phi2_hat, CUFFT_FORWARD);

    // --- Kernel launch config (all 1D) ---
    const int threads_per_block = 256;
    int blocks_per_grid = (N_total + threads_per_block - 1) / threads_per_block;

    // ============================================================
    // Main time loop
    // ============================================================
    for (int step = 0; step <= n_steps; step++) {

        double prog = (double)step / n_steps;
        double alpha = alpha_start + prog * (alpha_end - alpha_start);
        double c11   = c11_start   + prog * (c11_end   - c11_start);
        double c12   = c12_start   + prog * (c12_end   - c12_start);
        double c21   = c21_start   + prog * (c21_end   - c21_start);
        double c22   = c22_start   + prog * (c22_end   - c22_start);

        // --- Diagnostics ---
        bool do_log   = (step % log_interval == 0);
        bool do_field = (save_fields && step % field_interval == 0);

        if (do_log || do_field) {
            CUDA_CHECK(cudaMemcpy(d_phi1_work, d_phi1_hat, csize,
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_phi2_work, d_phi2_hat, csize,
                                  cudaMemcpyDeviceToDevice));
            cufftExecZ2Z(plan, d_phi1_work, d_phi1_work, CUFFT_INVERSE);
            cufftExecZ2Z(plan, d_phi2_work, d_phi2_work, CUFFT_INVERSE);

            CUDA_CHECK(cudaMemcpy(h_phi1, d_phi1_work, csize,
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_phi2, d_phi2_work, csize,
                                  cudaMemcpyDeviceToHost));
            for (int i = 0; i < N_total; i++) {
                h_phi1[i].x /= N_total; h_phi1[i].y = 0.0;
                h_phi2[i].x /= N_total; h_phi2[i].y = 0.0;
            }

            if (do_log) {
                CUDA_CHECK(cudaMemcpy(h_phi1_hat, d_phi1_hat, csize,
                                      cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_phi2_hat, d_phi2_hat, csize,
                                      cudaMemcpyDeviceToHost));

                double F_bulk = 0.0, sum_vort_sq = 0.0;
                double sum_S = 0.0, sum_qS = 0.0, F_grad = 0.0;

                for (int y = 0; y < N_grid; y++) {
                    for (int x = 0; x < N_grid; x++) {
                        int i = y * N_grid + x;
                        double p1 = h_phi1[i].x, p2 = h_phi2[i].x;

                        F_bulk += (pow(p1-c11,2)*pow(p1-c12,2)
                                 + pow(p2-c21,2)*pow(p2-c22,2)
                                 + chi*p1*p2
                                 + chi_prime*p1*p1*p2*p2) * (dx*dx);

                        double jx_xp,jy_xp,jx_xm,jy_xm,jx_yp,jy_yp,jx_ym,jy_ym;
                        get_flux((x+1)%N_grid, y, N_grid, dx, h_phi1, h_phi2,
                                 jx_xp, jy_xp);
                        get_flux((x-1+N_grid)%N_grid, y, N_grid, dx, h_phi1, h_phi2,
                                 jx_xm, jy_xm);
                        get_flux(x, (y+1)%N_grid, N_grid, dx, h_phi1, h_phi2,
                                 jx_yp, jy_yp);
                        get_flux(x, (y-1+N_grid)%N_grid, N_grid, dx, h_phi1, h_phi2,
                                 jx_ym, jy_ym);
                        double omega = (jy_xp - jy_xm)/(2.0*dx)
                                     - (jx_yp - jx_ym)/(2.0*dx);
                        sum_vort_sq += omega * omega;

                        double kx_val = (x < N_grid/2 ? x : x-N_grid) * 2.0*M_PI/(N_grid*dx);
                        double ky_val = (y < N_grid/2 ? y : y-N_grid) * 2.0*M_PI/(N_grid*dx);
                        double q2_val = kx_val*kx_val + ky_val*ky_val;
                        double S1 = pow(h_phi1_hat[i].x/N_total, 2)
                                  + pow(h_phi1_hat[i].y/N_total, 2);
                        double S2 = pow(h_phi2_hat[i].x/N_total, 2)
                                  + pow(h_phi2_hat[i].y/N_total, 2);

                        sum_S  += S1;
                        sum_qS += sqrt(q2_val) * S1;
                        F_grad += 0.5 * kappa * q2_val * (S1 + S2) * (dx*dx);
                    }
                }

                double F_total = F_bulk + F_grad;
                if (std::isnan(F_total) || std::isinf(F_total)) {
                    std::cerr << "ERROR: NaN/Inf detected at step " << step
                              << " (t=" << step*dt << "). Aborting." << std::endl;
                    break;
                }

                std::cout << std::fixed << std::setprecision(6)
                          << step << "\t" << step*dt << "\t" << alpha << "\t"
                          << F_total << "\t"
                          << sqrt(sum_vort_sq / N_total) << "\t"
                          << ((sum_qS > 0) ? (2.0*M_PI*sum_S/sum_qS) : 0.0)
                          << "\n";

                std::cerr << "Step " << step << " | t=" << step*dt
                          << " | Alpha: " << alpha
                          << " | F: " << F_total << "\n";
            }

            if (do_field) {
                char filename[256];
                sprintf(filename, "%s%010d.h5", prefix.c_str(), step);

                hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC,
                                          H5P_DEFAULT, H5P_DEFAULT);
                hsize_t dims[2] = {(hsize_t)N_grid, (hsize_t)N_grid};
                hid_t dspace = H5Screate_simple(2, dims, NULL);

                double* out1 = new double[N_total];
                double* out2 = new double[N_total];
                for (int i = 0; i < N_total; i++) {
                    out1[i] = h_phi1[i].x;
                    out2[i] = h_phi2[i].x;
                }

                hid_t ds1 = H5Dcreate2(file_id, "phi1", H5T_NATIVE_DOUBLE,
                                        dspace, H5P_DEFAULT, H5P_DEFAULT,
                                        H5P_DEFAULT);
                hid_t ds2 = H5Dcreate2(file_id, "phi2", H5T_NATIVE_DOUBLE,
                                        dspace, H5P_DEFAULT, H5P_DEFAULT,
                                        H5P_DEFAULT);
                H5Dwrite(ds1, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                         H5P_DEFAULT, out1);
                H5Dwrite(ds2, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                         H5P_DEFAULT, out2);
                H5Dclose(ds1); H5Dclose(ds2);
                H5Sclose(dspace); H5Fclose(file_id);
                delete[] out1; delete[] out2;
            }
        }

        if (step == n_steps) break;

        // ========================================================
        // ETDRK4 Time Step
        // ========================================================

        // --- Stage 1: Na = N(u^n) ---
        compute_N_hat(plan,
                      d_phi1_hat, d_phi2_hat,
                      d_phi1_work, d_phi2_work,
                      d_N1, d_N2, d_Na1_hat, d_Na2_hat,
                      d_q2, N_total, threads_per_block,
                      chi, chi_prime, alpha, c11, c12, c21, c22,
                      false);

        // --- Stage a ---
        etdrk4_stage_a<<<blocks_per_grid, threads_per_block>>>(
            d_a1_hat, d_phi1_hat, d_Na1_hat, d_E2, d_E2_1, N_total);
        etdrk4_stage_a<<<blocks_per_grid, threads_per_block>>>(
            d_a2_hat, d_phi2_hat, d_Na2_hat, d_E2, d_E2_1, N_total);

        // --- Stage 2: Nb = N(a) ---
        compute_N_hat(plan,
                      d_a1_hat, d_a2_hat,
                      d_phi1_work, d_phi2_work,
                      d_N1, d_N2, d_Nb1_hat, d_Nb2_hat,
                      d_q2, N_total, threads_per_block,
                      chi, chi_prime, alpha, c11, c12, c21, c22,
                      false);

        // --- Stage b ---
        etdrk4_stage_b<<<blocks_per_grid, threads_per_block>>>(
            d_b1_hat, d_phi1_hat, d_Nb1_hat, d_E2, d_E2_1, N_total);
        etdrk4_stage_b<<<blocks_per_grid, threads_per_block>>>(
            d_b2_hat, d_phi2_hat, d_Nb2_hat, d_E2, d_E2_1, N_total);

        // --- Stage 3: Nc = N(b) ---
        compute_N_hat(plan,
                      d_b1_hat, d_b2_hat,
                      d_phi1_work, d_phi2_work,
                      d_N1, d_N2, d_Nc1_hat, d_Nc2_hat,
                      d_q2, N_total, threads_per_block,
                      chi, chi_prime, alpha, c11, c12, c21, c22,
                      false);

        // --- Stage c ---
        etdrk4_stage_c<<<blocks_per_grid, threads_per_block>>>(
            d_c1_hat, d_a1_hat, d_Nc1_hat, d_Na1_hat, d_E2, d_E2_1, N_total);
        etdrk4_stage_c<<<blocks_per_grid, threads_per_block>>>(
            d_c2_hat, d_a2_hat, d_Nc2_hat, d_Na2_hat, d_E2, d_E2_1, N_total);

        // --- Stage 4: Nd = N(c) ---
        compute_N_hat(plan,
                      d_c1_hat, d_c2_hat,
                      d_phi1_work, d_phi2_work,
                      d_N1, d_N2, d_Nd1_hat, d_Nd2_hat,
                      d_q2, N_total, threads_per_block,
                      chi, chi_prime, alpha, c11, c12, c21, c22,
                      false);

        // --- Final update ---
        etdrk4_final<<<blocks_per_grid, threads_per_block>>>(
            d_phi1_hat, d_Na1_hat, d_Nb1_hat, d_Nc1_hat, d_Nd1_hat,
            d_E, d_f1, d_f2, d_f3, N_total);
        etdrk4_final<<<blocks_per_grid, threads_per_block>>>(
            d_phi2_hat, d_Na2_hat, d_Nb2_hat, d_Nc2_hat, d_Nd2_hat,
            d_E, d_f1, d_f2, d_f3, N_total);
    }

    std::cerr << "Simulation finished." << std::endl;

    // --- Cleanup ---
    cufftDestroy(plan);

    cudaFree(d_phi1_hat); cudaFree(d_phi2_hat);
    cudaFree(d_phi1_work); cudaFree(d_phi2_work);
    cudaFree(d_a1_hat); cudaFree(d_a2_hat);
    cudaFree(d_b1_hat); cudaFree(d_b2_hat);
    cudaFree(d_c1_hat); cudaFree(d_c2_hat);
    cudaFree(d_N1); cudaFree(d_N2);
    cudaFree(d_N1_hat); cudaFree(d_N2_hat);
    cudaFree(d_Na1_hat); cudaFree(d_Na2_hat);
    cudaFree(d_Nb1_hat); cudaFree(d_Nb2_hat);
    cudaFree(d_Nc1_hat); cudaFree(d_Nc2_hat);
    cudaFree(d_Nd1_hat); cudaFree(d_Nd2_hat);
    cudaFree(d_q2); cudaFree(d_E); cudaFree(d_E2);
    cudaFree(d_E2_1); cudaFree(d_f1); cudaFree(d_f2); cudaFree(d_f3);

    delete[] h_phi1; delete[] h_phi2;
    delete[] h_phi1_hat; delete[] h_phi2_hat;
    delete[] h_q2;
    delete[] h_E; delete[] h_E2; delete[] h_E2_1;
    delete[] h_f1; delete[] h_f2; delete[] h_f3;

    return 0;
}
