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
// CUDA Kernels
// ============================================================

// Initialize Fourier-space operators: q2, E (full step), E2 (half step),
// and ETDRK4 coefficients f1, f2, f3, E2_1 (half-step nonlinear weight)
// Coefficients are computed on the host via contour integrals and uploaded.
__global__ void init_q2_kernel(double* q2, int nx, int ny, double dx) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < nx && iy < ny) {
        int idx = iy * nx + ix;
        double kx = (ix < nx / 2 ? ix : ix - nx) * 2.0 * M_PI / (nx * dx);
        double ky = (iy < ny / 2 ? iy : iy - ny) * 2.0 * M_PI / (ny * dx);
        q2[idx] = kx * kx + ky * ky;
    }
}

// Compute nonlinear terms in real space (same as before)
__global__ void calc_nonlinear(cufftDoubleComplex* phi1, cufftDoubleComplex* phi2,
                               cufftDoubleComplex* N1, cufftDoubleComplex* N2,
                               int nx, int ny, double scale,
                               double chi, double chi_prime, double alpha,
                               double c11, double c12, double c21, double c22) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < nx && iy < ny) {
        int i = iy * nx + ix;

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

// Same as calc_nonlinear but does NOT overwrite phi arrays
// (used for RK sub-stages where phi is a temporary)
__global__ void calc_nonlinear_no_overwrite(
                               const cufftDoubleComplex* phi1,
                               const cufftDoubleComplex* phi2,
                               cufftDoubleComplex* N1, cufftDoubleComplex* N2,
                               int nx, int ny, double scale,
                               double chi, double chi_prime, double alpha,
                               double c11, double c12, double c21, double c22) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < nx && iy < ny) {
        int i = iy * nx + ix;

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

// Multiply nonlinear term by -q^2 in Fourier space (Cahn-Hilliard Laplacian)
__global__ void apply_laplacian(cufftDoubleComplex* N_hat, const double* q2,
                                int nx, int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < nx && iy < ny) {
        int i = iy * nx + ix;
        double mq2 = -q2[i];
        double re = N_hat[i].x * mq2;
        double im = N_hat[i].y * mq2;
        N_hat[i].x = re;
        N_hat[i].y = im;
    }
}

// ETDRK4 Stage a:  a_hat = E2 * u_hat + E2_1 * Na_hat
__global__ void etdrk4_stage_a(
    cufftDoubleComplex* a_hat,
    const cufftDoubleComplex* u_hat,
    const cufftDoubleComplex* Na_hat,
    const double* E2, const double* E2_1,
    int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        a_hat[i].x = E2[i] * u_hat[i].x + E2_1[i] * Na_hat[i].x;
        a_hat[i].y = E2[i] * u_hat[i].y + E2_1[i] * Na_hat[i].y;
    }
}

// ETDRK4 Stage b:  b_hat = E2 * u_hat + E2_1 * Nb_hat
__global__ void etdrk4_stage_b(
    cufftDoubleComplex* b_hat,
    const cufftDoubleComplex* u_hat,
    const cufftDoubleComplex* Nb_hat,
    const double* E2, const double* E2_1,
    int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        b_hat[i].x = E2[i] * u_hat[i].x + E2_1[i] * Nb_hat[i].x;
        b_hat[i].y = E2[i] * u_hat[i].y + E2_1[i] * Nb_hat[i].y;
    }
}

// ETDRK4 Stage c:  c_hat = E2 * a_hat + E2_1 * (2*Nc_hat - Na_hat)
__global__ void etdrk4_stage_c(
    cufftDoubleComplex* c_hat,
    const cufftDoubleComplex* a_hat,
    const cufftDoubleComplex* Nc_hat,
    const cufftDoubleComplex* Na_hat,
    const double* E2, const double* E2_1,
    int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double rhs_r = 2.0 * Nc_hat[i].x - Na_hat[i].x;
        double rhs_i = 2.0 * Nc_hat[i].y - Na_hat[i].y;
        c_hat[i].x = E2[i] * a_hat[i].x + E2_1[i] * rhs_r;
        c_hat[i].y = E2[i] * a_hat[i].y + E2_1[i] * rhs_i;
    }
}

// ETDRK4 Final update:
// u_hat^{n+1} = E * u_hat^n + f1*Na + 2*f2*(Nb + Nc) + f3*Nd
__global__ void etdrk4_final(
    cufftDoubleComplex* u_hat,
    const cufftDoubleComplex* Na_hat,
    const cufftDoubleComplex* Nb_hat,
    const cufftDoubleComplex* Nc_hat,
    const cufftDoubleComplex* Nd_hat,
    const double* E_full, const double* f1,
    const double* f2, const double* f3,
    int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
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
// (Kassam & Trefethen, SIAM J. Sci. Comput. 2005)
// ============================================================
void compute_etdrk4_coefficients(
    const double* h_q2, int N, double dt, double kappa,
    double* h_E, double* h_E2, double* h_E2_1,
    double* h_f1, double* h_f2, double* h_f3)
{
    const int M = 64; // number of contour points
    const double r = 1.0; // contour radius

    // Precompute contour points: z_j = r * exp(i * pi * (2j-1) / (2M))
    // placed in the upper half of the unit circle to stay in the left half-plane
    std::vector<std::complex<double>> contour_pts(M);
    for (int j = 0; j < M; j++) {
        double theta = M_PI * (2.0 * (j + 1) - 1.0) / (2.0 * M);
        contour_pts[j] = r * std::complex<double>(cos(theta), sin(theta));
    }

    for (int i = 0; i < N; i++) {
        double L = -kappa * h_q2[i] * h_q2[i]; // linear operator
        double Ldt = L * dt;
        double Ldt2 = L * dt * 0.5;

        // Full and half-step exponentials (exact, no cancellation issues)
        h_E[i]  = exp(Ldt);
        h_E2[i] = exp(Ldt2);

        // Contour integral evaluation for the four coefficients
        // E2_1 = (e^{Ldt/2} - 1) / L   (half-step nonlinear weight)
        // f1   = dt^{-2} * L^{-3} * [-4 - Ldt + e^{Ldt}(4 - 3Ldt + (Ldt)^2)]
        // f2   = dt^{-2} * L^{-3} * [2 + Ldt + e^{Ldt}(-2 + Ldt)]
        // f3   = dt^{-2} * L^{-3} * [-4 - 3Ldt - (Ldt)^2 + e^{Ldt}(4 - Ldt)]

        std::complex<double> sum_E2_1(0.0, 0.0);
        std::complex<double> sum_f1(0.0, 0.0);
        std::complex<double> sum_f2(0.0, 0.0);
        std::complex<double> sum_f3(0.0, 0.0);

        for (int j = 0; j < M; j++) {
            std::complex<double> z = Ldt + contour_pts[j]; // shift to Ldt
            std::complex<double> z2 = Ldt2 + contour_pts[j] * 0.5; // shift for half-step
            std::complex<double> ez = exp(z);
            std::complex<double> ez2 = exp(z2);

            // E2_1: (e^{z/2} - 1) / (z / dt)  but we want (e^{Ldt/2} - 1)/L
            // = dt * (e^{z2} - 1) / z  where z2 = z/2... 
            // Actually, let's be precise:
            // We want E2_1 = (e^{L*dt/2} - 1) / L
            // Using contour around Ldt/2: E2_1 = (dt/2) * (e^{z2} - 1) / z2
            sum_E2_1 += (ez2 - 1.0) / z2;

            // f1 = (-4 - z + e^z*(4 - 3z + z^2)) / (dt * z^3)
            // Note: we evaluate at z = Ldt + contour_pt, then average
            std::complex<double> z3 = z * z * z;
            sum_f1 += (-4.0 - z + ez * (4.0 - 3.0 * z + z * z)) / z3;

            // f2 = (2 + z + e^z*(-2 + z)) / (dt * z^3)
            sum_f2 += (2.0 + z + ez * (-2.0 + z)) / z3;

            // f3 = (-4 - 3z - z^2 + e^z*(4 - z)) / (dt * z^3)
            sum_f3 += (-4.0 - 3.0 * z - z * z + ez * (4.0 - z)) / z3;
        }

        h_E2_1[i] = (dt * 0.5) * (sum_E2_1 / (double)M).real();
        h_f1[i]   = dt * (sum_f1 / (double)M).real();
        h_f2[i]   = dt * (sum_f2 / (double)M).real();
        h_f3[i]   = dt * (sum_f3 / (double)M).real();
    }
}

// ============================================================
// Host helper for vorticity (unchanged from original)
// ============================================================
void get_flux(int x, int y, int nx, int ny, double dx,
              cufftDoubleComplex* h_phi1, cufftDoubleComplex* h_phi2,
              double &jx, double &jy) {
    int xp = (x + 1) % nx; int xm = (x - 1 + nx) % nx;
    int yp = (y + 1) % ny; int ym = (y - 1 + ny) % ny;

    double p1 = h_phi1[y * nx + x].x;
    double p2 = h_phi2[y * nx + x].x;
    double d1x = (h_phi1[y * nx + xp].x - h_phi1[y * nx + xm].x) / (2.0 * dx);
    double d1y = (h_phi1[yp * nx + x].x - h_phi1[ym * nx + x].x) / (2.0 * dx);
    double d2x = (h_phi2[y * nx + xp].x - h_phi2[y * nx + xm].x) / (2.0 * dx);
    double d2y = (h_phi2[yp * nx + x].x - h_phi2[ym * nx + x].x) / (2.0 * dx);

    jx = p1 * d2x - p2 * d1x;
    jy = p1 * d2y - p2 * d1y;
}

// ============================================================
// Helper: compute N_hat from phi_hat (IFFT -> nonlinear -> FFT -> Laplacian)
// If overwrite_phi is true, the IFFT result overwrites d_phi (with normalization)
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
    int nx, int ny, int N_total,
    double chi, double chi_prime, double alpha,
    double c11, double c12, double c21, double c22,
    dim3 blocks2d, dim3 threads2d, int threads1d_block,
    bool overwrite_phi,
    cufftDoubleComplex* d_phi1_out = nullptr,
    cufftDoubleComplex* d_phi2_out = nullptr)
{
    double scale = 1.0 / N_total;

    // Copy input hat to work arrays for IFFT (don't destroy input)
    CUDA_CHECK(cudaMemcpy(d_phi1_work, d_phi1_hat_in,
                          sizeof(cufftDoubleComplex) * N_total, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_phi2_work, d_phi2_hat_in,
                          sizeof(cufftDoubleComplex) * N_total, cudaMemcpyDeviceToDevice));

    // IFFT
    cufftExecZ2Z(plan, d_phi1_work, d_phi1_work, CUFFT_INVERSE);
    cufftExecZ2Z(plan, d_phi2_work, d_phi2_work, CUFFT_INVERSE);

    if (overwrite_phi && d_phi1_out && d_phi2_out) {
        // Use the version that overwrites phi arrays (for diagnostics)
        calc_nonlinear<<<blocks2d, threads2d>>>(
            d_phi1_work, d_phi2_work, d_N1, d_N2,
            nx, ny, scale, chi, chi_prime, alpha, c11, c12, c21, c22);
        // Copy normalized real-space fields out
        CUDA_CHECK(cudaMemcpy(d_phi1_out, d_phi1_work,
                              sizeof(cufftDoubleComplex) * N_total, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_phi2_out, d_phi2_work,
                              sizeof(cufftDoubleComplex) * N_total, cudaMemcpyDeviceToDevice));
    } else {
        calc_nonlinear_no_overwrite<<<blocks2d, threads2d>>>(
            d_phi1_work, d_phi2_work, d_N1, d_N2,
            nx, ny, scale, chi, chi_prime, alpha, c11, c12, c21, c22);
    }

    // FFT the nonlinear terms
    cufftExecZ2Z(plan, d_N1, d_N1_hat, CUFFT_FORWARD);
    cufftExecZ2Z(plan, d_N2, d_N2_hat, CUFFT_FORWARD);

    // Apply -q^2 (Cahn-Hilliard Laplacian in Fourier space)
    int blocks1d = (N_total + threads1d_block - 1) / threads1d_block;
    apply_laplacian<<<blocks1d, threads1d_block>>>(d_N1_hat, d_q2, nx, ny);
    apply_laplacian<<<blocks1d, threads1d_block>>>(d_N2_hat, d_q2, nx, ny);
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

    const int nx          = getI("nx", 128);
    const int ny          = getI("ny", 128);
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

    const int N = nx * ny;

    // --- Initialize fields ---
    srand(seed);
    cufftDoubleComplex* h_phi1     = new cufftDoubleComplex[N];
    cufftDoubleComplex* h_phi2     = new cufftDoubleComplex[N];
    cufftDoubleComplex* h_phi1_hat = new cufftDoubleComplex[N];
    cufftDoubleComplex* h_phi2_hat = new cufftDoubleComplex[N];

    for (int i = 0; i < N; i++) {
        h_phi1[i].x = 0.35 + 0.01 * ((double)rand() / RAND_MAX - 0.5);
        h_phi1[i].y = 0.0;
        h_phi2[i].x = 0.30 + 0.01 * ((double)rand() / RAND_MAX - 0.5);
        h_phi2[i].y = 0.0;
    }

    // --- Compute q^2 on host for coefficient computation ---
    double* h_q2 = new double[N];
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            double kx = (ix < nx / 2 ? ix : ix - nx) * 2.0 * M_PI / (nx * dx);
            double ky = (iy < ny / 2 ? iy : iy - ny) * 2.0 * M_PI / (ny * dx);
            h_q2[iy * nx + ix] = kx * kx + ky * ky;
        }
    }

    // --- Compute ETDRK4 coefficients on host ---
    double* h_E    = new double[N];
    double* h_E2   = new double[N];
    double* h_E2_1 = new double[N];
    double* h_f1   = new double[N];
    double* h_f2   = new double[N];
    double* h_f3   = new double[N];

    std::cerr << "Computing ETDRK4 coefficients via contour integrals (M=64)..."
              << std::endl;
    compute_etdrk4_coefficients(h_q2, N, dt, kappa,
                                h_E, h_E2, h_E2_1, h_f1, h_f2, h_f3);
    std::cerr << "Done." << std::endl;

    // --- Allocate device memory ---
    // Field arrays (2 species x {hat, real-space work, stage temporaries})
    cufftDoubleComplex *d_phi1_hat, *d_phi2_hat;       // current state
    cufftDoubleComplex *d_phi1_work, *d_phi2_work;     // IFFT workspace
    cufftDoubleComplex *d_a1_hat, *d_a2_hat;           // stage a
    cufftDoubleComplex *d_b1_hat, *d_b2_hat;           // stage b
    cufftDoubleComplex *d_c1_hat, *d_c2_hat;           // stage c
    cufftDoubleComplex *d_N1, *d_N2;                   // real-space nonlinear
    cufftDoubleComplex *d_N1_hat, *d_N2_hat;           // current N_hat
    cufftDoubleComplex *d_Na1_hat, *d_Na2_hat;         // N_hat at stage a (= N1)
    cufftDoubleComplex *d_Nb1_hat, *d_Nb2_hat;         // N_hat at stage b
    cufftDoubleComplex *d_Nc1_hat, *d_Nc2_hat;         // N_hat at stage c
    cufftDoubleComplex *d_Nd1_hat, *d_Nd2_hat;         // N_hat at stage d
    cufftDoubleComplex *d_phi1_real, *d_phi2_real;     // for diagnostics output

    double *d_q2, *d_E, *d_E2, *d_E2_1, *d_f1, *d_f2, *d_f3;

    size_t csize = sizeof(cufftDoubleComplex) * N;
    size_t dsize = sizeof(double) * N;

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
    CUDA_CHECK(cudaMalloc(&d_phi1_real, csize));
    CUDA_CHECK(cudaMalloc(&d_phi2_real, csize));

    CUDA_CHECK(cudaMalloc(&d_q2, dsize));
    CUDA_CHECK(cudaMalloc(&d_E, dsize));
    CUDA_CHECK(cudaMalloc(&d_E2, dsize));
    CUDA_CHECK(cudaMalloc(&d_E2_1, dsize));
    CUDA_CHECK(cudaMalloc(&d_f1, dsize));
    CUDA_CHECK(cudaMalloc(&d_f2, dsize));
    CUDA_CHECK(cudaMalloc(&d_f3, dsize));

    // Upload initial fields
    CUDA_CHECK(cudaMemcpy(d_phi1_work, h_phi1, csize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_phi2_work, h_phi2, csize, cudaMemcpyHostToDevice));

    // Upload coefficients
    CUDA_CHECK(cudaMemcpy(d_q2, h_q2, dsize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E, h_E, dsize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E2, h_E2, dsize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E2_1, h_E2_1, dsize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f1, h_f1, dsize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f2, h_f2, dsize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_f3, h_f3, dsize, cudaMemcpyHostToDevice));

    // --- Create FFT plan ---
    cufftHandle plan;
    cufftPlan2d(&plan, ny, nx, CUFFT_Z2Z);

    // --- FFT initial conditions to get phi_hat ---
    cufftExecZ2Z(plan, d_phi1_work, d_phi1_hat, CUFFT_FORWARD);
    cufftExecZ2Z(plan, d_phi2_work, d_phi2_hat, CUFFT_FORWARD);

    // --- Kernel launch configs ---
    dim3 threads2d(16, 16);
    dim3 blocks2d((nx + 15) / 16, (ny + 15) / 16);
    int threads1d = 256;

    std::cout << "Step\tTime\tAlpha\tFree_Energy\tRMS_Vorticity\tDomain_Size_R\n";

    // ============================================================
    // Main time loop
    // ============================================================
    for (int step = 0; step <= n_steps; step++) {

        double prog = (double)step / n_steps;
        double alpha = getD("alpha_start", 0.0)
                     + prog * (getD("alpha_end", 0.45) - getD("alpha_start", 0.0));
        double c11 = getD("c11_start", 0.2)
                   + prog * (getD("c11_end", 0.2) - getD("c11_start", 0.2));
        double c12 = getD("c12_start", 0.5)
                   + prog * (getD("c12_end", 0.5) - getD("c12_start", 0.5));
        double c21 = getD("c21_start", 0.1)
                   + prog * (getD("c21_end", 0.1) - getD("c21_start", 0.1));
        double c22 = getD("c22_start", 0.5)
                   + prog * (getD("c22_end", 0.5) - getD("c22_start", 0.5));

        // --- Diagnostics (before stepping) ---
        bool do_log   = (step % log_interval == 0);
        bool do_field = (save_fields && step % field_interval == 0);

        if (do_log || do_field) {
            // Get real-space fields via IFFT of current state
            CUDA_CHECK(cudaMemcpy(d_phi1_work, d_phi1_hat, csize,
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_phi2_work, d_phi2_hat, csize,
                                  cudaMemcpyDeviceToDevice));
            cufftExecZ2Z(plan, d_phi1_work, d_phi1_work, CUFFT_INVERSE);
            cufftExecZ2Z(plan, d_phi2_work, d_phi2_work, CUFFT_INVERSE);

            // Normalize on host
            CUDA_CHECK(cudaMemcpy(h_phi1, d_phi1_work, csize,
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_phi2, d_phi2_work, csize,
                                  cudaMemcpyDeviceToHost));
            for (int i = 0; i < N; i++) {
                h_phi1[i].x /= N; h_phi1[i].y = 0.0;
                h_phi2[i].x /= N; h_phi2[i].y = 0.0;
            }

            if (do_log) {
                CUDA_CHECK(cudaMemcpy(h_phi1_hat, d_phi1_hat, csize,
                                      cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_phi2_hat, d_phi2_hat, csize,
                                      cudaMemcpyDeviceToHost));

                double F_bulk = 0.0, sum_vort_sq = 0.0;
                double sum_S = 0.0, sum_qS = 0.0, F_grad = 0.0;

                for (int y = 0; y < ny; y++) {
                    for (int x = 0; x < nx; x++) {
                        int i = y * nx + x;
                        double p1 = h_phi1[i].x, p2 = h_phi2[i].x;

                        F_bulk += (pow(p1-c11,2)*pow(p1-c12,2)
                                 + pow(p2-c21,2)*pow(p2-c22,2)
                                 + chi*p1*p2
                                 + chi_prime*p1*p1*p2*p2) * (dx*dx);

                        double jx_xp,jy_xp,jx_xm,jy_xm,jx_yp,jy_yp,jx_ym,jy_ym;
                        get_flux((x+1)%nx, y, nx, ny, dx, h_phi1, h_phi2,
                                 jx_xp, jy_xp);
                        get_flux((x-1+nx)%nx, y, nx, ny, dx, h_phi1, h_phi2,
                                 jx_xm, jy_xm);
                        get_flux(x, (y+1)%ny, nx, ny, dx, h_phi1, h_phi2,
                                 jx_yp, jy_yp);
                        get_flux(x, (y-1+ny)%ny, nx, ny, dx, h_phi1, h_phi2,
                                 jx_ym, jy_ym);
                        double omega = (jy_xp - jy_xm)/(2.0*dx)
                                     - (jx_yp - jx_ym)/(2.0*dx);
                        sum_vort_sq += omega * omega;

                        double kx_val = (x < nx/2 ? x : x-nx) * 2.0*M_PI/(nx*dx);
                        double ky_val = (y < ny/2 ? y : y-ny) * 2.0*M_PI/(ny*dx);
                        double q2_val = kx_val*kx_val + ky_val*ky_val;
                        double S1 = pow(h_phi1_hat[i].x/N, 2)
                                  + pow(h_phi1_hat[i].y/N, 2);
                        double S2 = pow(h_phi2_hat[i].x/N, 2)
                                  + pow(h_phi2_hat[i].y/N, 2);

                        sum_S  += S1;
                        sum_qS += sqrt(q2_val) * S1;
                        F_grad += 0.5 * kappa * q2_val * (S1 + S2) * (dx*dx);
                    }
                }

                std::cout << std::fixed << std::setprecision(6)
                          << step << "\t" << step*dt << "\t" << alpha << "\t"
                          << (F_bulk + F_grad) << "\t"
                          << sqrt(sum_vort_sq / N) << "\t"
                          << ((sum_qS > 0) ? (2.0*M_PI*sum_S/sum_qS) : 0.0)
                          << "\n";

                std::cerr << "Step " << step << " | Alpha: " << alpha
                          << " | F: " << (F_bulk + F_grad) << "\n";
            }

            if (do_field) {
                char filename[256];
                sprintf(filename, "%s%010d.h5", prefix.c_str(), step);

                hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC,
                                          H5P_DEFAULT, H5P_DEFAULT);
                hsize_t dims[2] = {(hsize_t)ny, (hsize_t)nx};
                hid_t dspace = H5Screate_simple(2, dims, NULL);

                double* out1 = new double[N];
                double* out2 = new double[N];
                for (int i = 0; i < N; i++) {
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

        // Skip the actual time step on the last iteration
        if (step == n_steps) break;

        // ========================================================
        // ETDRK4 Time Step
        // ========================================================
        int blocks1d = (N + threads1d - 1) / threads1d;

        // --- Stage 1: Compute Na = N(u^n) ---
        compute_N_hat(plan,
                      d_phi1_hat, d_phi2_hat,       // input hat
                      d_phi1_work, d_phi2_work,     // IFFT workspace
                      d_N1, d_N2, d_Na1_hat, d_Na2_hat,
                      d_q2, nx, ny, N,
                      chi, chi_prime, alpha, c11, c12, c21, c22,
                      blocks2d, threads2d, threads1d,
                      false);

        // --- Stage a: a_hat = E2 * u_hat + E2_1 * Na_hat ---
        etdrk4_stage_a<<<blocks1d, threads1d>>>(
            d_a1_hat, d_phi1_hat, d_Na1_hat, d_E2, d_E2_1, N);
        etdrk4_stage_a<<<blocks1d, threads1d>>>(
            d_a2_hat, d_phi2_hat, d_Na2_hat, d_E2, d_E2_1, N);

        // --- Stage 2: Compute Nb = N(a) ---
        compute_N_hat(plan,
                      d_a1_hat, d_a2_hat,
                      d_phi1_work, d_phi2_work,
                      d_N1, d_N2, d_Nb1_hat, d_Nb2_hat,
                      d_q2, nx, ny, N,
                      chi, chi_prime, alpha, c11, c12, c21, c22,
                      blocks2d, threads2d, threads1d,
                      false);

        // --- Stage b: b_hat = E2 * u_hat + E2_1 * Nb_hat ---
        etdrk4_stage_b<<<blocks1d, threads1d>>>(
            d_b1_hat, d_phi1_hat, d_Nb1_hat, d_E2, d_E2_1, N);
        etdrk4_stage_b<<<blocks1d, threads1d>>>(
            d_b2_hat, d_phi2_hat, d_Nb2_hat, d_E2, d_E2_1, N);

        // --- Stage 3: Compute Nc = N(b) ---
        compute_N_hat(plan,
                      d_b1_hat, d_b2_hat,
                      d_phi1_work, d_phi2_work,
                      d_N1, d_N2, d_Nc1_hat, d_Nc2_hat,
                      d_q2, nx, ny, N,
                      chi, chi_prime, alpha, c11, c12, c21, c22,
                      blocks2d, threads2d, threads1d,
                      false);

        // --- Stage c: c_hat = E2 * a_hat + E2_1 * (2*Nc - Na) ---
        etdrk4_stage_c<<<blocks1d, threads1d>>>(
            d_c1_hat, d_a1_hat, d_Nc1_hat, d_Na1_hat, d_E2, d_E2_1, N);
        etdrk4_stage_c<<<blocks1d, threads1d>>>(
            d_c2_hat, d_a2_hat, d_Nc2_hat, d_Na2_hat, d_E2, d_E2_1, N);

        // --- Stage 4: Compute Nd = N(c) ---
        compute_N_hat(plan,
                      d_c1_hat, d_c2_hat,
                      d_phi1_work, d_phi2_work,
                      d_N1, d_N2, d_Nd1_hat, d_Nd2_hat,
                      d_q2, nx, ny, N,
                      chi, chi_prime, alpha, c11, c12, c21, c22,
                      blocks2d, threads2d, threads1d,
                      false);

        // --- Final update ---
        // u^{n+1} = E*u^n + f1*Na + 2*f2*(Nb + Nc) + f3*Nd
        etdrk4_final<<<blocks1d, threads1d>>>(
            d_phi1_hat, d_Na1_hat, d_Nb1_hat, d_Nc1_hat, d_Nd1_hat,
            d_E, d_f1, d_f2, d_f3, N);
        etdrk4_final<<<blocks1d, threads1d>>>(
            d_phi2_hat, d_Na2_hat, d_Nb2_hat, d_Nc2_hat, d_Nd2_hat,
            d_E, d_f1, d_f2, d_f3, N);
    }

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
    cudaFree(d_phi1_real); cudaFree(d_phi2_real);
    cudaFree(d_q2); cudaFree(d_E); cudaFree(d_E2);
    cudaFree(d_E2_1); cudaFree(d_f1); cudaFree(d_f2); cudaFree(d_f3);

    delete[] h_phi1; delete[] h_phi2;
    delete[] h_phi1_hat; delete[] h_phi2_hat;
    delete[] h_q2;
    delete[] h_E; delete[] h_E2; delete[] h_E2_1;
    delete[] h_f1; delete[] h_f2; delete[] h_f3;

    return 0;
}

