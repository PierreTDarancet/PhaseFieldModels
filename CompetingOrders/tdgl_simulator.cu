#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cufft.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

// --- Thrust Functors for absolute mean and cross-correlation ---
struct abs_functor {
    __host__ __device__ float operator()(const float& x) const { return fabsf(x); }
};

struct mult_functor {
    __host__ __device__ float operator()(const float& x, const float& y) const { return x * y; }
};

// --- CUDA Kernels ---

// Initialize cuRAND states
__global__ void init_rand_kernel(curandState *state, unsigned long seed, int Nx, int Ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;
    if (x < Nx && y < Ny) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

// Initialize fields with equilibrium values + noise
__global__ void init_fields_kernel(float *psi1, float *psi2, curandState *state, 
                                   float p1_eq, float p2_eq, int Nx, int Ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * Nx + x;
    if (x < Nx && y < Ny) {
        psi1[idx] = p1_eq + 0.05f * curand_normal(&state[idx]);
        psi2[idx] = p2_eq + 0.05f * curand_normal(&state[idx]);
    }
}

// Main TDGL Euler-Maruyama Step
__global__ void tdgl_step_kernel(float *p1, float *p2, float *p1_new, float *p2_new, curandState *state,
                                 int Nx, int Ny, float dx, float dt, float a1_eq, float current_a2,
                                 float b1, float b2, float u1, float u2, float c, float Gamma1, float Gamma2,
                                 float noise_scale1, float noise_scale2) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < Nx && y < Ny) {
        int idx = y * Nx + x;
        
        // Periodic boundary indices
        int xp = (x + 1) % Nx;
        int xm = (x - 1 + Nx) % Nx;
        int yp = (y + 1) % Ny;
        int ym = (y - 1 + Ny) % Ny;
        
        float v1 = p1[idx];
        float v2 = p2[idx];
        
        // Laplacian
        float lap1 = (p1[y * Nx + xp] + p1[y * Nx + xm] + p1[yp * Nx + x] + p1[ym * Nx + x] - 4.0f * v1) / (dx * dx);
        float lap2 = (p2[y * Nx + xp] + p2[y * Nx + xm] + p2[yp * Nx + x] + p2[ym * Nx + x] - 4.0f * v2) / (dx * dx);
        
        // Free energy derivatives
        float dF1 = a1_eq * v1 - b1 * lap1 + 4.0f * u1 * v1 * v1 * v1 + 2.0f * c * v1 * v2 * v2;
        float dF2 = current_a2 * v2 - b2 * lap2 + 4.0f * u2 * v2 * v2 * v2 + 2.0f * c * v2 * v1 * v1;
        
        // Langevin noise
        float eta1 = noise_scale1 * curand_normal(&state[idx]);
        float eta2 = noise_scale2 * curand_normal(&state[idx]);
        
        // Update
        p1_new[idx] = v1 - dt * Gamma1 * dF1 + eta1;
        p2_new[idx] = v2 - dt * Gamma2 * dF2 + eta2;
    }
}

// Compute fluctuations (delta psi) and cast to cuFFT complex type
__global__ void prep_fft_kernel(float *psi, cufftComplex *fft_in, float mean_val, int Nx, int Ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < Nx && y < Ny) {
        int idx = y * Nx + x;
        fft_in[idx].x = psi[idx] - mean_val; // Real part is fluctuation
        fft_in[idx].y = 0.0f;                // Imaginary part is zero
    }
}

// Extract fluctuations for cross-correlation
__global__ void calc_fluctuations_kernel(float *p1, float *p2, float *dp1, float *dp2, 
                                         float m1, float m2, int Nx, int Ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < Nx && y < Ny) {
        int idx = y * Nx + x;
        dp1[idx] = p1[idx] - m1;
        dp2[idx] = p2[idx] - m2;
    }
}

// --- Main Program ---
int main() {
    // 1. Parse Input from stdin
    std::map<std::string, std::string> params;
    std::string line;
    while (std::getline(std::cin, line)) {
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) line = line.substr(0, comment_pos);
        size_t eq_pos = line.find('=');
        if (eq_pos != std::string::npos) {
            std::string key = line.substr(0, eq_pos);
            std::string val = line.substr(eq_pos + 1);
            key.erase(0, key.find_first_not_of(" \t")); key.erase(key.find_last_not_of(" \t") + 1);
            val.erase(0, val.find_first_not_of(" \t")); val.erase(val.find_last_not_of(" \t") + 1);
            params[key] = val;
        }
    }

    // Default parameters
    int Nx = params.count("Nx") ? std::stoi(params["Nx"]) : 100;
    int Ny = params.count("Ny") ? std::stoi(params["Ny"]) : 100;
    int N = Nx * Ny;
    float dx = params.count("dx") ? std::stof(params["dx"]) : 1.0f;
    float dt = params.count("dt") ? std::stof(params["dt"]) : 0.001f;
    int time_steps = params.count("time_steps") ? std::stoi(params["time_steps"]) : 20000;

    float a1_eq = params.count("a1_eq") ? std::stof(params["a1_eq"]) : -1.0f;
    float a2_eq = params.count("a2_eq") ? std::stof(params["a2_eq"]) : -1.1f;
    float a2_prime = params.count("a2_prime") ? std::stof(params["a2_prime"]) : 3.5f;
    float b1 = params.count("b1") ? std::stof(params["b1"]) : 7.0f;
    float b2 = params.count("b2") ? std::stof(params["b2"]) : 2.0f;
    float u1 = params.count("u1") ? std::stof(params["u1"]) : 2.0f;
    float u2 = params.count("u2") ? std::stof(params["u2"]) : 1.0f;
    float c = params.count("c") ? std::stof(params["c"]) : 1.5f;
    float Gamma1 = params.count("Gamma1") ? std::stof(params["Gamma1"]) : 2.5f;
    float Gamma2 = params.count("Gamma2") ? std::stof(params["Gamma2"]) : 1.5f;
    float tau_0 = params.count("tau_0") ? std::stof(params["tau_0"]) : 0.1f;
    float tc = params.count("tc") ? std::stof(params["tc"]) : 2.0f;
    float Tv = params.count("Tv") ? std::stof(params["Tv"]) : 0.07f;
    int print_interval = params.count("print_interval") ? std::stoi(params["print_interval"]) : 100;
    
    unsigned long seed = params.count("seed") ? std::stoul(params["seed"]) : 42;
    std::string prefix = params.count("output_prefix") ? params["output_prefix"] : "sim_out";

    float noise1 = sqrtf(Gamma1 * Tv * dt / (dx * dx));
    float noise2 = sqrtf(Gamma2 * Tv * dt / (dx * dx));

    // 2. Allocate Device Memory
    size_t size = N * sizeof(float);
    float *d_p1, *d_p2, *d_p1_new, *d_p2_new, *d_dp1, *d_dp2;
    cudaMalloc(&d_p1, size);
    cudaMalloc(&d_p2, size);
    cudaMalloc(&d_p1_new, size);
    cudaMalloc(&d_p2_new, size);
    cudaMalloc(&d_dp1, size);
    cudaMalloc(&d_dp2, size);

    curandState *d_state;
    cudaMalloc(&d_state, N * sizeof(curandState));

    cufftComplex *d_fft_in, *d_fft_out;
    cudaMalloc(&d_fft_in, N * sizeof(cufftComplex));
    cudaMalloc(&d_fft_out, N * sizeof(cufftComplex));

    cufftHandle plan;
    cufftPlan2d(&plan, Nx, Ny, CUFFT_C2C);

    // Block/Grid configuration
    dim3 threads(16, 16);
    dim3 blocks((Nx + threads.x - 1) / threads.x, (Ny + threads.y - 1) / threads.y);

    // 3. Initialize
    float p1_eq = sqrtf(fabsf(a1_eq) / (4.0f * u1));
    float p2_eq = sqrtf(fabsf(a2_eq) / (4.0f * u2));
    
    init_rand_kernel<<<blocks, threads>>>(d_state, seed, Nx, Ny);
    init_fields_kernel<<<blocks, threads>>>(d_p1, d_p2, d_state, p1_eq, p2_eq, Nx, Ny);

    thrust::device_ptr<float> ptr_p1(d_p1);
    thrust::device_ptr<float> ptr_p2(d_p2);
    thrust::device_ptr<float> ptr_dp1(d_dp1);
    thrust::device_ptr<float> ptr_dp2(d_dp2);

    // Output File
    std::string out_file = prefix + "_timeseries.dat";
    std::ofstream fout(out_file);
    fout << "Time(ps) Mean_Psi1 Mean_Psi2 CrossCorr_S12 Xi_CDW\n";

    std::cout << "Starting CUDA Simulation. Prefix: " << prefix << std::endl;

    // 4. Time Loop
    for (int step = 0; step <= time_steps; ++step) {
        float t = step * dt;
        float current_a2 = (t < 0) ? a2_eq : ((t < tau_0) ? a2_prime : a2_eq + (a2_prime - a2_eq) * expf(-(t - tau_0) / tc));

        tdgl_step_kernel<<<blocks, threads>>>(d_p1, d_p2, d_p1_new, d_p2_new, d_state,
                                              Nx, Ny, dx, dt, a1_eq, current_a2,
                                              b1, b2, u1, u2, c, Gamma1, Gamma2, noise1, noise2);
        
        // Swap pointers
        float *temp1 = d_p1; d_p1 = d_p1_new; d_p1_new = temp1;
        float *temp2 = d_p2; d_p2 = d_p2_new; d_p2_new = temp2;
        ptr_p1 = thrust::device_pointer_cast(d_p1);
        ptr_p2 = thrust::device_pointer_cast(d_p2);

        if (step % print_interval == 0) {
            // Compute Means
            float sum_p1 = thrust::transform_reduce(ptr_p1, ptr_p1 + N, abs_functor(), 0.0f, thrust::plus<float>());
            float sum_p2 = thrust::transform_reduce(ptr_p2, ptr_p2 + N, abs_functor(), 0.0f, thrust::plus<float>());
            float mean_p1 = sum_p1 / N;
            float mean_p2 = sum_p2 / N;

            // Compute Cross-Correlation S^12
            float raw_mean1 = thrust::reduce(ptr_p1, ptr_p1 + N) / N;
            float raw_mean2 = thrust::reduce(ptr_p2, ptr_p2 + N) / N;
            calc_fluctuations_kernel<<<blocks, threads>>>(d_p1, d_p2, d_dp1, d_dp2, raw_mean1, raw_mean2, Nx, Ny);
            float sum_s12 = thrust::inner_product(ptr_dp1, ptr_dp1 + N, ptr_dp2, 0.0f);
            float s12 = sum_s12 / N;

            // Compute Correlation Length Xi via cuFFT
            prep_fft_kernel<<<blocks, threads>>>(d_p1, d_fft_in, raw_mean1, Nx, Ny);
            cufftExecC2C(plan, d_fft_in, d_fft_out, CUFFT_FORWARD);
            
            cufftComplex *h_fft_out = new cufftComplex[N];
            cudaMemcpy(h_fft_out, d_fft_out, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
            
            float S_0 = h_fft_out[0].x * h_fft_out[0].x + h_fft_out[0].y * h_fft_out[0].y;
            float S_1 = 0.25f * (
                (h_fft_out[1].x * h_fft_out[1].x + h_fft_out[1].y * h_fft_out[1].y) +
                (h_fft_out[Nx-1].x * h_fft_out[Nx-1].x + h_fft_out[Nx-1].y * h_fft_out[Nx-1].y) +
                (h_fft_out[Nx].x * h_fft_out[Nx].x + h_fft_out[Nx].y * h_fft_out[Nx].y) +
                (h_fft_out[(Ny-1)*Nx].x * h_fft_out[(Ny-1)*Nx].x + h_fft_out[(Ny-1)*Nx].y * h_fft_out[(Ny-1)*Nx].y)
            );
            
            float xi = 0.0f;
            if (S_1 > 0 && S_0 > S_1) {
                float q1 = 2.0f * M_PI / (Nx * dx);
                xi = (1.0f / q1) * sqrtf((S_0 / S_1) - 1.0f);
            }
            delete[] h_fft_out;

            printf("Step: %8d | t: %8.3f | Psi1: %6.4f | Psi2: %6.4f | S12: %7.4f | Xi: %6.4f\n", 
                   step, t, mean_p1, mean_p2, s12, xi);
            fout << t << " " << mean_p1 << " " << mean_p2 << " " << s12 << " " << xi << "\n";
        }
    }

    // Cleanup
    fout.close();
    cufftDestroy(plan);
    cudaFree(d_p1); cudaFree(d_p2); cudaFree(d_p1_new); cudaFree(d_p2_new);
    cudaFree(d_dp1); cudaFree(d_dp2); cudaFree(d_state);
    cudaFree(d_fft_in); cudaFree(d_fft_out);

    return 0;
}

