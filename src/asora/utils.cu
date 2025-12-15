#include "utils.cuh"

#include <exception>
#include <format>
#include <iostream>

namespace {

    __host__ __device__ cuda::std::array<int, 2> get_ij(int q, int s) {
        auto j = (s - 1) / (2 * q);
        auto i = (s - 1) % (2 * q) + j - q;
        if (i + j > q) {
            i -= q;
            j -= q + 1;
        }
        return {i, j};
    }

}  // namespace

namespace asora {

    void safe_cuda(cudaError_t err, const std::source_location &loc) {
        if (err != cudaSuccess) {
            auto msg = std::format(
                "CUDA Error {}: {}. At {} in {}:{}", cudaGetErrorName(err),
                cudaGetErrorString(err), loc.function_name(), loc.file_name(),
                loc.line()
            );
            std::cerr << msg << "\n";
            throw std::runtime_error(msg);
        }
    }

    __host__ __device__ cuda::std::array<int, 3> linthrd2cart(int q, int s) {
        if (s == 0) return {q, 0, 0};

        auto s_top = 2 * q * (q + 1) + 1;
        if (s == s_top) return {q - 1, 0, -1};

        int sign = 1;
        int q1 = q;
        if (s > s_top) {
            s -= s_top;
            q1 -= 1;
            sign = -1;
        }

        auto &&[i, j] = get_ij(q1, s);
        return {i, j, sign * (q - abs(i) - abs(j))};
    }

    __host__ __device__ cuda::std::array<int, 2> cart2linthrd(int i, int j, int k) {
        auto q = abs(i) + abs(j) + abs(k);
        if (i == q && j == 0 && k == 0) return {q, 0};

        auto s_top = 2 * q * (q + 1) + 1;
        if (i == q - 1 && j == 0 && k == -1) return {q, s_top};

        auto q1 = q;
        auto s_off = 0;
        if (k < 0) {
            q1 -= 1;
            s_off = s_top;
        }

        // Guess a solution
        auto s = (2 * q1 - 1) * j + i + q1 + 1;
        if (s > 0) {
            auto &&[i0, j0] = get_ij(q1, s);
            if (i0 == i && j0 == j) return {q, s + s_off};
        }

        // It's the other solution
        s += 2 * q1 * (q1 + 1) - 1 + s_off;
        return {q, s};
    }

    __host__ __device__ size_t cells_in_shell(int q) {
        if (q < 0) return 0;
        if (q == 0) return 1;
        return 4 * q * q + 2;
    }

    // Return the cumulative number of cells up to the shell.
    __host__ __device__ size_t cells_to_shell(int q) {
        if (q < 0) return 0;
        return (1 + 2 * q) * (3 + 2 * q * (1 + q)) / 3;
    }

}  // namespace asora
