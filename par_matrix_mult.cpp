#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/parallel/algorithms/generate.hpp>
#include <iostream>
#include <chrono>
#include <vector>


using matrix = std::vector<std::vector<int>>;
using time_result = std::pair<std::chrono::duration<double>, matrix>;


int inner_loop_parallel(const matrix& M1, const matrix& M2, size_t row, size_t col, size_t n)
{
    std::size_t partial_result = 0;
    hpx::experimental::for_loop(hpx::execution::par, 0, n, [&](auto i) { partial_result += M1[row][i] * M2[i][col]; });

    return partial_result;
}


time_result sequential_multiplication(const matrix& m1, const matrix& m2, const size_t n)
{
    matrix result(n, std::vector<int>(n));
    
    // initializing result with 0
    for (size_t i = 0; i < n; i++){
        std::fill(result[i].begin(), result[i].end(), 0);
    }

    // start time before computing
    auto before = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            result[i][j] += inner_loop_parallel(m1, m2, i, j, n);
        }
    }
    auto after = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_taken = after - before;

    return std::make_pair(time_taken, result);
}


time_result parallel_multiplication(const matrix& m1, const matrix& m2, const size_t n)
{
    matrix result(n, std::vector<int>(n));
    // initializing result with 0
    for (size_t i = 0; i < n; i++) {
        std::fill(result[i].begin(), result[i].end(), 0);
    }

    // start time before computing
    auto before = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < n; i++)
    {
        hpx::experimental::for_loop(hpx::execution::par, 0, n,
            [&](auto j) { result[i][j] = inner_loop_parallel(m1, m2, i, j, n); });
    }
    auto after = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_taken = after - before;

    return std::make_pair(time_taken, result);
}


int hpx_main(hpx::program_options::variables_map& vm)
{
    std::size_t n = vm["n"].as<std::size_t>();

    // initializing 2 matrices
    matrix m1(n, std::vector<int>(n));
    matrix m2(n, std::vector<int>(n));
    
    // filling matrices with random values
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            m1[i][j] = rand() % 50;
            m2[i][j] = rand() % 50;
        }
    }
    
    time_result seq_res = sequential_multiplication(m1, m2, n);
    time_result par_res = parallel_multiplication(m1, m2, n);

    std::cout << "Time by Sequential : " << seq_res.first.count()<< "s\n";
    std::cout << "Time by Parallel : " << par_res.first.count()<< "s\n";

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    using namespace hpx::program_options;

    options_description desc_commandline;
    desc_commandline.add_options()("n", value<std::size_t>()->default_value(10), "Matrix dimension");

    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;

    return hpx::init(argc, argv, init_args);
}
