#include <algorithm>
#include <chrono>
#include <fstream>
#include <gmp.h>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

std::vector<unsigned int> read_primes_from_file(const std::string &filename) {
  std::vector<unsigned int> primes;
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Unable to open file " << filename << std::endl;
    return primes;
  }

  unsigned int prime;
  while (file >> prime)
    primes.push_back(prime);
  file.close();
  return primes;
}

__global__ void kershaw_prime_kernel(unsigned int* primes, unsigned int* results, std::size_t size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    unsigned int p = primes[idx];
    unsigned int val = 2, order = 1;

    while (val != 1) {
      val = (val << 1) % p;
      
      order++;
    }

    unsigned long long long_val = 3;
    unsigned int count = 1;
    const long long max_val = ULLONG_MAX / 3 - 2;

    while (count < order) {
      ++count;

      if (max_val < long_val)
        long_val = long_val % p;

      long_val = long_val * 3;
    }

    unsigned int base = long_val % p;

    val = base;
    count = 1;
    unsigned int limit = (p - 1) / order;

    while (count < limit) {
      val = val * base;
      count++;

      if (val > p) {
        val = val % p;
      }

      unsigned int result = val;
      if (3 * result > p) {
        result = (3 * result) % p;
        if (result == 2) {
          results[idx] = p;
          break;
        }
      }
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: <filename>" << std::endl;
  }
  std::string filename = argv[1];

  std::vector<unsigned int> primes = read_primes_from_file(filename);
std::vector<unsigned int> results(primes.size());

  unsigned int *d_primes;
  unsigned int *d_results;
  std::size_t num_primes = primes.size();

  std::cout << "Number of primes loaded: " << num_primes << std::endl;
  cudaMalloc(&d_primes, num_primes * sizeof(unsigned int));
  cudaMalloc(&d_results, num_primes * sizeof(unsigned int));

  cudaMemcpy(d_primes, primes.data(), num_primes * sizeof(unsigned int), cudaMemcpyHostToDevice);

  unsigned int block_size = 256;
  unsigned int num_blocks = (num_primes + block_size - 1) / block_size;
  auto start = std::chrono::high_resolution_clock::now();
  
  kershaw_prime_kernel<<<num_blocks, block_size>>>(d_primes, d_results, num_primes);
  cudaDeviceSynchronize();

  auto end = std::chrono::high_resolution_clock::now();

  cudaMemcpy(results.data(), d_results, num_primes * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  cudaFree(d_primes);
  cudaFree(d_results);

  for (std::size_t i = 0; i < num_primes; ++i) {
    if (results[i])
    std::cout << "Kershaw Prime: " << results[i] << std::endl;
  }

  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";

  return 0;
}
