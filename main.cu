#include <chrono>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <thread>
#include <mutex>

// CPU
std::vector<unsigned int> read_primes_from_file(const std::string &filename);

// GPU
__global__ void kershaw_prime_kernel(unsigned int* primes, std::size_t size);
__device unsigned int mod_exp(unsigned int base, unsigned int exp, unsigned int p);
__device__ unsigned int compute_order(unsigned int p);
__device__ unsigned int compute_base(unsigned int p, unsigned int order);
__device__ bool compute_mod(unsigned int p, unsigned int order, unsigned int base);

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "usage: <filename>" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::string filename = argv[1];

  // CPU
  auto start = std::chrono::high_resolution_clock::now();

  std::vector<unsigned int> primes = read_primes_from_file(filename);
  std::size_t num_primes = primes.size();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "CPU: Elapsed time: " << elapsed_seconds.count() << "s\n";
  
  // GPU
  start = std::chrono::high_resolution_clock::now();

  unsigned int *d_primes;

  cudaMalloc(&d_primes, num_primes * sizeof(unsigned int));
  cudaMemcpy(d_primes, primes.data(), num_primes * sizeof(unsigned int), cudaMemcpyHostToDevice);

  unsigned int block_size = 256;
  unsigned int num_blocks = (num_primes + block_size - 1) / block_size;
  start = std::chrono::high_resolution_clock::now();
  
  kershaw_prime_kernel<<<num_blocks, block_size>>>(d_primes, num_primes);

  cudaDeviceSynchronize();

  cudaFree(d_primes);

  end = std::chrono::high_resolution_clock::now();
  elapsed_seconds = end - start;
  std::cout << "GPU: Elapsed time: " << elapsed_seconds.count() << "s\n";

  return 0;
}

// CPU
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

  std::cout << "Number of primes loaded: " << primes.size() << std::endl;
  return primes;
}

// GPU
__device__ unsigned int compute_order(unsigned int p) {
  const unsigned int n = p - 1;
  const unsigned int lim = sqrt(n);
  unsigned int divisors[lim + 1];
  int num_divisors = 0;

  for (unsigned int o = 1; i <= lim; ++i) {
    if (n % i == 0) { // Found divisor
      if (mod_exp(2, i, p) == 1)
        return i;
      
      unsigned int inverse_divisor = n / i;
      if (i != inverse_divisor)
        divisors[num_divisors++] = inverse_divisor
    }
  }

  for (int j = num_divisors - 1; j >= 0; --j) {
    if (mod_exp(2, divisors[j], p) == 1)
      return divisors[j];
  }

  return -1;
}

__device unsigned int mod_exp(unsigned int base, unsigned int exp, unsigned int p) [
  unsigned int result = 1;
  base = base % p;
  while (exp) {
    if (exp & 1)
      result = (result * base) % p;
    
    exp >>= 1;
    base = (base * base) % p;
  }
}

__device__ unsigned int compute_base(unsigned int p, unsigned int order) {
  unsigned long long result = 1;
  unsigned long long base = 3;
  while (order > 0) {
    if (order & 1)
      result = (result * base) % p;

    base = (base * base) % p;
    order >>= 1;
  }

  return result;
}

__device__ bool compute_mod(unsigned int p, unsigned int order, unsigned int base) {
  unsigned long long val = base;
  unsigned int count = 1;
  const unsigned int limit = (p - 1) / order;

  while (count < limit) {
    val = val * base;
    count++;

    if (val > p)
      val = val % p;

    unsigned int result = 3 * val;
    if (result > p) {
      result = result % p;
      if (result == 2)
        return true;
    }
  }
  return false;
}

__global__ void kershaw_prime_kernel(unsigned int* primes, unsigned int* orders, std::size_t size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    unsigned int p      = primes[idx];
    unsigned int order  = orders[idx];
    unsigned int base   = compute_base(p, order);
    bool found          = compute_mod(p, order, base);

    if (found)
      printf("Prime found! Prime: %d, Order: %d, Base: %d\n", p, order, base);
  }
}