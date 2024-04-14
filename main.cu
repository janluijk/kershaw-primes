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
__global__ void kershaw_prime_kernel(uint32_t* primes, std::size_t size);
__device__ uint32_t mod_exp(unsigned long long base, uint32_t exp, uint32_t p);
__device__ uint32_t compute_order(uint32_t p);
__device__ uint32_t compute_base(uint32_t p, uint32_t order);
__device__ bool compute_mod(uint32_t p, uint32_t order, uint32_t base);

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

  uint32_t *d_primes;
  cudaError_t cuda_error;

  cuda_error = cudaMalloc(&d_primes, num_primes * sizeof(uint32_t));
  if(cuda_error != cudaSuccess) {
      std::cerr << "CUDA error (cudaMalloc): " << cudaGetErrorString(cuda_error) << std::endl;
  }

  cuda_error = cudaMemcpy(d_primes, primes.data(), num_primes * sizeof(uint32_t), cudaMemcpyHostToDevice);
  if(cuda_error != cudaSuccess) {
      std::cerr << "CUDA error (cudaMemcpy HostToDevice): " << cudaGetErrorString(cuda_error) << std::endl;
  }

  uint32_t block_size = 256;
  uint32_t num_blocks = (num_primes + block_size - 1) / block_size;

  kershaw_prime_kernel<<<num_blocks, block_size>>>(d_primes, num_primes);
  cuda_error = cudaGetLastError();
  if(cuda_error != cudaSuccess) {
      std::cerr << "CUDA error (kernel launch): " << cudaGetErrorString(cuda_error) << std::endl;
  }

  cudaDeviceSynchronize();

  cuda_error = cudaGetLastError(); // Check for any errors during kernel execution
  if(cuda_error != cudaSuccess) {
      std::cerr << "CUDA error (kernel execution): " << cudaGetErrorString(cuda_error) << std::endl;
  }

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
__global__ void kershaw_prime_kernel(uint32_t *primes, std::size_t size) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    uint32_t p      = primes[idx];
    uint32_t order  = compute_order(p);
    uint32_t base   = compute_base(p, order);
    bool found      = compute_mod(p, order, base);


    if (found)
      printf("Prime found! Prime: %d, Order: %d, Base: %d\n", p, order, base);
  }
}

__device__ uint32_t compute_order(const uint32_t p) {
  uint32_t n = p - 1;
  const uint32_t lim = (uint32_t) sqrt((double)n);
  uint32_t divisors[150000];
  int num_divisors = 0;

  for (uint32_t i = 1; i < lim; ++i) {
    if (n % i == 0) { // Found divisor
      if (mod_exp(2, i, p) == 1) {
        return i;
      }
      
      uint32_t inverse_divisor = n / i;
      if (i != inverse_divisor)
        divisors[num_divisors] = inverse_divisor;
        num_divisors++;
    }
  }

  for (int j = num_divisors - 1; j >= 0; --j) {
    if (mod_exp(2, divisors[j], p) == 1) {
      unsigned int result = divisors[j]; 
      return result;
    }
  }

  return p - 1; // Shut up compiler
}

__device__ uint32_t mod_exp(unsigned long long base, uint32_t exp, uint32_t p) {
  unsigned long long result = 1;

  while (exp) {
    if (exp & 1) {
      result = (result * base) % p;
      if (result > p) 
        result = result % p;
    }
    
    base = base * base;
    if (base > p) 
      base = base % p;
    exp >>= 1;
  }
  return result;
}

__device__ uint32_t compute_base(uint32_t p, uint32_t order) {
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

__device__ bool compute_mod(uint32_t p, uint32_t order, uint32_t base) {
  unsigned long long val = base;
  uint32_t count = 1;
  const uint32_t limit = (p - 1) / order;

  while (count < limit) {
    val = val * base;
    count++;

    if (val > p)
      val = val % p;

    uint32_t result = 3 * val;
    if (result > p) {
      result = result % p;
      if (result == 2)
        return true;
    }
  }
  return false;
}
