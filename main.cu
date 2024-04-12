#include <chrono>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>

// CPU
std::vector<unsigned int> read_primes_from_file(const std::string &filename);
std::vector<unsigned int> find_divisors(unsigned int n);
unsigned int mod_exp(long long base, unsigned int exp, long long n);

// GPU
__global__ void kershaw_prime_kernel(unsigned int* primes, unsigned int* orders, unsigned int* results, std::size_t size);
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
  std::vector<unsigned int> results;
  std::vector<unsigned int> orders;
  // std::vector<unsigned int> timing_data;
  std::size_t num_primes = primes.size();
  results.reserve(num_primes * sizeof(unsigned int));
  orders.reserve(num_primes * sizeof(unsigned int));
  // timing_data.reserve(num_primes * 4 * sizeof(unsiged int));

  // Find orders 
  for(unsigned int prime: primes) {
    std::vector<unsigned int> divisors = find_divisors(prime - 1);
    for(unsigned int divisor : divisors) {
      if (mod_exp(2, divisor, prime) == 1) {
        orders.push_back(divisor);
        break;      
      }
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "CPU: Elapsed time: " << elapsed_seconds.count() << "s\n";
  
  // GPU
  start = std::chrono::high_resolution_clock::now();

  unsigned int *d_primes;
  unsigned int *d_orders;
  unsigned int *d_results;
  // unsigned int *d_timing_data;

  cudaMalloc(&d_primes, num_primes * sizeof(unsigned int));
  cudaMalloc(&d_orders, num_primes * sizeof(unsigned int));
  cudaMalloc(&d_results, num_primes * sizeof(unsigned int));
  // cudaMalloc(&d_timing_data, 4 * num_primes * sizeof(unsigned int));

  cudaMemcpy(d_primes, primes.data(), num_primes * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_orders, orders.data(), num_primes * sizeof(unsigned int), cudaMemcpyHostToDevice);

  unsigned int block_size = 256;
  unsigned int num_blocks = (num_primes + block_size - 1) / block_size;
  start = std::chrono::high_resolution_clock::now();
  
  kershaw_prime_kernel<<<num_blocks, block_size>>>(d_primes, d_orders, d_results, num_primes);

  cudaDeviceSynchronize();

  cudaMemcpy(results.data(), d_results, num_primes * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  // cudaMemcpy(timing_data.data(), d_timing_data, 4 * num_primes * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  cudaFree(d_primes);
  cudaFree(d_orders);
  cudaFree(d_results);
  // cudaFree(d_timing_data);

  for (std::size_t i = 0; i < num_primes; ++i) {
    if (results[i])
      std::cout << "Kershaw Prime: " << results[i] << std::endl;
  }

  end = std::chrono::high_resolution_clock::now();

  // for (std::size_t i = 0; i < num_primes; ++i) {
  //   std::cout << 
  //   timing_data[i * 4]
  //   << ' ' << 
  //   timing_data[i * 4 + 1]
  //   << ' ' << 
  //   timing_data[i * 4 + 2]
  //   << ' ' << 
  //   timing_data[i * 4 + 3]
  //   << std::endl;
  // }


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

std::vector<unsigned int> find_divisors(unsigned int n) {
  std::vector<unsigned int> divisors;
  unsigned int sqrtN = sqrt(n);
  for (unsigned int i = 1; i <= sqrtN; ++i) {
    if (n % i == 0) {
      divisors.push_back(i);
      unsigned int inverse = n / i;
      if (i != inverse) {
        divisors.push_back(inverse);
      }
    }
  }
  std::sort(divisors.begin(), divisors.end());
  return divisors;
}

unsigned int mod_exp(long long base, unsigned int exp, long long p) {
  long long result = 1;

  while (exp > 0) {
    if (exp & 1) {
      result = (result * base);
      if (result > p)
        result = result % p;
    }
    base = (base * base);
    if (base > p)
      base = base % p;
    exp >>= 1;
  }

  return result;
}

// GPU

// __device__ unsigned int compute_order(unsigned int p) {
//   unsigned int val = 2, order = 1;
//   while (val != 1) {
//     val = val * 2;
//     if (val > p) {
//       val = val % p;
//     }
//     order++;
//   }

//   return order;
// }

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

__global__ void kershaw_prime_kernel(unsigned int* primes, unsigned int* orders, unsigned int* results, std::size_t size) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    // unsigned int start_order = clock();
    // unsigned int end_order = clock();
    // unsigned int start_base = clock();
    // unsigned int end_base = clock();
    // unsigned int start_mod = clock();
    // unsigned int end_mod = clock();
    // timing_data[idx * 4] = end_order - start_order;
    // timing_data[idx * 4 + 1] = end_base - start_base;
    // timing_data[idx * 4 + 2] = end_mod - start_mod;
    // timing_data[idx * 4 + 3] = (end_mod - start_order);

    unsigned int p = primes[idx];
    unsigned int order  = orders[idx];
    unsigned int base   = compute_base(p, order);
    bool found          = compute_mod(p, order, base);

    if (found) {
      printf("Prime found! Prime: %d, Order: %d, Base: %d\n", p, order, base);
      results[idx] = p;
    }
  }
}