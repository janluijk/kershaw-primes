#include <chrono>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <thread>
#include <mutex>

// CPU
std::vector<unsigned int> read_primes_from_file(const std::string &filename);
std::vector<unsigned int> find_divisors(unsigned int n);
unsigned int mod_exp(long long base, unsigned int exp, long long n);

// GPU
__global__ void kershaw_prime_kernel(unsigned int* primes, unsigned int* orders, std::size_t size);
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
  std::vector<std::pair<unsigned int, unsigned int>> prime_order; // Store results with prime number
  std::size_t num_primes = primes.size();

  prime_order.reserve(num_primes * sizeof(std::pair<unsigned int, unsigned int>));

  unsigned int num_threads = std::thread::hardware_concurrency();
  std::size_t chunk_size = primes.size() / num_threads;
  std::vector<std::vector<unsigned int>> chunks(num_threads);
  for (unsigned int i = 0; i < num_threads; i++) {
    auto start = primes.begin() + i * chunk_size;
    auto end = (i == num_threads - 1) ? primes.end() : start + chunk_size;
    chunks[i].assign(start, end);
  }
  std::mutex prime_order_mutex;
  std::vector<std::thread> threads;
  for (unsigned int i = 0; i < num_threads; i++) {
    threads.emplace_back([&, i]() {
      for (unsigned int prime : chunks[i]) {
        std::vector<unsigned int> divisors = find_divisors(prime - 1);
        unsigned int order;
        for(unsigned int divisor : divisors) {
          if (mod_exp(2, divisor, prime) == 1) {
            order = divisor;
            break;      
          }
        }
        prime_order_mutex.lock();
        prime_order.emplace_back(prime, order);
        prime_order_mutex.unlock();
      }
    });
  }

  for (std::thread &t : threads) {
    t.join();
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "CPU: Elapsed time: " << elapsed_seconds.count() << "s\n";
  
  // GPU
  start = std::chrono::high_resolution_clock::now();

  unsigned int *d_primes;
  unsigned int *d_orders;

  cudaMalloc(&d_primes, num_primes * sizeof(unsigned int));
  cudaMalloc(&d_orders, num_primes * sizeof(unsigned int));

  cudaMemcpy(d_primes, &prime_order[0].first, num_primes * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_orders, &prime_order[0].second, num_primes * sizeof(unsigned int), cudaMemcpyHostToDevice);

  unsigned int block_size = 256;
  unsigned int num_blocks = (num_primes + block_size - 1) / block_size;
  start = std::chrono::high_resolution_clock::now();
  
  kershaw_prime_kernel<<<num_blocks, block_size>>>(d_primes, d_orders, num_primes);

  cudaDeviceSynchronize();

  cudaFree(d_primes);
  cudaFree(d_orders);

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