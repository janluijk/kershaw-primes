#include <chrono>
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <thread>
#include <mutex>

std::vector<unsigned int> read_primes_from_file(const std::string &filename);
std::vector<unsigned int> compute_primes(unsigned int from, unsigned int to);
unsigned int compute_order(unsigned int p);
bool compute_mod(unsigned int p, unsigned int order, unsigned int base);
unsigned int mod_exp(unsigned long long base, unsigned int exp, unsigned int p);

int main() {
  for(unsigned int i = 2; i < 4e9; i += 2e7) {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<unsigned int> primes = compute_primes(i, i + 2e7);
    std::size_t num_primes = primes.size();

    unsigned int num_threads = std::thread::hardware_concurrency();

    std::size_t chunk_size = num_primes / num_threads;
    std::vector<std::vector<unsigned int>> chunks(num_threads);

    for (unsigned int i = 0; i < num_threads; i++) {
      auto start = primes.begin() + i * chunk_size;
      auto end = (i == num_threads - 1) ? primes.end() : start + chunk_size;

      chunks[i].assign(start, end);
    }

    std::vector<std::thread> threads;
    for (unsigned int i = 0; i < num_threads; i++) {
      threads.emplace_back([&, i]() {
        for (unsigned int prime : chunks[i]) {

          // Use Fermat's little theorem to find ord_p(2)
          unsigned int order  = compute_order(prime);

          // Compute 3^ord % p
          unsigned int base   = mod_exp(3, order, prime);

          // compute 3^(nm-1) % p
          bool found          = compute_mod(prime, order, base);
          if (found) 
            std::cout << "Found a prime: " << prime << " Order: " << order << " Base: " << base << std::endl; 
        }
      });
    }

    for (std::thread &t : threads) {
      t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "CPU: Elapsed time: " << elapsed_seconds.count() << "s\n";
  }
  
  return 0;
}

std::vector<unsigned int> compute_primes(unsigned int from, unsigned int to) {
  std::vector<bool> isPrime(to - from + 1, true);
  std::vector<unsigned int> primes;

  for(unsigned int p = 2; p * p <= to; ++p) {
    for (unsigned int i = std::max(p * p, (from + p - 1) / p * p); i <= to; i += p) {
        isPrime[i - from] = false;
    }
  }

  for (int i = from; i <= to; ++i) {
    if (isPrime[i - from]) {
      primes.push_back(i);
    }
  }

  return primes;  
}

unsigned int compute_order(unsigned int p) {
  const unsigned int n = p - 1;
  const unsigned int lim = sqrt(n);

  std::vector<unsigned int> divisors;
  for (unsigned int i = 1; i <= lim; ++i) {
    if (n % i == 0) { // Found divisor
      if (mod_exp(2, i, p) == 1)  // Calculate 2^n % p 
        return i; 
      
      unsigned int inverse_divisor = n / i; // Really a terrible name
      if(i != inverse_divisor)  
        divisors.push_back(inverse_divisor);
    }
  }

  for (auto div = divisors.rbegin(); div != divisors.rend(); ++div) {
    if (mod_exp(2, *div, p) == 1) 
      return *div; 
  }

  return -1;
}

bool compute_mod(unsigned int p, unsigned int order, unsigned int base) {
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

unsigned int mod_exp(unsigned long long base, unsigned int exp, unsigned int p) {
  unsigned long long result = 1;

  while (exp) {
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