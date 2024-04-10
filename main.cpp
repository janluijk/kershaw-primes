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

void kershaw_prime(unsigned int prime) {
  unsigned int p = prime;
  unsigned int val = 2, order = 1;

  while (val != 1) {
    val = (val << 1) % p;
    
    order++;
  }

  std::cout << order << std::endl;

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

  std::cout << base << std::endl;
  val = base;
  count = 1;
  unsigned int limit = (p - 1) / order;

  while (count < limit) {
    val = val * base;
    count++;

    val = val % p;

    unsigned int result = 3 * val % p;
    if (result == 2) {
      std::cout << "kershaw prime: " << p << std::endl;
      break;
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

  auto start = std::chrono::high_resolution_clock::now();

  for(unsigned int p: primes) {
    kershaw_prime(p);
  }

  auto end = std::chrono::high_resolution_clock::now();


  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";

  return 0;
}
