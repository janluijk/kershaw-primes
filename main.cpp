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

unsigned int compute_order(unsigned int p) {
  unsigned int val = 2, order = 1;

  while (val != 1) {
    val = (val << 1) % p;
    order++;
  }

  return order;
}

unsigned int compute_base(unsigned int order, unsigned int p) {
  unsigned long long val = 3;
  unsigned int count = 1;
  const long long max_val = ULLONG_MAX / 3 - 2;

  while (count < order) {
    ++count;

    if (max_val < val)
      val = val % p;

    val = val * 3;
  }

  return val % p;
}

bool compute_mod(unsigned int order, unsigned int base, unsigned int p) {
  unsigned int val = base, count = 1;
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
      if (result == 2)
        return true;
    }
  }

  return false;
}

void kershaw_prime(unsigned int p) {

  unsigned int order = compute_order(p);
  unsigned int base = compute_base(order, p);
  bool found = compute_mod(order, base, p);
  if (found) {
    std::cout << "Found kershaw prime: " << p << std::endl;
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: <filename>" << std::endl;
  }
  std::string filename = argv[1];

  std::vector<unsigned int> primes = read_primes_from_file(filename);

  std::cout << "Number of primes loaded: " << primes.size() << std::endl;

  unsigned int num_threads = std::thread::hardware_concurrency();
  std::size_t chunk_size = primes.size() / num_threads;
  std::vector<std::vector<unsigned int>> chunks(num_threads);
  for (unsigned int i = 0; i < num_threads; i++) {
    auto start = primes.begin() + i * chunk_size;
    auto end = (i == num_threads - 1) ? primes.end() : start + chunk_size;
    chunks[i].assign(start, end);
  }
  auto start = std::chrono::high_resolution_clock::now();

  std::vector<std::thread> threads;
  for (unsigned int i = 0; i < num_threads; i++) {
    threads.emplace_back([&, i]() {
      for (unsigned int prime : chunks[i]) {
        kershaw_prime(prime);
      }
      std::cout << "thread done" << std::endl;
    });
  }

  for (std::thread &t : threads) {
    t.join();
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";

  return 0;
}
