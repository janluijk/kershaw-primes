kershaw_primes: main.cpp
	g++ -O3 kershawprimes main.cpp

.PHONY: test clean

test: kershaw_primes
	./kershaw_primes primes12.txt

clean:
	rm -f kershaw_primes

