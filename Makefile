kershaw_primes: main.cpp
	g++ -O3 -o kershawprimes main.cpp

.PHONY: test clean

test: kershaw_primes
	./kershawprimes primestest.txt

clean:
	rm -f kershawprimes

