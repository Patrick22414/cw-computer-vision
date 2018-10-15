import concurrent.futures
import math
import time

PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1125899839733759,
    128777389867557,
    106728982367279]


def is_prime(n):
    if n % 2 == 0:
        return False

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True


def main():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for number, ans in zip(PRIMES, executor.map(is_prime, PRIMES)):
            print('%d is prime: %s' % (number, ans))


def main2():
    for number, ans in zip(PRIMES, map(is_prime, PRIMES)):
        print('%d is prime: %s' % (number, ans))


if __name__ == '__main__':
    start = time.time()
    main()
    print(time.time() - start)

    start = time.time()
    main2()
    print(time.time() - start)
