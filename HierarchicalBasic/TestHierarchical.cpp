#include <assert.h>
#include <iostream>
#include <omp.h>

#define nG 1
#define mT 64 // maximum number of threads

int main(int argc, char **argv) {
  int v[nG * mT] = {0};
  int nT; // actual number of threads executed.

  auto lambda = [=](const int i, int *v) {
    int nT = omp_get_num_threads();
    for (int j = 0; j < nT; j++)
      v[i * nT + j] = i * nT + j;
  };

#pragma omp target teams distribute num_teams(nG) thread_limit(mT) map(from    \
                                                                       : nT)   \
    map(tofrom                                                                 \
        : v [0:nG * mT])
  for (int i = 0; i < nG; ++i) {
#pragma omp parallel num_threads(mT)
    { lambda(i, v); }
  }
  size_t check = 0;
  size_t ref = nG * mT;
  for (int i = 0; i < nG; ++i)
    for (int j = 0; j < mT; ++j)
      check += v[i * mT + j];

  if (check == ref * (ref - 1) / 2)
    std::cout << "Success! "
              << "\n";
  else {
    std::cout << "Error expected = " << ref * (ref - 1) / 2
              << " got = " << check << "\n";
  }

  return 0;
}
