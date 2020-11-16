#include <iostream>
#include <omp.h>
#define N 100
#define nT 64
int main(int argc, char **argv) {
#if USE_SINGLE
  std::cout << "Using \"omp single \" to initialize scratch " << "\n";
#else
  std::cout << "Using \"omp barrier \" to initialize scratch " << "\n";
#endif

  int scalar = 0;
  int vector[N];

  size_t scratch_size = N * nT * sizeof(int);
  int *scratch = static_cast<int *>(
      omp_target_alloc(scratch_size, omp_get_default_device()));

  auto lambda1 = [=](const int i, int *scratch, int& scalar) {
    int *my_scratch = scratch + (omp_get_team_num() * nT);
#pragma omp barrier
#if USE_SINGLE
#pragma omp single
    my_scratch[0] = 0;
#else
    my_scratch[0] = 0;
#pragma omp barrier
#endif

#pragma omp for reduction(+ : my_scratch [0:1])
    for (int j = 0; j < N; ++j) {
      my_scratch[0] += i + 1;
    } // end j

    scalar += my_scratch[0];
  };

  auto lambda2 = [=](const int i, int *scratch) {
    int *my_scratch = scratch + (omp_get_team_num() * nT);
#pragma omp barrier
#if USE_SINGLE
#pragma omp single
    my_scratch[0] = 0;
#else
    my_scratch[0] = 0;
#pragma omp barrier
#endif

#pragma omp for reduction(+ : my_scratch [0:1])
    for (int j = 0; j < N; ++j) {
      my_scratch[0] += i + 1;
    } // end j
  };

  // Case 1 - reduction is over teams and parallel-for level.
#pragma omp target teams distribute reduction(+ : scalar) thread_limit(nT) \
  is_device_ptr(scratch)
  for (int i = 0; i < N; ++i) {
#pragma omp parallel num_threads(nT)
    {
      lambda1(i, scratch, scalar);
    } // end parallel
  }   // end teams

  // Case 2 - reduction is over parallel-for level.
#pragma omp target teams distribute thread_limit(nT) \
  map(tofrom: vector[0:N]) \
  is_device_ptr(scratch)
  for (int i = 0; i < N; ++i) {
#pragma omp parallel num_threads(nT)
    {
      int *my_scratch = scratch + (omp_get_team_num() * nT);
      lambda2(i, scratch);
      vector[i] = my_scratch[0];
    } // end parallel
  }   // end teams

  size_t result = 0;
  for(int i = 0; i < N; ++i)
    result += vector[i];

  if(result != scalar)
  {
    printf("Failure : scalar!=vector, scalar = %d, vector = %zu\n",2*scalar, result);
    exit(EXIT_FAILURE);
  }

  if (scalar == ((N * N * (N + 1)) / 2)) {
    std::cout << "success" << std::endl;
  } else {
    std::cout << "failure" << std::endl;
  }
  std::cout << "expected = " << ((N * N * (N + 1)) / 2) << std::endl;
  std::cout << "scalar   = " << scalar << std::endl;
}
