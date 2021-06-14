#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <assert.h>
#include <Kokkos_Core.hpp>

using ExecSpace = Kokkos::DefaultExecutionSpace;

struct ParallelForTag {};

struct MyComplex {
    int64_t re, im;

    MyComplex() = default;

    KOKKOS_INLINE_FUNCTION
    MyComplex(int64_t re_, int64_t im_) : re(re_), im(im_) {}

    KOKKOS_INLINE_FUNCTION
    MyComplex(const MyComplex& src) : re(src.re), im(src.im) {}

    KOKKOS_INLINE_FUNCTION
    void operator=(MyComplex src) {
        re = src.re;
        im = src.im;
    }

    KOKKOS_INLINE_FUNCTION
    void operator+=(MyComplex& src) {
        re += src.re;
        im += src.im;
    }

    KOKKOS_INLINE_FUNCTION
    void operator+=(MyComplex const volatile& src) volatile {
        re += src.re;
        im += src.im;
    }
};

struct MyComplexArray {
    using ComplexArray = Kokkos::View<MyComplex*, ExecSpace>;
    ComplexArray view = ComplexArray("view", 0);
    int64_t N;

    MyComplexArray() = default;

    MyComplexArray(int64_t N_) : N(N_) { view = ComplexArray("view", N); }

    MyComplexArray(const MyComplexArray& src) : view(src.view) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const ParallelForTag, const int i) const {
        view(i) = MyComplex(i, -i);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, MyComplex& update) const { update += view(i); }
};

int main(int argc, char* argv[]) {
    Kokkos::ScopeGuard scope(argc, argv);

    printf("Execution Space = %s\n", typeid(ExecSpace).name());

    int64_t N = argc > 1 ? atoi(argv[1]) : 10;

    MyComplexArray input(N);
    MyComplex finalComplex(0, 0);

    Kokkos::parallel_for("parallel_for",
                         Kokkos::RangePolicy<ParallelForTag>(0, N), input);
    Kokkos::fence();
    Kokkos::parallel_reduce("parallel_reduce", N, input, finalComplex);

    //    assert(finalComplex.re == N * (N - 1) / 2);

    printf("finalComplex = (%ld,%ld) \n", finalComplex.re, finalComplex.im);
    printf("Done\n");
}
