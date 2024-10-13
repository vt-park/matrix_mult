#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#define DIM 1024
#define NUM_WORKERS 4
#define SUCCESS 0
#define FAILURE -1

typedef void (*multiply_function)(
        const double * const a,
        const double * const b,
        double * const c,
        const int dim,
        const int num_workers
);

typedef struct Args {
    const double * a;
    const double * b;
    double * c;
    int row_start;
    int chunk;
    int dim;
} Args;
pid_t fork_checked();

void * mmap_checked(size_t length);

void munmap_checked(void * addr, size_t length);

void init_matrix(double * const matrix, const int dim);

void multiply_serial(
        const double * const a,
        const double * const b,
        double * const c,
        const int dim,
        const int num_workers
        );

void multiply_parallel_processes(const double * const a,
        const double * const b,
        double * const c,
        const int dim,
        const int num_workers);

void multiply_parallel_threads(const double * const a,
        const double * const b,
        double * const c,
        const int dim,
        const int num_workers);

void * task(void * arg);

void multiply_chunk(const double * const a,
        const double * const b,
        double * const c,
        const int row_start,
        const int chunk,
        const int dim);

void print_matrix(const double * const matrix, const int dim);

int verify(const double * const m1, const double * const m2, const int dim);

void print_verification(const double * const m1,
        const double * const m2,
        const int dim,
        const char * const name);

struct timeval time_diff(struct timeval * start, struct timeval * end);

void print_elapsed_time(struct timeval * start,
        struct timeval * end,
        const char * const name);

void run_and_time(
        multiply_function multiply_matrices,
        const double * const a,
        const double * const b,
        double * const c,
        const double * const gold,
        const int dim,
        const char * const name,
        const int num_workers,
        const bool verify
        );
