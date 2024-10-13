#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

#include "matrix_mult.h"

pid_t fork_checked() {
    pid_t pid = fork();
    if (pid == FAILURE) {
        perror("Failure in fork");
        exit(EXIT_FAILURE);
    }
    return pid;
}

void * mmap_checked(size_t length) {
    void * result = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANON, 0, 0);
    if (result == MAP_FAILED) {
        perror("Failure in mmap");
        exit(EXIT_FAILURE);
    }
    return result;
}

void munmap_checked(void * addr, size_t length) {
    int result = munmap(addr, length);
    if (result == FAILURE) {
        perror("Failure in munmap");
        exit(EXIT_FAILURE);
    }
}

void init_matrix(double * const matrix, const int dim) {
    for (int i = 0; i < dim * dim; i++) {
        matrix[i] = i+1;
    }
}

void multiply_serial(
        const double * const a,
        const double * const b,
        double * const c,
        const int dim,
        const int num_workers
        ) {
    multiply_chunk(a, b, c, 0, dim, dim);
}

void multiply_chunk(const double * const a,
        const double * const b,
        double * const c,
        const int row_start,
        const int chunk,
        const int dim) {
    int end = row_start + chunk;
    for (int i = row_start; i < end; i++) {
        for (int j = 0; j < dim; j++) {
            c[i * dim + j] = 0;
            for (int k = 0; k < dim; k++) {
                c[i * dim + j] += a[i * dim + k] * b[k * dim + j];
            }
        }
    }
}

void multiply_parallel_processes(const double * const a,
                                 const double * const b,
                                 double * const c,
                                 const int dim,
                                 const int num_workers) {
    int num_procs = num_workers - 1;
    int chunk = dim / num_workers;
    double * result_matrix = mmap_checked(dim * dim * sizeof(*c));
    int i = 0;
    int row_start = 0;
    for (; i < num_procs; i++) {
        pid_t pid = fork_checked();
        if (pid == 0) {
            multiply_chunk(a, b, result_matrix, row_start, chunk, dim);
        exit(EXIT_SUCCESS);
        }
    row_start += chunk;
    }
    int last_chunk = num_procs * chunk;
    multiply_chunk(a, b, result_matrix, num_procs * chunk, dim - last_chunk, dim);
    while (wait(NULL) > 0);
    for (int i = 0; i < dim * dim; i++) {
        c[i] = result_matrix[i];
    }
    munmap_checked(result_matrix, dim * dim);
}

void multiply_parallel_threads(const double * a,
        const double * b,
        double * const c,
        const int dim,
        const int num_workers) {
    int num_threads = num_workers - 1;
    int chunk = dim / num_workers;
    Args args_set[num_workers];
    int row_start = 0;
    for (int id = 0; id < num_workers; id++) {
        args_set[id].a = a;
        args_set[id].b = b;
        args_set[id].c = c;
        args_set[id].row_start = row_start;
        args_set[id].chunk = chunk;
        args_set[id].dim = dim;
        row_start += chunk;
    }
    pthread_t threads[num_threads];
    for (int i = 0; i < num_threads; i++) {
        pthread_create(threads + i, NULL, task, args_set + i);
    }
    
    args_set[num_workers - 1].chunk = dim - args_set[num_workers - 1].row_start;
    task((void *)&args_set[num_workers - 1]); 
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}

void * task(void * arg) {
    Args * args = (Args *)arg;
    multiply_chunk(args->a, args->b, args->c, args->row_start, args->chunk, args->dim);
    return NULL;
}

void print_matrix(const double * const matrix, const int dim){
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            printf("%f ", matrix[i * dim + j]);
        }
        printf("\n");
    }
}

int verify(const double * const m1, const double * const m2, const int dim) {
    for (int i = 0; i < dim * dim; i++) {
        if ((int)m1[i] != (int)m2[i]) {
            return FAILURE;
        }
    }
    return SUCCESS;
}

void print_verification(const double * const m1,
        const double * const m2,
        const int dim,
        const char * const name) {
    int status = verify(m1, m2, dim);
    if (status == SUCCESS) {
        printf("Verification for %s: success.\n", name);
    } else {
        printf("Verification for %s: failure.\n", name);
    }

}

struct timeval time_diff(struct timeval * start, struct timeval * end) {
    struct timeval diff;
    diff.tv_sec = end->tv_sec - start->tv_sec;
    diff.tv_usec = end->tv_usec - start->tv_usec;
    if (diff.tv_usec < 0) {
        diff.tv_usec += 1000000;
        diff.tv_sec -= 1;
    }
    return diff;
}

void print_elapsed_time(struct timeval * start,
        struct timeval * end,
        const char * const name) {
    struct timeval diff = time_diff(start, end);
    printf("Time elapsed for %s: %ld seconds and %d microseconds.\n", name, diff.tv_sec, diff.tv_usec);
}

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
        ) {
    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);
    multiply_matrices(a, b, c, dim, num_workers);
    printf("Algorithm: %s with %d worker%s.\n", name, num_workers, (num_workers != 1) ? "s" : "");
    gettimeofday(&end, NULL);
    print_elapsed_time(&start, &end, name);
    if (strcmp(name, "serial") != 0) {
        print_verification(c, gold, dim, name);
    }
}


