/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 * (C) 2015 by Argonne National Laboratory.
 *     See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <mpi.h>
#include <cblas.h>
#include <assert.h>

/* This benchmark measures adaptation for the execution contains multiple
 * communication-intensive phases and computation-intensive phases with 3D
 * non-contigunous double data.
 * -During the execution of each window, multiple phases exist;
 * -During each phase, multiple collective calls (win_set_info) exist;
 * -Between two collective calls, every process performs all-to-all
 *  GET-flush-compute-ACC-flush-barrier, with fixed computing time and increasing
 *  number of operations. The computation intensive phase runs with larger dgemm
 *  and a small number of operations; the communication intensive phase runs
 *  with small dgemm and increasing number of operations.
 *
 * It generates following modes :
 * - no adaptation (static on of off)
 * - user guided adaptation, set hint in every win_allocate
 * - user guided adaptation, set hint in every win_set_info
 * - self-profiling based adaptation
 * - self-profiling based adaptation with ghost-synchronization.
 * */


// #define DLEN 4
// #define SUB_DLEN 2

int DLEN = 4;
int SUB_DLEN = 2;
int COMPT = 1;
/* 3D matrix on target window */
#define WINSIZE (DLEN*DLEN*DLEN)
/* Local 3D submatrix */
#define BUFSIZE (SUB_DLEN*SUB_DLEN*SUB_DLEN)

double *winbuf = NULL;
double *locbuf = NULL;
int rank = 0, nprocs = 0;
int shm_rank = 0, shm_nprocs = 0;
MPI_Win win = MPI_WIN_NULL;

int WOKERS;
int PHASE_ITER = 10, COLL_ITER = 50, SKIP = 10, NWINS = 2;
int NOP_L_MAX = 16, NOP_L_MIN = 16, NOP_L_ITER = 2, NOP_S = 1, NOP_L = 1;
int MS = 1, ML = 100;           /* total problem size */

MPI_Datatype target_type = MPI_DATATYPE_NULL;
int target_size = 0;
MPI_Aint target_ext = 0;

double *A, *B, *C;
int m, n, k;                    /* local size */
double t_comp = 0.0;

static void target_computation_init(void)
{
    int ml, err;

    /* allocate max size with alignment on 64-byte boundary for better performance */
    ml = ML / nprocs;

    err = posix_memalign((void **)&A, 64, ml * ml * sizeof(double));
    assert(err == 0);
    err = posix_memalign((void **)&B, 64, ml * ml * sizeof(double));
    assert(err == 0);
    err = posix_memalign((void **)&C, 64, ml * ml * sizeof(double));
    assert(err == 0);

    int i;
    for (i = 0; i < (ml * ml); i++)
            A[i] = (double) (i + 1);

    for (i = 0; i < (ml * ml); i++)
        B[i] = (double) (-i - 1);

    for (i = 0; i < (ml * ml); i++)
        C[i] = 0.0;
}

static void target_computation_set_size(int M, int K, int N)
{
    m = M / nprocs;
    k = K / nprocs;
    n = N / nprocs;
}

static void target_computation(void)
{
    int i, j;
    double alpha = 1.0, beta = 0.0;
    double t0;

    t0 = MPI_Wtime();

    if (m > 0 && k > 0 && n > 0) {
        /* reset data */
        

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k, alpha, A, k, B, n, beta, C, n);
    }

    t_comp += MPI_Wtime() - t0;
}

static void target_computation_destroy(void)
{
    free(A);
    free(B);
    free(C);
    A = NULL;
    B = NULL;
    C = NULL;
}

static int run_iteration()
{
    int i, wx, cx, px, x, errs = 0;
    int dst;
    int nop = NOP_S;
    MPI_Info win_info = MPI_INFO_NULL;
    MPI_Info async_info = MPI_INFO_NULL;
    double t0, t_total = 0.0;
    double st0 = 0.0, t_comm_phase = 0.0, t_comp_phase = 0.0, t_comm_comp = 0.0, t_comp_comp = 0.0;
    /* min, max, avg */
    double sum_total_times[3], sum_t_comm_comps[3], sum_t_comp_comps[3], sum_t_comp_phases[3],
        sum_t_comm_phases[3];
    int comp_pcnt = 0, comm_pcnt = 0;

#if defined(OUTPUT_ALL_PHASES)
    int phase_idx = 0;
    int nphases = NWINS * PHASE_ITER;
    double *phase_times = NULL;
    phase_times = malloc(sizeof(double) * nphases);
    memset(phase_times, 0, sizeof(phase_times));
#endif

    MPI_Info_create(&win_info);
    MPI_Info_create(&async_info);

    MPI_Info_set(win_info, (char *) "epoch_type", (char *) "lockall|fence");
    MPI_Info_set(async_info, (char *) "symmetric", (char *) "true");

#if defined(DISABLE_SHM)
    /* disable shm rma for original case because too heavy per-op lock contention */
    MPI_Info_set(win_info, (char *) "alloc_shm", (char *) "false");
#endif

    /* skip */
    MPI_Win_allocate(WINSIZE * sizeof(double), sizeof(double), win_info,
                     MPI_COMM_WORLD, &winbuf, &win);
    for (x = 0; x < SKIP; x++) {
        /* reset window */
        MPI_Win_fence(MPI_MODE_NOPRECEDE, win);
        memset(winbuf, 0, sizeof(double) * WINSIZE);
        MPI_Win_fence(0, win);

        MPI_Win_lock_all(0, win);
        for (dst = 0; dst < nprocs; dst++) {
            MPI_Get(locbuf, BUFSIZE, MPI_DOUBLE, dst, 0, 1, target_type, win);
            MPI_Win_flush(dst, win);
        }
        MPI_Win_unlock_all(win);
    }
    MPI_Win_free(&win);

    /* start */
    t0 = MPI_Wtime();
    for (wx = 0; wx < NWINS; wx++) {
#if defined(ENABLE_CSP_ADPT_U)
        /* first phase is comp heavy */
        MPI_Info_set(win_info, (char *) "async_config", (char *) "on");
        MPI_Info_set(async_info, (char *) "async_config", (char *) "on");
#endif
        MPI_Win_allocate(WINSIZE * sizeof(double), sizeof(double), win_info,
                         MPI_COMM_WORLD, &winbuf, &win);

        /* reset window */
        MPI_Win_fence(MPI_MODE_NOPRECEDE, win);
        memset(winbuf, 0, sizeof(double) * WINSIZE);
        MPI_Win_fence(0, win);
        MPI_Win_lock_all(0, win);

        for (px = 0; px < PHASE_ITER; px += 1) {
            st0 = MPI_Wtime();

            if (px < PHASE_ITER / 2) {  /* heavy comp */
                target_computation_set_size(ML, ML, ML);
                nop = NOP_S;
                comp_pcnt++;
            }
            else {      /* heavy comm */
                target_computation_set_size(MS, MS, MS);
                nop = NOP_L;
                comm_pcnt++;
            }

            for (x = 0; x < COLL_ITER; x += 1) {
                for (dst = 0; dst < WOKERS; dst++) {
                    int real_dst;
                    if(dst >= nprocs)
                        real_dst = rand() % nprocs;
                    else
                        real_dst = dst;
                    for (i = 0; i < nop; i++) {
                        MPI_Get(locbuf, BUFSIZE, MPI_DOUBLE, real_dst, 0, 1, target_type, win);
                        MPI_Win_flush(dst, win);
                    }
                }

                for(i = 0; i< COMPT; ++i)
                    target_computation();

                for (dst = 0; dst < nprocs; dst++) {
                    int real_dst;
                    if(dst >= nprocs)
                        real_dst = rand() % nprocs;
                    else
                        real_dst = dst;
                    for (i = 0; i < nop; i++) {
                        MPI_Accumulate(locbuf, BUFSIZE, MPI_DOUBLE, real_dst, 0, 1, target_type, MPI_SUM,
                                       win);
                        MPI_Win_flush(dst, win);
                    }
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);

#if defined(ENABLE_CSP_ADPT_U)
            if (px == PHASE_ITER / 2 - 1) {     /* next is heavy comm, update info */
                MPI_Info_set(async_info, (char *) "async_config", (char *) "off");
            }
#endif
            MPI_Win_set_info(win, async_info);

            if (px < PHASE_ITER / 2) {  /* heavy comp */
                t_comp_phase += (MPI_Wtime() - st0);
                t_comp_comp += t_comp;
                t_comp = 0.0;
            }
            else {      /* heavy comm */
                t_comm_phase += (MPI_Wtime() - st0);
                t_comm_comp += t_comp;
                t_comp = 0.0;
            }

#if defined(OUTPUT_ALL_PHASES)
            phase_times[phase_idx++] = (MPI_Wtime() - st0);
#endif
        }

        MPI_Win_unlock_all(win);
        MPI_Win_free(&win);
    }

    t_total = (MPI_Wtime() - t0);       /* s */

    /* min, max, avg */
    MPI_Reduce(&t_total, &sum_total_times[0], 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comp_phase, &sum_t_comp_phases[0], 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm_phase, &sum_t_comm_phases[0], 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comp_comp, &sum_t_comp_comps[0], 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm_comp, &sum_t_comm_comps[0], 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    MPI_Reduce(&t_total, &sum_total_times[1], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comp_phase, &sum_t_comp_phases[1], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm_phase, &sum_t_comm_phases[1], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comp_comp, &sum_t_comp_comps[1], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm_comp, &sum_t_comm_comps[1], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&t_total, &sum_total_times[2], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comp_phase, &sum_t_comp_phases[2], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm_phase, &sum_t_comm_phases[2], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comp_comp, &sum_t_comp_comps[2], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm_comp, &sum_t_comm_comps[2], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        char header[256];

        memset(header, 0, sizeof(header));
        sum_total_times[2] = sum_total_times[2] / nprocs;
        sum_t_comp_phases[2] = sum_t_comp_phases[2] / nprocs;
        sum_t_comm_phases[2] = sum_t_comm_phases[2] / nprocs;
        sum_t_comp_comps[2] = sum_t_comp_comps[2] / nprocs;
        sum_t_comm_comps[2] = sum_t_comm_comps[2] / nprocs;
#ifdef ENABLE_CSP
        const char *nh = getenv("CSP_NG");
        const char *gasync = getenv("CSP_ASYNC_CONFIG");
        if (!strncmp(gasync, "auto", strlen("auto"))) {
            const char *p_lev = getenv("CSP_ASYNC_SCHED_LEVEL");
            const char *p_int = getenv("CSP_RUNTIME_ASYNC_SCHED_MIN_INT");
            const char *p_freq_l = getenv("CSP_RUNTIME_ASYNC_SCHED_THR_L");
            const char *p_freq_h = getenv("CSP_RUNTIME_ASYNC_SCHED_THR_H");
            const char *g_int = getenv("CSP_RUNTIME_ASYNC_TIMED_GSYNC_INT");

            if (!strncmp(p_lev, "anytime", strlen("anytime"))) {
                sprintf(header, "csp-%s-I%sL%sH%sG%s-nh%s", p_lev, p_int, p_freq_l,
                        p_freq_h, g_int, nh);
            }
            else {
                sprintf(header, "csp-%s-I%sL%sH%s-nh%s", p_lev, p_int, p_freq_l, p_freq_h, nh);
            }
        }
        else {
            /* static on/off */
            sprintf(header, "csp-%s-nh%s", gasync, nh);
        }
#elif defined(ENABLE_CSP_ADPT_U)
        const char *nh = getenv("CSP_NG");
        const char *gasync = getenv("CSP_ASYNC_CONFIG");
        const char *p_lev = getenv("CSP_ASYNC_SCHED_LEVEL");
        /* per-window or per-collective user guide */
        sprintf(header, "csp-u-%s-nh%s", p_lev, nh);
#elif defined(DISABLE_SHM)
        sprintf(header, "orig-noshm");
#else
        sprintf(header, "orig");
#endif
        // total_time phase1_total phase2_total phase1_compute phase2_compute
        fprintf(stdout, "%.2lf %.2lf %.2lf %.2lf %.2lf\n",
                sum_total_times[2],
                sum_t_comp_phases[2],
                sum_t_comm_phases[2], sum_t_comp_comps[2], sum_t_comm_comps[2]);

        // fprintf(stdout,
        //         "%s: nprocs %d MS %d %d ML %d %d num_op_s %d num_op_l %d "
        //         "nwins %d nphase %d ncoll %d "
        //         "total_time %.2lf comp-p %.2lf comm-p %.2lf "
        //         "comp-comp %.2lf comp-comm %.2lf\n",
        //         header, nprocs, MS, MS / nprocs, ML, ML / nprocs, NOP_S, NOP_L,
        //         NWINS, PHASE_ITER, COLL_ITER,
        //         sum_total_times[2],
        //         sum_t_comp_phases[2],
        //         sum_t_comm_phases[2], sum_t_comp_comps[2], sum_t_comm_comps[2]);
        fflush(stdout);
#if defined(OUTPUT_ALL_PHASES)
        fprintf(stdout, "%s-phases: ", header);
        for (phase_idx = 0; phase_idx < nphases; phase_idx++) {
            fprintf(stdout, "%.2lf ", phase_times[phase_idx]);
        }
        fprintf(stdout, "\n");
        fflush(stdout);
#endif
    }

    MPI_Info_free(&win_info);
    MPI_Info_free(&async_info);

#if defined(OUTPUT_ALL_PHASES)
    free(phase_times);
#endif

    return errs;
}

static void create_datatype(void)
{
    int sizes[3], subsizes[3], starts[3];
    MPI_Aint lb = 0;

    sizes[0] = sizes[1] = sizes[2] = DLEN;
    subsizes[0] = subsizes[1] = subsizes[2] = SUB_DLEN;
    starts[0] = starts[1] = starts[2] = 1;

    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &target_type);
    MPI_Type_commit(&target_type);

    MPI_Type_get_extent(target_type, &lb, &target_ext);
    MPI_Type_size(target_type, &target_size);
}

int main(int argc, char *argv[])
{
    int i;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    WOKERS = nprocs;
    srand(nprocs);
    // NOP_S NOP_L_MIN NOP_L_MAX NOP_L_ITER MS ML NWINS PHASE_ITER COLL_ITER
    if (nprocs < 2) {
        fprintf(stderr, "Please run using at least two processes\n");
        goto exit;
    }

    if (argc >= 5) {
        NOP_S = atoi(argv[1]);
        NOP_L_MIN = atoi(argv[2]);
        NOP_L_MAX = atoi(argv[3]);
        NOP_L_ITER = atoi(argv[4]);
    }
    if (argc >= 7) {
        MS = atoi(argv[5]);
        ML = atoi(argv[6]);
    }

    if (argc >= 8) {
        NWINS = atoi(argv[7]);
    }

    if (argc >= 9) {
        PHASE_ITER = atoi(argv[8]);
    }

    if (argc >= 10) {
        COLL_ITER = atoi(argv[9]);
    }

    if(argc >= 11){
        DLEN = atoi(argv[10]);
        SUB_DLEN = atoi(argv[11]);
    }

    if(argc >= 13){
        WOKERS = atoi(argv[12]);
    }

    if(argc >= 14){
        COMPT = atoi(argv[13]);
    }

    /* initialize local buffer */
    locbuf = malloc(BUFSIZE * sizeof(double));
    for (i = 0; i < BUFSIZE; i++)
        locbuf[i] = i;

    target_computation_init();
    create_datatype();

    for (NOP_L = NOP_L_MIN; NOP_L <= NOP_L_MAX; NOP_L *= NOP_L_ITER) {
        run_iteration();
    }

  exit:
    target_computation_destroy();
    MPI_Type_free(&target_type);
    free(locbuf);
    MPI_Finalize();

    return 0;
}
