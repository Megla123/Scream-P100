// cpu_moe.cu (cpu_patch.py, 2026-11-15 night; IDEJE 7 again, for Q4): FNX_CPU_EXPERTS=1 lets each card's own socket compute some of a
// decode layer's missing experts instead of copying them over PCIe. A card holds half of every expert (gate|up rows [0, 2 ffs), down's
// columns of its ff part), and so does its host copy; the socket computes that half from the copy, the card adds nothing new: the
// pairs' rows land in part like the card's own, and the ring sums them as before.
//   cache_plan (moe.cuh) gives the CPU the misses with the fewest rows (up to FNX_CPU_NMAX rows, at most FNX_CPU_MAX a layer); they
//   take no slot and are not copied. k_cpu_pub publishes them with the window's FFN input (f16, as the row kernels read it) to a
//   mailbox in mapped host memory and sets their jobs' rows to 0 (the row kernels then skip them). The card's workers (FNX_CPU_THREADS,
//   pinned to the card's socket) compute them with ggml's CPU dots (activations as q8_K / q8_1, so not the GPU's bits) and write the
//   pairs' rows; k_cpu_land, just before the ring, waits for them and copies them into part.
// Without FNX_CPU_EXPERTS nothing here runs and the engine computes the same bits as before.
#include "kernels.hpp"
#include "model.hpp"
#include "cpu_moe.hpp"

#include <cuda_runtime.h>
#include <sched.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef FNX_CPU_GGML
#include "ggml.h"
#include "ggml-cpu.h"
#endif

namespace fnx {

namespace {
struct CpuEnt {
    const char* gu;   // the card's half in host memory: [2 ffs][gu_row]
    const char* dn;   //                                 [N][dn_row]
    int n;
    int y[MAXT];      // the window's rows of the expert's tokens
    int out[MAXT];    // their pair rows
};
struct CpuBox {       // mapped host memory, one a card
    volatile unsigned seq;     // the card's publish
    int n, T, layer, gu_type, dn_type, ffs;
    unsigned long long gu_row, dn_row;
    CpuEnt ent[CPU_JOBS];
    alignas(64) volatile unsigned done;   // the workers' answer: the seq they finished
    alignas(64) uint16_t x[MAXT * N];     // the window's FFN input as f16
    alignas(64) float out[MAXT * TOPK * N];   // the pairs' rows (pair row e at e * N)
};
struct CpuDev {       // device memory
    unsigned seq, want;
    unsigned ticket;
    int npairs;
    int pairs[MAXT * TOPK];
};
}  // namespace

// ---------------------------------------------------------------------------------------------------------------------------- device
namespace {
constexpr int PUB_BLOCKS = 8;
__global__ void __launch_bounds__(256) k_cpu_pub(Job* gu, Job* dn, const int* __restrict__ cpu, ExpertCache ec, int layer,
                                                 const uint16_t* __restrict__ xh, int T, CpuBox* box, CpuDev* d, int gu_type, int dn_type,
                                                 int ffs, unsigned long long gu_row, unsigned long long dn_row, const int* njobs, int dbg) {
    int n = min(cpu[0], CPU_JOBS);
    // FNX_CPU_DBG (the warm-up's illegal access, 15. 11.): every listed job checked; 1: the list only (the card computes those jobs itself,
    // from RAM), 2: no pairs copied back
    if (n < 0 || cpu[0] > CPU_JOBS) {
        if (blockIdx.x == 0 && threadIdx.x == 0) printf("fnx cpu: layer %d: list of %d\n", layer, cpu[0]);
        n = 0;
    }
    if (n > 0 && blockIdx.x == 0 && threadIdx.x == 0) {
        const int nj = *njobs;
        for (int k = 0; k < n; ++k) {
            const int j = cpu[1 + k];
            const bool bad = j < 0 || j >= nj || gu[j].w < 0 || gu[j].w >= NEXP || gu[j].n < 1 || gu[j].n > MAXT ||
                             !ec.host_gu[layer * NEXP + (j >= 0 && j < nj ? gu[j].w & (NEXP - 1) : 0)];
            if (bad) printf("fnx cpu: layer %d T %d: entry %d of %d: job %d of %d, w %d, rows %d\n", layer, T, k, n, j, nj,
                            j >= 0 && j < nj ? gu[j].w : -9999, j >= 0 && j < nj ? gu[j].n : -9999);
        }
    }
    if (dbg == 1) n = 0;
    if (n == 0) {
        if (blockIdx.x == 0 && threadIdx.x == 0) d->want = 0;
        return;
    }
    // the input rows, every block a share (uint4 = 8 halves)
    const uint4* src = reinterpret_cast<const uint4*>(xh);
    uint4* dst = reinterpret_cast<uint4*>(box->x);
    const int words = T * N / 8;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < words; i += PUB_BLOCKS * 256) dst[i] = src[i];
    if (blockIdx.x == 0) {
        for (int k = threadIdx.x; k < n; k += 256) {
            const int j = cpu[1 + k], g = layer * NEXP + gu[j].w, m = gu[j].n;
            CpuEnt e;
            e.gu = ec.host_gu[g];
            e.dn = ec.host_dn[g];
            e.n = m;
            for (int s = 0; s < MAXT; ++s) {
                e.y[s] = s < m ? gu[j].y[s] : 0;
                e.out[s] = s < m ? dn[j].out[s] : 0;
            }
            box->ent[k] = e;
        }
        __syncthreads();
        if (threadIdx.x == 0) {   // the pairs for k_cpu_land, then the jobs' rows to 0 (the row kernels skip them)
            int np = 0;
            for (int k = 0; k < n; ++k) {
                const int j = cpu[1 + k], m = gu[j].n;
                for (int s = 0; s < m; ++s) d->pairs[np++] = dn[j].out[s];
                gu[j].n = 0;
                dn[j].n = 0;
            }
            d->npairs = np;
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0 && atomicAdd(&d->ticket, 1u) == PUB_BLOCKS - 1) {   // the last block: the mailbox's head, then its seq
        __threadfence_system();
        box->n = n;
        box->T = T;
        box->layer = layer;
        box->gu_type = gu_type;
        box->dn_type = dn_type;
        box->ffs = ffs;
        box->gu_row = gu_row;
        box->dn_row = dn_row;
        const unsigned s = ++d->seq;
        d->want = s;
        d->ticket = 0;
        __threadfence_system();
        box->seq = s;
        __threadfence_system();
    }
}

constexpr int LAND_BLOCKS = 16;
__global__ void __launch_bounds__(256) k_cpu_land(CpuDev* d, const CpuBox* box, float* __restrict__ part, unsigned* err, int dbg) {
    const unsigned want = *(volatile unsigned*) &d->want;
    if (want == 0) return;
    if (blockIdx.x == 0 && threadIdx.x == 0)
        for (int i = 0; i < d->npairs; ++i)
            if (d->pairs[i] < 0 || d->pairs[i] >= MAXT * TOPK) printf("fnx cpu: land: pair %d of %d is %d\n", i, d->npairs, d->pairs[i]);
    if (threadIdx.x == 0) {
        const long long t0 = clock64();
        while ((int) (box->done - want) < 0)
            if (clock64() - t0 > 4000000000LL) {   // ~3 s: the workers are gone
                atomicOr(err, 2u);
                break;
            }
        __threadfence_system();
    }
    __syncthreads();
    if (dbg == 2) return;
    const int np = d->npairs, per = N / 4;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < np * per; i += LAND_BLOCKS * 256) {
        const int e = d->pairs[i / per], c = i % per;
        // volatile scalar loads (ld.volatile, as the flags are read): __ldcv's ld.global.cv of mapped host memory was an illegal access
        const volatile float* src = box->out + (size_t) e * N + 4 * c;
        reinterpret_cast<float4*>(part + (size_t) e * N)[c] = make_float4(src[0], src[1], src[2], src[3]);
    }
}
}  // namespace

// ------------------------------------------------------------------------------------------------------------------------------ host
namespace {
struct CpuCard {
    int dev = 0, card = 0;
    CpuBox* box = nullptr;
    CpuDev* d = nullptr;
    unsigned* err = nullptr;
    std::vector<std::thread> th;
    std::atomic<bool> stop{false};
    std::atomic<unsigned> gen{0};
    std::atomic<int> next{0}, done_items{0}, x_left{0};
    std::atomic<int> gu_left[CPU_JOBS];
    std::atomic<int> h_ready[CPU_JOBS];
    int total = 0, nx = 0, ngu = 0, n = 0, T = 0, ffs = 0;
    size_t gu_row = 0, dn_row = 0, xq_row = 0, hq_row = 0;
    CpuEnt ent[CPU_JOBS];
#ifdef FNX_CPU_GGML
    const ggml_type_traits_cpu* tgu = nullptr;
    const ggml_type_traits_cpu* tdn = nullptr;
    // the activations go to the dots' types by those types' own from_float (a weight type's from_float makes that type)
    const ggml_type_traits_cpu* txq = nullptr;
    const ggml_type_traits_cpu* thq = nullptr;
#endif
    std::vector<uint8_t> xq;   // [MAXT][xq_row]
    std::vector<float> g;      // [CPU_JOBS][MAXT][2 ffs]
    std::vector<uint8_t> hq;   // [CPU_JOBS][MAXT][hq_row]
    // FNX_CPU_STATS
    long st_layers = 0, st_jobs = 0, st_pairs = 0;
    double st_us = 0;
};
constexpr int GU_CHUNK = 128, DN_CHUNK = 512;

int env_i(const char* k, int d) {
    const char* e = std::getenv(k);
    return e && *e ? std::atoi(e) : d;
}

void cpu_pin(int dev) {
    char bus[32] = {};
    if (cudaDeviceGetPCIBusId(bus, sizeof bus, dev) != cudaSuccess) return;
    std::string id(bus);
    for (auto& ch : id) ch = (char) std::tolower((unsigned char) ch);
    std::ifstream f("/sys/bus/pci/devices/" + id + "/local_cpulist");
    std::string s;
    if (!f || !std::getline(f, s)) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    size_t i = 0;
    while (i < s.size()) {
        const size_t j = s.find(',', i);
        const std::string part = s.substr(i, j == std::string::npos ? std::string::npos : j - i);
        const size_t dash = part.find('-');
        const int a = std::atoi(part.c_str()), b = dash == std::string::npos ? a : std::atoi(part.c_str() + dash + 1);
        for (int c = a; c <= b; ++c) CPU_SET(c, &set);
        if (j == std::string::npos) break;
        i = j + 1;
    }
    sched_setaffinity(0, sizeof set, &set);
}

inline void relax() { __builtin_ia32_pause(); }

#ifdef FNX_CPU_GGML
// one work item: [0, nx) an input row to q8_K (gate|up's dot type), [nx, nx + ngu) a gate|up chunk of a job, then down chunks
void cpu_item(CpuCard* c, int it) {
    const int K = N, ffs = c->ffs;
    if (it < c->nx) {
        float x[N];
        ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t*>(c->box->x) + (size_t) it * K, x, K);
        c->txq->from_float(x, c->xq.data() + (size_t) it * c->xq_row, K);
        c->x_left.fetch_sub(1, std::memory_order_acq_rel);
        return;
    }
    while (c->x_left.load(std::memory_order_acquire) > 0) relax();
    const int gpj = (2 * ffs + GU_CHUNK - 1) / GU_CHUNK;
    if (it < c->nx + c->ngu) {
        const int i = it - c->nx, j = i / gpj, ch = i % gpj;
        const CpuEnt& e = c->ent[j];
        float* gj = c->g.data() + (size_t) j * MAXT * 2 * ffs;
        const int r1 = std::min(2 * ffs, (ch + 1) * GU_CHUNK);
        for (int r = ch * GU_CHUNK; r < r1; ++r) {
            const char* w = e.gu + (size_t) r * c->gu_row;
            for (int s = 0; s < e.n; ++s) c->tgu->vec_dot(K, gj + (size_t) s * 2 * ffs + r, 0, w, 0, c->xq.data() + (size_t) e.y[s] * c->xq_row, 0, 1);
        }
        if (c->gu_left[j].fetch_sub(1, std::memory_order_acq_rel) == 1) {   // the job's last chunk: its h = silu(gate) * up, as down's dot type
            float h[FF];
            for (int s = 0; s < e.n; ++s) {
                const float* gs = gj + (size_t) s * 2 * ffs;
                for (int q = 0; q < ffs; ++q) h[q] = gs[q] / (1.0f + std::exp(-gs[q])) * gs[ffs + q];
                c->thq->from_float(h, c->hq.data() + ((size_t) j * MAXT + s) * c->hq_row, ffs);
            }
            c->h_ready[j].store(1, std::memory_order_release);
        }
        return;
    }
    const int dpj = (N + DN_CHUNK - 1) / DN_CHUNK;
    const int i = it - c->nx - c->ngu, j = i / dpj, ch = i % dpj;
    while (!c->h_ready[j].load(std::memory_order_acquire)) relax();
    const CpuEnt& e = c->ent[j];
    const int r1 = std::min(N, (ch + 1) * DN_CHUNK);
    for (int r = ch * DN_CHUNK; r < r1; ++r) {
        const char* w = e.dn + (size_t) r * c->dn_row;
        for (int s = 0; s < e.n; ++s) {
            float v;
            c->tdn->vec_dot(ffs, &v, 0, w, 0, c->hq.data() + ((size_t) j * MAXT + s) * c->hq_row, 0, 1);
            c->box->out[(size_t) e.out[s] * N + r] = v;
        }
    }
}
#endif

void cpu_work(CpuCard* c) {
#ifdef FNX_CPU_GGML
    for (int it; (it = c->next.fetch_add(1, std::memory_order_acq_rel)) < c->total;) {
        cpu_item(c, it);
        c->done_items.fetch_add(1, std::memory_order_acq_rel);
    }
#else
    (void) c;
#endif
}

void cpu_helper(CpuCard* c) {
    cpu_pin(c->dev);
    unsigned seen = c->gen.load();
    auto idle = std::chrono::steady_clock::now();
    while (!c->stop.load(std::memory_order_relaxed)) {
        const unsigned g = c->gen.load(std::memory_order_acquire);
        if (g == seen) {
            if (std::chrono::steady_clock::now() - idle > std::chrono::milliseconds(20)) std::this_thread::sleep_for(std::chrono::microseconds(100));
            else relax();
            continue;
        }
        seen = g;
        cpu_work(c);
        idle = std::chrono::steady_clock::now();
    }
}

void cpu_master(CpuCard* c) {
    cudaSetDevice(c->dev);
    cpu_pin(c->dev);
    static const bool stats = env_i("FNX_CPU_STATS", 0) != 0;
    unsigned last = c->box->seq;
    auto idle = std::chrono::steady_clock::now();
    while (!c->stop.load(std::memory_order_relaxed)) {
        const unsigned s = c->box->seq;
        if (s == last) {
            if (std::chrono::steady_clock::now() - idle > std::chrono::milliseconds(20)) std::this_thread::sleep_for(std::chrono::microseconds(100));
            else relax();
            continue;
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        const auto t0 = std::chrono::steady_clock::now();
        c->n = std::min(c->box->n, CPU_JOBS);
        c->T = c->box->T;
        c->ffs = c->box->ffs;
        c->gu_row = (size_t) c->box->gu_row;
        c->dn_row = (size_t) c->box->dn_row;
        int pairs = 0;
        for (int k = 0; k < c->n; ++k) {
            c->ent[k] = c->box->ent[k];
            pairs += c->ent[k].n;
        }
#ifdef FNX_CPU_GGML
        c->tgu = ggml_get_type_traits_cpu((ggml_type) c->box->gu_type);
        c->tdn = ggml_get_type_traits_cpu((ggml_type) c->box->dn_type);
        c->txq = ggml_get_type_traits_cpu(c->tgu->vec_dot_type);
        c->thq = ggml_get_type_traits_cpu(c->tdn->vec_dot_type);
        c->xq_row = ggml_row_size(c->tgu->vec_dot_type, N);
        c->hq_row = ggml_row_size(c->tdn->vec_dot_type, c->ffs);
        if (c->xq.size() < (size_t) MAXT * c->xq_row) c->xq.resize((size_t) MAXT * c->xq_row);
        if (c->hq.size() < (size_t) CPU_JOBS * MAXT * c->hq_row) c->hq.resize((size_t) CPU_JOBS * MAXT * c->hq_row);
        const int gpj = (2 * c->ffs + GU_CHUNK - 1) / GU_CHUNK, dpj = (N + DN_CHUNK - 1) / DN_CHUNK;
        for (int k = 0; k < c->n; ++k) {
            c->gu_left[k].store(gpj, std::memory_order_relaxed);
            c->h_ready[k].store(0, std::memory_order_relaxed);
        }
        c->nx = c->T;
        c->ngu = c->n * gpj;
        c->total = c->nx + c->ngu + c->n * dpj;
        c->x_left.store(c->nx, std::memory_order_relaxed);
        c->done_items.store(0, std::memory_order_relaxed);
        c->next.store(0, std::memory_order_release);
        c->gen.fetch_add(1, std::memory_order_acq_rel);
        cpu_work(c);
        while (c->done_items.load(std::memory_order_acquire) < c->total) relax();
#endif
        std::atomic_thread_fence(std::memory_order_seq_cst);
        c->box->done = s;
        last = s;
        idle = std::chrono::steady_clock::now();
        if (stats) {
            c->st_layers++;
            c->st_jobs += c->n;
            c->st_pairs += pairs;
            c->st_us += std::chrono::duration<double, std::micro>(idle - t0).count();
            if (c->st_layers % 20000 == 0) {
                std::fprintf(stderr, "fnx cpu: card %d: %ld layers with jobs, %.2f jobs and %.2f pairs a layer, %.1f us a layer\n", c->card,
                             c->st_layers, (double) c->st_jobs / c->st_layers, (double) c->st_pairs / c->st_layers, c->st_us / c->st_layers);
            }
        }
    }
}
}  // namespace

bool cpu_moe_on() {
#ifdef FNX_CPU_GGML
    static const bool on = env_i("FNX_CPU_EXPERTS", 0) != 0;
    return on;
#else
    static const bool warn = [] {
        if (env_i("FNX_CPU_EXPERTS", 0)) std::fprintf(stderr, "fnx: FNX_CPU_EXPERTS: this build has no ggml CPU library (FNX_CPU_GGML_LIBS), off\n");
        return false;
    }();
    return warn;
#endif
}
int cpu_moe_max() {
    static const int v = std::max(0, std::min(CPU_JOBS, env_i("FNX_CPU_MAX", 8)));
    return v;
}
int cpu_moe_nmax() {
    static const int v = std::max(1, std::min(MAXT, env_i("FNX_CPU_NMAX", 2)));
    return v;
}
int cpu_moe_min_t() {
    static const int v = std::max(1, env_i("FNX_CPU_MIN_T", 1));
    return v;
}

void* cpu_moe_start(int dev, int card) {
#ifdef FNX_CPU_GGML
    static const bool init = [] {
        ggml_cpu_init();
        return true;
    }();
    (void) init;
#endif
    auto* c = new CpuCard;
    c->dev = dev;
    c->card = card;
    for (int k = 0; k < CPU_JOBS; ++k) c->gu_left[k].store(0), c->h_ready[k].store(0);
    cudaSetDevice(dev);
    void* p = nullptr;
    if (cudaHostAlloc(&p, sizeof(CpuBox), cudaHostAllocPortable | cudaHostAllocMapped) != cudaSuccess)
        throw std::runtime_error("FNX_CPU_EXPERTS: cudaHostAlloc of the mailbox failed");
    std::memset(p, 0, sizeof(CpuBox));
    c->box = (CpuBox*) p;
    if (cudaMalloc(&c->d, sizeof(CpuDev)) != cudaSuccess || cudaMemset(c->d, 0, sizeof(CpuDev)) != cudaSuccess)
        throw std::runtime_error("FNX_CPU_EXPERTS: the card's words");
    if (cudaMalloc(&c->err, sizeof(unsigned)) != cudaSuccess || cudaMemset(c->err, 0, sizeof(unsigned)) != cudaSuccess)
        throw std::runtime_error("FNX_CPU_EXPERTS: the error word");
    c->g.resize((size_t) CPU_JOBS * MAXT * 2 * FF);
    const int nt = std::max(1, env_i("FNX_CPU_THREADS", 6));
    c->th.emplace_back(cpu_master, c);
    for (int i = 1; i < nt; ++i) c->th.emplace_back(cpu_helper, c);
    std::fprintf(stderr, "fnx: FNX_CPU_EXPERTS: card %d: %d threads on its socket, at most %d experts a layer of up to %d rows, from windows of %d rows\n",
                 card, nt, cpu_moe_max(), cpu_moe_nmax(), cpu_moe_min_t());
    return c;
}
void cpu_moe_stop(void* h) {
    auto* c = (CpuCard*) h;
    c->stop.store(true);
    for (auto& t : c->th) t.join();
    cudaFreeHost(c->box);
    cudaFree(c->d);
    cudaFree(c->err);
    delete c;
}
void cpu_moe_publish(void* h, Job* jobs_gu, Job* jobs_dn, const int* njobs, const int* cpu_list, const ExpertCache& ec, int layer, const uint16_t* xh16, int T,
                     int gu_type, int dn_type, int ffs, size_t gu_row, size_t dn_row, void* st) {
    auto* c = (CpuCard*) h;
    static const int dbg = env_i("FNX_CPU_DBG", 0);
    k_cpu_pub<<<PUB_BLOCKS, 256, 0, (cudaStream_t) st>>>(jobs_gu, jobs_dn, cpu_list, ec, layer, xh16, T, c->box, c->d, gu_type, dn_type, ffs,
                                                         (unsigned long long) gu_row, (unsigned long long) dn_row, njobs, dbg);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("cpu_moe_publish: ") + cudaGetErrorString(e));
}
void cpu_moe_land(void* h, float* part, void* st) {
    auto* c = (CpuCard*) h;
    static const int dbg = env_i("FNX_CPU_DBG", 0);
    k_cpu_land<<<LAND_BLOCKS, 256, 0, (cudaStream_t) st>>>(c->d, c->box, part, c->err, dbg);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("cpu_moe_land: ") + cudaGetErrorString(e));
}
unsigned cpu_moe_err(void* h) {
    auto* c = (CpuCard*) h;
    unsigned v = 0;
    cudaMemcpy(&v, c->err, sizeof v, cudaMemcpyDeviceToHost);
    return v;
}

}  // namespace fnx
