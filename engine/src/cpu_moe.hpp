// cpu_moe.hpp (cpu_patch.py): FNX_CPU_EXPERTS, a decode layer's missing experts with few rows computed by the card's socket (cpu_moe.cu)
#pragma once
#include <cstddef>
#include <cstdint>

namespace fnx {
struct Job;
struct ExpertCache;
constexpr int CPU_JOBS = 64;   // the most a card's socket takes a layer
bool cpu_moe_on();
int cpu_moe_max();
int cpu_moe_nmax();
int cpu_moe_min_t();
void* cpu_moe_start(int dev, int card);
void cpu_moe_stop(void* h);
// after the router's plan (its CPU list in cpu_list) and the window's f16 rows: the mailbox to the workers, the CPU jobs' rows to 0
void cpu_moe_publish(void* h, Job* jobs_gu, Job* jobs_dn, const int* njobs, const int* cpu_list, const ExpertCache& ec, int layer, const uint16_t* xh16, int T,
                     int gu_type, int dn_type, int ffs, size_t gu_row, size_t dn_row, void* st);
// before the ring: wait for the workers and copy their pairs' rows into part
void cpu_moe_land(void* h, float* part, void* st);
unsigned cpu_moe_err(void* h);
}  // namespace fnx
