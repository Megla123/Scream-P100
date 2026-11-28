#include "../shared/inkling_loop.hpp"

#include <array>
#include <cstdio>
#include <vector>

using scream::inkling::LayerVisit;
using scream::inkling::LoopSchedule;
using scream::inkling::inkling2_mini;

int main() {
    // Compare the actual callback order and both cache layouts with the original layer sequence.
    constexpr std::array<int, 12> core{6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    constexpr std::array<int, 6> coda{18, 19, 20, 21, 22, 23};
    for (int passes = 1; passes <= 3; ++passes) {
        const LoopSchedule schedule{24, 6, 18, passes};
        const int slots = 24 + (passes - 1) * 12;
        if (!schedule.fits_trained_slots(inkling2_mini) || schedule.visit_count() != slots) return 1;

        std::vector<int> layers{0, 1, 2, 3, 4, 5};
        std::vector<int> pass_at(6, 0);
        for (int pass = 0; pass < passes; ++pass) {
            layers.insert(layers.end(), core.begin(), core.end());
            pass_at.insert(pass_at.end(), core.size(), pass);
        }
        layers.insert(layers.end(), coda.begin(), coda.end());
        pass_at.insert(pass_at.end(), coda.size(), 0);

        int seen = 0, boundaries = 0;
        std::vector<bool> allocated(slots);
        std::array<bool, inkling2_mini.visit_count()> cuda_slots{};
        const bool completed = schedule.walk([&](LayerVisit visit) {
            if (seen >= slots || visit.index != seen || visit.layer != layers[seen] ||
                visit.pass != pass_at[seen]) return false;
            const int slot = schedule.physical_first_slot(visit);
            if (slot < 0 || slot >= slots || allocated[slot] ||
                schedule.layer_for_physical_first_slot(slot) != visit.layer) return false;
            allocated[slot] = true;
            const int megla_slot = visit.pass == 0 ? visit.layer : 24 + 12 * (visit.pass - 1) + visit.layer - 6;
            const int strata_slot = visit.layer < 18 ? seen : 42 + visit.layer - 18;
            const int trained_slot = schedule.trained_slot(visit, inkling2_mini);
            if (trained_slot != strata_slot || cuda_slots[trained_slot]) return false;
            cuda_slots[trained_slot] = true;
            ++seen;
            return slot == megla_slot;
        }, [&](int pass) {
            if (pass == boundaries && seen == 6 + 12 * pass) ++boundaries;
        });
        if (!completed || seen != slots || boundaries != passes) {
            std::fprintf(stderr, "Inkling visit or cache layout differs at %d pass(es)\n", passes);
            return 1;
        }
        for (int slot = 0; slot < inkling2_mini.visit_count(); ++slot) {
            if (schedule.uses_trained_slot(slot, inkling2_mini) != cuda_slots[slot]) return 1;
        }
        if (schedule.uses_trained_slot(-1, inkling2_mini) ||
            schedule.uses_trained_slot(inkling2_mini.visit_count(), inkling2_mini)) return 1;
    }
    const LoopSchedule invalid[] = {
        {24, -1, 18, 1}, {24, 18, 6, 1}, {24, 6, 25, 1}, {24, 6, 18, 0},
        {24, 6, 18, std::numeric_limits<int>::max()}, {24, 6, 6, 2},
    };
    for (const auto& schedule : invalid) if (schedule.valid()) return 1;
    if (LoopSchedule{24, 6, 18, 4}.fits_trained_slots(inkling2_mini)) return 1;
    if (LoopSchedule{24, 7, 18, 3}.fits_trained_slots(inkling2_mini)) return 1;

    int seen = 0, boundaries = 0;
    if (inkling2_mini.walk([&](LayerVisit) { return ++seen < 7; }, [&](int) { ++boundaries; }) ||
        seen != 7 || boundaries != 1) return 1;
    std::puts("Inkling loop traversal and backend cache layouts passed");
}
