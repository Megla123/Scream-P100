#pragma once

#include <limits>

namespace scream::inkling {

struct LayerVisit {
    int index;
    int layer;
    int pass;
};

struct LoopSchedule {
    int physical_layers;
    int core_begin;
    int core_end;
    int passes;

    constexpr int core_length() const noexcept { return core_end - core_begin; }

    constexpr bool valid() const noexcept {
        return physical_layers > 0 && 0 <= core_begin && core_begin <= core_end &&
               core_end <= physical_layers && passes > 0 &&
               (core_length() == 0 ? passes == 1
                : passes - 1 <= (std::numeric_limits<int>::max() - physical_layers) / core_length());
    }

    constexpr int visit_count() const noexcept {
        return physical_layers + (passes - 1) * core_length();
    }

    constexpr int coda_begin() const noexcept { return core_end + (passes - 1) * core_length(); }

    // Megla indexes the first visit by physical layer and appends the repeated core passes.
    constexpr int physical_first_slot(LayerVisit visit) const noexcept {
        return visit.pass == 0 ? visit.layer
            : physical_layers + (visit.pass - 1) * core_length() + (visit.layer - core_begin);
    }

    constexpr int layer_for_physical_first_slot(int slot) const noexcept {
        return slot < physical_layers ? slot : core_begin + (slot - physical_layers) % core_length();
    }

    // Strata's specialized CUDA path leaves a gap before the coda when fewer passes are run.
    constexpr int trained_slot(LayerVisit visit, const LoopSchedule& trained) const noexcept {
        return visit.layer < core_end ? visit.index : trained.coda_begin() + visit.layer - core_end;
    }

    constexpr bool uses_trained_slot(int slot, const LoopSchedule& trained) const noexcept {
        return slot >= 0 && slot < trained.visit_count() &&
               (slot < coda_begin() || slot >= trained.coda_begin());
    }

    constexpr bool fits_trained_slots(const LoopSchedule& trained) const noexcept {
        return valid() && physical_layers == trained.physical_layers && core_begin == trained.core_begin &&
               core_end == trained.core_end && passes <= trained.passes;
    }

    template<class Visit, class Begin>
    bool walk(Visit&& visit, Begin&& begin_pass) const {
        int index = 0;
        for (int layer = 0; layer < core_begin; ++layer) {
            if (!visit(LayerVisit{index++, layer, 0})) return false;
        }
        for (int pass = 0; pass < passes; ++pass) {
            begin_pass(pass);
            for (int layer = core_begin; layer < core_end; ++layer) {
                if (!visit(LayerVisit{index++, layer, pass})) return false;
            }
        }
        for (int layer = core_end; layer < physical_layers; ++layer) {
            if (!visit(LayerVisit{index++, layer, 0})) return false;
        }
        return true;
    }
};

inline constexpr LoopSchedule inkling2_mini{24, 6, 18, 3};

}  // namespace scream::inkling
