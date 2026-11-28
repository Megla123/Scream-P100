#pragma once

#include <cstddef>
#include <vector>

namespace scream::inkling {

inline constexpr std::size_t repetition_guard_max_period = 32;
inline constexpr std::size_t repetition_guard_long_copies = 10;
inline constexpr std::size_t repetition_guard_history = repetition_guard_max_period * repetition_guard_long_copies;

// Strata's token repetition guard (#6620). Check a verified draft suffix before committing it.
template <typename Token>
bool detect_repetition_loop(const Token* output, std::size_t output_size, const Token* tail = nullptr,
                            std::size_t tail_size = 0) noexcept {
    const std::size_t count = output_size + tail_size;
    if (count < 96) return false;
    const auto from_end = [&](std::size_t k) {
        return k < tail_size ? tail[tail_size - 1 - k] : output[count - 1 - k];
    };
    for (std::size_t period = 1; period <= repetition_guard_max_period; ++period) {
        const std::size_t copies = period <= 4 ? 24 : repetition_guard_long_copies;
        if (count < period * copies) continue;
        std::size_t k = 0;
        for (; k < period * (copies - 1); ++k) {
            if (from_end(k) != from_end(k + period)) break;
        }
        if (k == period * (copies - 1)) return true;
    }
    return false;
}

template <typename Token>
bool detect_repetition_loop(const std::vector<Token>& output, const Token* tail = nullptr,
                            std::size_t tail_size = 0) noexcept {
    return detect_repetition_loop(output.data(), output.size(), tail, tail_size);
}

}  // namespace scream::inkling
