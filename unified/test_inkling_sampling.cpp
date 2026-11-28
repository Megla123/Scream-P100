#include "../shared/inkling_sampling.hpp"
#include "../shared/inkling_repetition.hpp"

#include <array>
#include <cstdio>

int main() {
    const auto parsed = ink2::parse_sampling_fields("0.5 0.9 128 0.125 1.25 0.5 -0.25 23 18446744073709551615", 128);
    if (parsed.temperature != 0.5f || parsed.top_p != 0.9f || parsed.top_k != 128 || parsed.min_p != 0.125f ||
        parsed.rep != 1.25f || parsed.freq != 0.5f || parsed.pres != -0.25f || parsed.last_n != 23 ||
        parsed.seed != std::numeric_limits<uint64_t>::max()) {
        std::fputs("Native sampler did not decode all configuration fields\n", stderr);
        return 1;
    }
    try {
        ink2::parse_sampling_fields("0.5 1 128 0 1 0 0 64 0", 64);
        std::fputs("Native sampler accepted a shortlist beyond its backend's limit\n", stderr);
        return 1;
    } catch (const std::runtime_error&) {
        // Megla uses a 64-entry shortlist while Strata Inkling allows 128.
    }
    constexpr std::array<float, 5> logits{1.2f, 1.0f, 1.0f, 0.9f, 0.8f};
    const std::vector<int32_t> prompt{3, 1};
    ink2::SampleOpts sampler;
    sampler.temperature = 0.85f;
    sampler.top_p = 0.9f;
    sampler.top_k = 4;
    sampler.pres = 1.2f;
    sampler.freq = 0.4f;
    sampler.pen_from = static_cast<int>(prompt.size());

    // A native slot handoff restarts the RNG from the seed and replays the number of prior draws.
    std::mt19937_64 uninterrupted(3133), resumed(3133);
    for (int i = 0; i < 11; ++i) {
        auto row = logits;
        ink2::sample_logits(row.data(), row.size(), sampler, prompt, uninterrupted);
    }
    ink2::skip_target_draws(resumed, 11);
    if (uninterrupted != resumed) {
        std::fputs("A resumed Inkling sampler did not preserve the seeded draw sequence\n", stderr);
        return 1;
    }

    // Compare sequential draws with verification over the same target rows.
    for (const bool reject : {false, true}) {
        auto sequential_history = prompt;
        sequential_history.push_back(0); // Anchor token fed before the first draw.
        std::array<int, 4> window{0};
        std::array<int, 4> picks{};
        std::mt19937_64 sequential(2307), speculative(2307);
        const int last_row = reject ? 1 : 3;
        for (int row = 0; row <= last_row; ++row) {
            auto logits_row = logits;
            picks[row] = ink2::sample_logits(logits_row.data(), logits_row.size(), sampler,
                                            sequential_history, sequential);
            sequential_history.push_back(picks[row]);
            if (row + 1 < static_cast<int>(window.size())) window[row + 1] = picks[row];
        }
        if (reject) window[2] = (picks[1] + 1) % logits.size();
        auto window_history = prompt;
        window_history.insert(window_history.end(), window.begin(), window.end());
        int next = -1;
        const int accepted = ink2::sample_draft_prefix(window.data(), window.size(), next, [&](int row) {
            auto logits_row = logits;
            return ink2::sample_logits(logits_row.data(), logits_row.size(), sampler, window_history,
                                       speculative, prompt.size() + row + 1);
        }, [](int) { return true; });
        if (accepted != last_row || next != picks[last_row] || sequential() != speculative()) {
            std::fprintf(stderr, "Speculative verification changed the seeded draw sequence (reject=%d)\n", reject);
            return 1;
        }
    }
    // The matched token that completes a loop remains pending. Later logits must not be sampled or kept.
    const std::vector<int> output(94, 17);
    const std::array<int, 4> repeated_window{17, 17, 17, 17};
    int drawn = 0, pending = -1;
    const int before_stop = ink2::sample_draft_prefix(repeated_window.data(), repeated_window.size(), pending,
        [&](int) { ++drawn; return 17; }, [&](int count) {
            return !scream::inkling::detect_repetition_loop(output, repeated_window.data() + 1, count);
        });
    if (before_stop != 1 || drawn != 2 || pending != 17) return 1;

    std::vector<int> longer;
    for (int i = 0; i < 319; ++i) longer.push_back(i % 32);
    if (scream::inkling::detect_repetition_loop(longer)) return 1;
    longer.push_back(31);
    if (!scream::inkling::detect_repetition_loop(longer)) return 1;
    longer.back() = -1;
    if (scream::inkling::detect_repetition_loop(longer)) return 1;
    std::puts("Inkling sampler and speculative prefix verification passed");
}
