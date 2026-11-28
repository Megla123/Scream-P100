#include "../shared/inference_sampling_fields.hpp"

#include <array>
#include <cstdio>

int main() {
    const auto fields = scream::read_sampling_fields("0.5 - 32 0.125 1.25 -0.5 0.25 4096 0", 64, 4096);
    if (fields.temperature != 0.5f || fields.top_p || fields.top_k != 32 || fields.min_p != 0.125f ||
        fields.repetition_penalty != 1.25f || fields.frequency_penalty != -0.5f ||
        fields.presence_penalty != 0.25f || fields.penalty_last_n != 4096 || fields.seed != 0) return 1;
    const auto defaults = scream::read_sampling_fields("- - - - - - - - -", 64);
    if (defaults.seed || defaults.temperature || defaults.penalty_last_n) return 2;
    const auto max_seed = scream::read_sampling_fields("- - - - - - - - 18446744073709551615", 64);
    if (max_seed.seed != std::numeric_limits<uint64_t>::max()) return 3;

    constexpr std::array invalid{
        "0.5 - - - - - - -", "0.5 - - - - - - - - extra", "nan - - - - - - - -",
        "0.5 - 65 - - - - - -", "- - - - - - - 4097 -", "0.5 - - - - - - - -3",
        "- - - - - - - - 18446744073709551616",
    };
    for (const char* value : invalid) {
        try {
            (void) scream::read_sampling_fields(value, 64, 4096);
            std::fprintf(stderr, "Accepted invalid sampling fields %s\n", value);
            return 4;
        } catch (const std::runtime_error&) {
        }
    }
    std::puts("Native sampling fields retain unset defaults and explicit seeds");
}
