#pragma once

#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace scream {

struct SamplingFields {
    std::optional<float> temperature, top_p, min_p, repetition_penalty, frequency_penalty, presence_penalty;
    std::optional<int> top_k, penalty_last_n;
    std::optional<uint64_t> seed;
};

template <typename Number>
void read_sampling_field(std::istringstream& input, std::optional<Number>& result) {
    std::string word;
    if (!(input >> word)) throw std::runtime_error("invalid sampling fields");
    if (word == "-") return;
    Number number{};
    const auto [end, error] = std::from_chars(word.data(), word.data() + word.size(), number);
    if (error != std::errc{} || end != word.data() + word.size())
        throw std::runtime_error("invalid sampling fields");
    result = number;
}

// The launcher and native workers exchange one ordered block. A dash leaves a field at its backend default.
inline SamplingFields read_sampling_fields(const std::string& fields, int max_top_k,
                                           int max_window = std::numeric_limits<int>::max()) {
    SamplingFields s;
    std::istringstream input(fields);
    input.imbue(std::locale::classic());
    read_sampling_field(input, s.temperature);
    read_sampling_field(input, s.top_p);
    read_sampling_field(input, s.top_k);
    read_sampling_field(input, s.min_p);
    read_sampling_field(input, s.repetition_penalty);
    read_sampling_field(input, s.frequency_penalty);
    read_sampling_field(input, s.presence_penalty);
    read_sampling_field(input, s.penalty_last_n);
    read_sampling_field(input, s.seed);
    std::string extra;
    if (input >> extra) throw std::runtime_error("invalid sampling fields");

    const float t = s.temperature.value_or(0.f);
    const float p = s.top_p.value_or(1.f);
    const float min_p = s.min_p.value_or(0.f);
    const float repeat = s.repetition_penalty.value_or(1.f);
    const float freq = s.frequency_penalty.value_or(0.f);
    const float present = s.presence_penalty.value_or(0.f);
    const int k = s.top_k.value_or(1), last_n = s.penalty_last_n.value_or(0);
    if (!std::isfinite(t) || t < 0.f || !(p > 0.f && p <= 1.f) ||
        k < 1 || k > max_top_k || !(min_p >= 0.f && min_p <= 1.f) ||
        !std::isfinite(repeat) || repeat <= 0.f || !(freq >= -2.f && freq <= 2.f) ||
        !(present >= -2.f && present <= 2.f) || last_n < 0 || last_n > max_window) {
        throw std::runtime_error("sampling fields out of range");
    }
    return s;
}

}  // namespace scream
