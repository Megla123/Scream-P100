#pragma once

#include <vector>

namespace scream::inkling {

inline constexpr int max_lookup_match = 16;

// Take tokens following the most recent occurrence of the longest matching text suffix.
template <typename Token>
std::vector<Token> prompt_lookup(const std::vector<Token>& history, int limit, int vocab,
                                 int* match_length = nullptr, int image_token = -1) {
    if (match_length) *match_length = 0;
    const int n = static_cast<int>(history.size());
    if (limit <= 0 || n < 4) return {};
    const auto text_token = [=](Token t) { return t >= 0 && t < vocab && t != image_token; };
    if (!text_token(history.back())) return {};
    int best = 0, at = -1;
    for (int p = n - 2; p >= 0 && best < max_lookup_match; --p) {
        if (history[p] != history.back()) continue;
        int len = 1;
        while (len < max_lookup_match && p - len >= 0 && history[p - len] == history[n - 1 - len] &&
               text_token(history[p - len])) ++len;
        if (len > best) {
            best = len;
            at = p;
        }
    }
    if (best < 3) return {};
    if (match_length) *match_length = best;
    std::vector<Token> draft;
    for (int i = at + 1; i < n && static_cast<int>(draft.size()) < limit && text_token(history[i]); ++i) {
        draft.push_back(history[i]);
    }
    return draft;
}

}  // namespace scream::inkling
