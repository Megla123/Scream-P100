#include "../shared/inkling_lookup.hpp"
#include <cstdio>
#include <vector>

int main() {
    using scream::inkling::prompt_lookup;
    int matched = -1;
    const auto longer = prompt_lookup(std::vector<int>{8, 1, 2, 3, 4, 5, 9, 1, 2, 3, 7, 8, 1, 2, 3}, 2, 20, &matched);
    if (longer != std::vector<int>{4, 5} || matched != 4) return 1;

    const auto latest = prompt_lookup(std::vector<int>{1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 3}, 1, 20, &matched);
    if (latest != std::vector<int>{5} || matched != 3) return 2;

    const auto image_boundary = prompt_lookup(std::vector<int>{1, 2, 3, 4, 19, 8, 1, 2, 3}, 5, 20, &matched, 19);
    if (image_boundary != std::vector<int>{4} || matched != 3) return 3;
    if (!prompt_lookup(std::vector<int>{1, 19, 3, 4, 1, 19, 3}, 5, 20, &matched, 19).empty() || matched != 0) return 4;
    if (!prompt_lookup(std::vector<int>{1, 2, 3, 4, 1, 2, 3}, 0, 20, &matched).empty() || matched != 0) return 5;
    std::puts("Inkling prompt lookup matches longest and latest text suffixes and stops at image tokens");
}
