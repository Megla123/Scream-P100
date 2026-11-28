#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

int scream_ink2_v1_main(int argc, char** argv);
int scream_strata_inkling_main(int argc, char** argv);
int scream_strata_main(int argc, char** argv);
#ifdef SCREAM_HAS_UPSTREAM_STRATA
int scream_strata_upstream_main(int argc, char** argv);
#endif

namespace {

constexpr struct Backend {
    const char* name;
    const char* executable;
    int (*run)(int, char**);
} backends[] = {
    {"ink2_v1", "ink2_v1", scream_ink2_v1_main},
    {"strata-inkling", "strata-inkling", scream_strata_inkling_main},
    {"strata-split", "strata-split-2", scream_strata_main},
#ifdef SCREAM_HAS_UPSTREAM_STRATA
    {"strata", "scream-strata-layer", scream_strata_upstream_main},
#endif
};

int run_backend(const Backend& backend, int argc, char** argv) {
    try {
        return backend.run(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s: %s\n", backend.name, error.what());
        return 2;
    }
}

int run_launcher(int argc, char** argv) {
    std::error_code error;
    const auto executable = std::filesystem::read_symlink("/proc/self/exe", error);
    if (error) {
        std::fprintf(stderr, "scream-p100: cannot locate this executable: %s\n", error.message().c_str());
        return 2;
    }
    const auto record = executable.parent_path() / "scream-launcher.path";
    std::ifstream location(record);
    std::string script;
    if (!std::getline(location, script) || script.empty()) {
        std::fprintf(stderr, "scream-p100: cannot read the profile launcher path from %s\n", record.c_str());
        return 2;
    }
    if (setenv("SCREAM_ENTRY_BINARY", executable.c_str(), 1) != 0) {
        std::perror("scream-p100: set launcher environment");
        return 2;
    }
    std::string launcher = (executable.parent_path() / script).lexically_normal().string();
    char python[] = "python3";
    std::vector<char*> command{python, launcher.data()};
    const int first = argc > 1 && std::strcmp(argv[1], "--configured") == 0 ? 2 : 1;
    command.insert(command.end(), argv + first, argv + argc);
    command.push_back(nullptr);
    execvp(python, command.data());
    std::perror("scream-p100: launch python3");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--list-backends") == 0) {
        for (const auto& backend : backends) std::puts(backend.name);
        return 0;
    }
    if (argc >= 3 && std::strcmp(argv[1], "--backend") == 0) {
        for (const auto& backend : backends) {
            if (std::strcmp(argv[2], backend.name) == 0) {
                return run_backend(backend, argc - 2, argv + 2);
            }
        }
        std::fprintf(stderr, "Unknown backend %s\n", argv[2]);
        return 2;
    }
    const char* slash = std::strrchr(argv[0], '/');
    const char* executable = slash ? slash + 1 : argv[0];
    for (const auto& backend : backends) {
        if (std::strcmp(executable, backend.executable) == 0) {
            return run_backend(backend, argc, argv);
        }
    }
    return run_launcher(argc, argv);
}
