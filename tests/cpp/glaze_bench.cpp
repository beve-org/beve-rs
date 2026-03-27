#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "glaze/beve/read.hpp"
#include "glaze/beve/write.hpp"

struct LargeVecs {
    std::string name{};
    std::vector<double> values{};
    std::vector<uint32_t> ids{};
    std::vector<bool> flags{};
};

template <>
struct glz::meta<LargeVecs> {
    using T = LargeVecs;
    static constexpr auto value = object(
        "name", &T::name,
        "values", &T::values,
        "ids", &T::ids,
        "flags", &T::flags);
};

static LargeVecs make_data(size_t n) {
    LargeVecs d;
    d.name = "benchmark";
    d.values.resize(n);
    d.ids.resize(n);
    d.flags.resize(n);
    for (size_t i = 0; i < n; ++i) {
        d.values[i] = std::sin(static_cast<double>(i) * 0.001) * 100.0;
        d.ids[i] = static_cast<uint32_t>(i);
        d.flags[i] = (i % 3 == 0);
    }
    return d;
}

int main(int argc, char** argv) {
    size_t n = 100000;
    int iters = 1000;
    if (argc >= 2) n = static_cast<size_t>(std::atol(argv[1]));
    if (argc >= 3) iters = std::atoi(argv[2]);

    auto data = make_data(n);

    // Warmup
    std::string buf;
    glz::write_beve(data, buf);
    size_t encoded_size = buf.size();

    // Benchmark write
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        buf.clear();
        glz::write_beve(data, buf);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double write_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;

    // Benchmark read
    LargeVecs out;
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        out = LargeVecs{};
        glz::read_beve(out, buf);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double read_ns = std::chrono::duration<double, std::nano>(t3 - t2).count() / iters;

    std::printf("elements:     %zu\n", n);
    std::printf("iterations:   %d\n", iters);
    std::printf("encoded_size: %zu bytes\n", encoded_size);
    std::printf("write:        %.0f ns  (%.2f MB/s)\n", write_ns,
                (double)encoded_size / write_ns * 1e3);
    std::printf("read:         %.0f ns  (%.2f MB/s)\n", read_ns,
                (double)encoded_size / read_ns * 1e3);
    return 0;
}
