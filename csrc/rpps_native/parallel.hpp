// Run independent items (codewords, sequences) on several threads. Called with the GIL released.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace rpps {

// threads: 0 = RPPS_THREADS from the environment, else the hardware's count; never more than the items
inline int thread_count(long long items, int threads) {
    if (threads <= 0) {
        const char *env = std::getenv("RPPS_THREADS");
        threads = env ? std::atoi(env) : 0;
        if (threads <= 0)
            threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    }
    return static_cast<int>(std::max<long long>(1, std::min<long long>(threads, items)));
}

// fn(i) for every i in [0, n), items handed out one at a time. The first exception is rethrown here.
template <typename F>
void parallel_for(long long n, int threads, F fn) {
    const int T = thread_count(n, threads);
    if (T <= 1) {
        for (long long i = 0; i < n; i++) fn(i);
        return;
    }
    std::atomic<long long> next{0};
    std::exception_ptr err;
    std::mutex err_lock;
    auto work = [&]() {
        for (;;) {
            const long long i = next.fetch_add(1);
            if (i >= n) return;
            try {
                fn(i);
            } catch (...) {
                std::lock_guard<std::mutex> g(err_lock);
                if (!err) err = std::current_exception();
                next = n;
                return;
            }
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(T - 1);
    for (int t = 1; t < T; t++) pool.emplace_back(work);
    work();
    for (auto &th : pool) th.join();
    if (err) std::rethrow_exception(err);
}

} // namespace rpps
