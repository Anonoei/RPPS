// Run independent items (codewords, sequences, output chunks) on several threads. Called with the GIL released.
//
// The threads are a pool started on first use and kept for the life of the process: a per-block kernel would otherwise
// pay a thread start and join (tens of microseconds each) on every call. Several Python threads may call kernels at
// once, and a kernel may run inside another's item: the caller always works through its own items too, and takes back
// the tickets no worker has picked up, so it never waits on a busy pool.
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#ifndef _WIN32
#include <pthread.h>
#endif

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

namespace detail {

// One parallel_for call: its items are claimed one at a time by whoever runs it (the caller and the workers it got)
struct Job {
    const std::function<void(long long)> *fn;
    long long n;
    std::atomic<long long> next{0};
    std::exception_ptr err;
    std::mutex lock;
    std::condition_variable done;
    int running = 0; // workers inside run(); changed under `lock` (and taken up under the pool's lock)

    void run() {
        for (;;) {
            const long long i = next.fetch_add(1);
            if (i >= n) return;
            try {
                (*fn)(i);
            } catch (...) {
                std::lock_guard<std::mutex> g(lock);
                if (!err) err = std::current_exception();
                next = n;
                return;
            }
        }
    }
};

class Pool {
public:
    static Pool &get() {
        static bool once = [] {
#ifndef _WIN32
            // A forked child (the process executor) has none of the threads and maybe a lock taken: a new, empty pool
            pthread_atfork(nullptr, nullptr, [] { instance() = new Pool(); });
#endif
            return true;
        }();
        (void)once;
        return *instance();
    }

    // Offer `tickets` workers to the job; returns at once
    void offer(Job *job, int tickets) {
        {
            std::lock_guard<std::mutex> g(lock_);
            grow(tickets);
            for (int t = 0; t < tickets; t++) queue_.push_back(job);
        }
        if (tickets == 1) ready_.notify_one();
        else ready_.notify_all();
    }

    // Take back the job's tickets no worker picked up; afterwards no new worker joins it
    void withdraw(Job *job) {
        std::lock_guard<std::mutex> g(lock_);
        queue_.erase(std::remove(queue_.begin(), queue_.end(), job), queue_.end());
    }

private:
    static Pool *&instance() {
        static Pool *pool = new Pool(); // never destroyed: its threads live until the process ends
        return pool;
    }

    void grow(int want) { // under lock_
        const int have = static_cast<int>(workers_);
        for (int t = have; t < want; t++) {
            std::thread(&Pool::work, this).detach();
            workers_++;
        }
    }

    void work() {
        for (;;) {
            Job *job;
            {
                std::unique_lock<std::mutex> g(lock_);
                ready_.wait(g, [&] { return !queue_.empty(); });
                job = queue_.front();
                queue_.pop_front();
                std::lock_guard<std::mutex> j(job->lock);
                job->running++; // before lock_ goes: withdraw() can't miss this worker
            }
            job->run();
            std::lock_guard<std::mutex> j(job->lock);
            if (--job->running == 0) job->done.notify_all();
        }
    }

    std::mutex lock_;
    std::condition_variable ready_;
    std::deque<Job *> queue_;
    std::size_t workers_ = 0;
};

} // namespace detail

// fn(i) for every i in [0, n), items handed out one at a time. The first exception is rethrown here.
template <typename F>
void parallel_for(long long n, int threads, F fn) {
    const int T = thread_count(n, threads);
    if (T <= 1) {
        for (long long i = 0; i < n; i++) fn(i);
        return;
    }
    const std::function<void(long long)> f = fn;
    detail::Job job;
    job.fn = &f;
    job.n = n;
    auto &pool = detail::Pool::get();
    pool.offer(&job, T - 1);
    job.run();
    pool.withdraw(&job);
    {
        std::unique_lock<std::mutex> g(job.lock);
        job.done.wait(g, [&] { return job.running == 0; });
    }
    if (job.err) std::rethrow_exception(job.err);
}

} // namespace rpps
