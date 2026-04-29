#include <bits/stdc++.h>
using namespace std;

static inline uint32_t lcg_next(uint32_t &state) {
    constexpr uint32_t a = 1664525u;
    constexpr uint32_t c = 1013904223u;
    state = a * state + c; // overflow wraps modulo 2^32
    return state;
}

static inline long long max_subarray_sum(int n, uint32_t seed, int min_val, int max_val) {
    const int range = max_val - min_val + 1;
    long long best = LLONG_MIN;
    long long cur = 0;
    uint32_t state = seed;
    for (int i = 0; i < n; ++i) {
        uint32_t v = lcg_next(state);
        int x = int(v % (uint32_t)range) + min_val;
        cur = std::max<long long>(x, cur + x);
        best = std::max(best, cur);
    }
    return best;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    
    const int n = 10000;
    const uint32_t initial_seed = 42u;
    const int min_val = -10;
    const int max_val = 10;
    
    auto start = chrono::high_resolution_clock::now();
    
    long long total = 0;
    uint32_t state = initial_seed;
    for (int k = 0; k < 20; ++k) {
        uint32_t seed = lcg_next(state);
        total += max_subarray_sum(n, seed, min_val, max_val);
    }
    
    auto end = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(end - start).count();
    
    cout << "Total Maximum Subarray Sum (20 runs): " << total << "\n";
    cout << "Execution Time: " << fixed << setprecision(6) << elapsed << " seconds\n";
    return 0;
}