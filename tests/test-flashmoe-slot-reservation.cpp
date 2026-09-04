#include "../src/llama-flash-moe-slots.h"
#include "../vendor/nlohmann/json.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <map>
#include <numeric>
#include <random>

static void check(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

// Use the production protection/victim selection helpers, and the runtime's
// deferred install ordering. bank[] represents the expert bytes actually loaded.
struct test_bank {
    std::vector<int32_t> bank;
    std::vector<int32_t> expert_to_slot;
    std::vector<uint64_t> ages;
    std::vector<uint32_t> reserved;
    uint32_t epoch = 0;
    uint64_t age = 0;
    uint64_t misses = 0;

    explicit test_bank(int slots) : bank(slots, -1), expert_to_slot(256, -1), ages(slots), reserved(slots) {}

    std::vector<int32_t> run(const std::vector<int32_t> & experts, bool protect = true) {
        if (++epoch == 0) {
            std::fill(reserved.begin(), reserved.end(), 0);
            epoch = 1;
        }
        if (protect) {
            flash_moe_protect_request_slots(experts, expert_to_slot, reserved, epoch);
        }
        std::vector<int32_t> seen(256, -1);
        std::vector<int32_t> slots;
        std::vector<std::pair<int32_t, int32_t>> pending;
        std::vector<int32_t> touched;
        for (int32_t expert : experts) {
            int32_t slot = seen.at(expert);
            if (slot < 0) {
                slot = expert_to_slot.at(expert);
                if (slot < 0) {
                    slot = flash_moe_select_slot(bank, ages, reserved, epoch);
                    if (slot < 0) {
                        throw std::overflow_error("request exceeds slot capacity");
                    }
                    pending.emplace_back(expert, slot);
                }
                reserved[slot] = epoch;
                seen[expert] = slot;
                touched.push_back(slot);
            }
            slots.push_back(slot);
        }
        for (auto load : pending) {
            if (bank[load.second] >= 0) {
                expert_to_slot[bank[load.second]] = -1;
            }
        }
        for (auto load : pending) {
            bank[load.second] = load.first;
            expert_to_slot[load.first] = load.second;
        }
        for (int32_t slot : touched) {
            ages[slot] = ++age;
        }
        misses += pending.size();
        return slots;
    }

    void verify(const std::vector<int32_t> & experts) {
        const auto slots = run(experts);
        std::map<int32_t, int32_t> owners;
        for (size_t i = 0; i < experts.size(); ++i) {
            check(bank[slots[i]] == experts[i], "selected expert bytes overwritten by another selected expert");
            check(expert_to_slot[experts[i]] == slots[i], "expert-to-slot mapping inconsistent");
            auto inserted = owners.emplace(slots[i], experts[i]);
            check(inserted.second || inserted.first->second == experts[i], "distinct experts alias a slot");
        }
        for (size_t s = 0; s < bank.size(); ++s) {
            if (bank[s] >= 0) {
                check(expert_to_slot[bank[s]] == int32_t(s), "bank reverse mapping inconsistent");
            }
        }
    }
};

static void regression(int capacity) {
    test_bank original(capacity);
    std::vector<int32_t> initial(capacity);
    std::iota(initial.begin(), initial.end(), 0);
    original.verify(initial);
    test_bank fixed = original;
    const std::vector<int32_t> request = {255, 2, 3, 4, 5, 6, 7, 0};
    const auto old_slots = original.run(request, false);
    check(old_slots.front() == old_slots.back(), "fixture must reproduce the old deferred-install alias");
    check(original.bank[old_slots.back()] != request.back(), "fixture must use the wrong expert with old algorithm");
    fixed.verify(request);
    check(fixed.expert_to_slot[0] == 0, "oldest required resident must be protected");
    check(fixed.expert_to_slot[255] == 1, "oldest unrequested expert must be evicted");
    check(fixed.misses == uint64_t(capacity + 1), "protecting hits must not add redundant reads");
    fixed.verify({0, 0, 255, 255, 2, 2, 3, 3});
    fixed.epoch = UINT32_MAX;
    fixed.verify({254, 0, 255, 2, 3, 4, 5, 6});
}

static void invalid_requests() {
    test_bank bank(8);
    bank.verify({0, 1, 2, 3, 4, 5, 6, 7});
    const auto before = bank.bank;
    for (const auto & invalid : std::vector<std::vector<int32_t>>{{0, -1}, {0, 256}}) {
        const auto old_reservations = bank.reserved;
        bool rejected = false;
        try {
            bank.run(invalid);
        } catch (const std::out_of_range &) {
            rejected = true;
        }
        check(rejected && bank.bank == before && bank.reserved == old_reservations,
                "invalid IDs must be rejected before changing reservations or bank contents");
    }
    bool overflow = false;
    try {
        bank.run({0, 1, 2, 3, 4, 5, 6, 7, 8});
    } catch (const std::overflow_error &) {
        overflow = true;
    }
    check(overflow && bank.bank == before, "overflow must not install partial request");
    bank.verify({8, 2, 3, 4, 5, 6, 7, 0});
    bank.verify({});
}

static void randomized() {
    std::mt19937 rng(20260904);
    for (int capacity : {8, 16, 96, 256}) {
        test_bank bank(capacity);
        std::vector<int32_t> ids(256);
        std::iota(ids.begin(), ids.end(), 0);
        for (int step = 0; step < 10000; ++step) {
            std::shuffle(ids.begin(), ids.end(), rng);
            const size_t unique = step % 3 == 0 ? size_t(capacity) : 8;
            std::vector<int32_t> request(ids.begin(), ids.begin() + unique);
            // Batched routing: repeat selected experts without requiring extra slots.
            for (size_t i = 0; i < unique; ++i) {
                if ((rng() & 3) == 0) {
                    request.push_back(request[i]);
                }
            }
            std::shuffle(request.begin(), request.end(), rng);
            bank.verify(request);
        }
    }
}

static void replay_trace(const char * path) {
    std::ifstream input(path);
    check(input.is_open(), "cannot open trace");
    std::map<int32_t, test_bank> layers;
    std::string line;
    size_t calls = 0;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const auto record = nlohmann::json::parse(line);
        const int32_t layer = record.at("layer").get<int32_t>();
        auto it = layers.emplace(layer, test_bank(96)).first;
        it->second.verify(record.at("experts").get<std::vector<int32_t>>());
        ++calls;
    }
    check(calls > 0, "empty trace");
    printf("trace: %zu requests, zero expert-slot aliases\n", calls);
}

int main(int argc, char ** argv) {
    try {
        regression(8);
        regression(16);
        regression(96);
        invalid_requests();
        randomized();
        if (argc == 3 && std::string(argv[1]) == "--trace") {
            replay_trace(argv[2]);
        } else if (argc != 1) {
            throw std::runtime_error("usage: test-flashmoe-slot-reservation [--trace path]");
        }
        printf("PASS: slot reservation regression, duplicates, overflow, epoch wrap and 40000 randomized requests\n");
    } catch (const std::exception & ex) {
        fprintf(stderr, "FAIL: %s\n", ex.what());
        return 1;
    }
    return 0;
}
