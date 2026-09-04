#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Reserve ALL resident inputs before selecting any victims. Installation is
// deferred until the complete request has been resolved, so a victim's old
// expert_to_slot mapping remains visible during reservation. Without this
// first pass, a later hit can alias a slot already promised to an earlier miss.
inline void flash_moe_protect_request_slots(
        const std::vector<int32_t> & experts,
        const std::vector<int32_t> & expert_to_slot,
        std::vector<uint32_t> & slot_reserved_epoch,
        uint32_t epoch) {
    // Validate the entire request before changing reservations.
    for (const int32_t expert : experts) {
        if (expert < 0 || std::size_t(expert) >= expert_to_slot.size()) {
            throw std::out_of_range("Flash-MoE expert id " + std::to_string(expert) + " is out of range");
        }
        const int32_t slot = expert_to_slot[expert];
        if (slot < -1 || (slot >= 0 && std::size_t(slot) >= slot_reserved_epoch.size())) {
            throw std::logic_error("Flash-MoE invalid expert-to-slot mapping");
        }
    }
    for (const int32_t expert : experts) {
        const int32_t slot = expert_to_slot[expert];
        if (slot >= 0) {
            slot_reserved_epoch[slot] = epoch;
        }
    }
}

inline int32_t flash_moe_select_slot(
        const std::vector<int32_t> & slot_to_expert,
        const std::vector<uint64_t> & slot_age,
        const std::vector<uint32_t> & slot_reserved_epoch,
        uint32_t epoch) {
    for (std::size_t slot = 0; slot < slot_to_expert.size(); ++slot) {
        if (slot_reserved_epoch[slot] != epoch && slot_to_expert[slot] < 0) {
            return int32_t(slot);
        }
    }
    int32_t victim = -1;
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    for (std::size_t slot = 0; slot < slot_to_expert.size(); ++slot) {
        if (slot_reserved_epoch[slot] != epoch && slot_age[slot] < oldest) {
            oldest = slot_age[slot];
            victim = int32_t(slot);
        }
    }
    return victim;
}
