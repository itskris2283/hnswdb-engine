#pragma once
#include <vector>
#include <cstdint>
#include <mutex>
#include <memory>
#include <algorithm>

namespace hnswdb {

class VisitedList {
public:
    explicit VisitedList(std::size_t num_elements)
        : visited_(num_elements, 0), current_epoch_(1) {}

    void reset() {
        ++current_epoch_;
        if (current_epoch_ == 0) { // Wraparound of uint16_t
            std::fill(visited_.begin(), visited_.end(), static_cast<uint16_t>(0));
            current_epoch_ = 1;
        }
    }

    bool mark_visited(std::size_t idx) {
        if (idx >= visited_.size()) return false;
        if (visited_[idx] == current_epoch_) return false;
        visited_[idx] = current_epoch_;
        return true;
    }

    bool is_visited(std::size_t idx) const {
        if (idx >= visited_.size()) return false;
        return visited_[idx] == current_epoch_;
    }

    void resize(std::size_t new_size) {
        visited_.resize(new_size, 0);
    }

private:
    std::vector<uint16_t> visited_;
    uint16_t current_epoch_{1};
};

class VisitedListPool {
public:
    VisitedListPool(std::size_t init_pools, std::size_t num_elements)
        : num_elements_(num_elements) {
        for (std::size_t i = 0; i < init_pools; ++i) {
            pool_.push_back(std::make_unique<VisitedList>(num_elements_));
        }
    }

    std::unique_ptr<VisitedList> acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pool_.empty()) {
            return std::make_unique<VisitedList>(num_elements_);
        }
        auto list = std::move(pool_.back());
        pool_.pop_back();
        list->reset();
        return list;
    }

    void release(std::unique_ptr<VisitedList> list) {
        if (!list) return;
        std::lock_guard<std::mutex> lock(mutex_);
        pool_.push_back(std::move(list));
    }

    void resize(std::size_t new_num_elements) {
        std::lock_guard<std::mutex> lock(mutex_);
        num_elements_ = new_num_elements;
        for (auto& list : pool_) {
            list->resize(new_num_elements);
        }
    }

private:
    std::vector<std::unique_ptr<VisitedList>> pool_;
    std::mutex mutex_;
    std::size_t num_elements_;
};

// RAII guard for VisitedList
class VisitedListGuard {
public:
    VisitedListGuard(VisitedListPool& pool)
        : pool_(pool), list_(pool.acquire()) {}

    ~VisitedListGuard() {
        pool_.release(std::move(list_));
    }

    VisitedList* get() { return list_.get(); }
    VisitedList* operator->() { return list_.get(); }

private:
    VisitedListPool& pool_;
    std::unique_ptr<VisitedList> list_;
};

} // namespace hnswdb
