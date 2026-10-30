#include "hnswdb/metadata.h"
#include "hnswdb/types.h"
#include <algorithm>
#include <cstring>
#include <mutex>

namespace hnswdb {

namespace {

bool compare_values(const MetaValue& actual, FilterOp op, const MetaValue& target) {
    auto get_num = [](const MetaValue& v, double& out) -> bool {
        if (std::holds_alternative<int64_t>(v)) {
            out = static_cast<double>(std::get<int64_t>(v));
            return true;
        }
        if (std::holds_alternative<double>(v)) {
            out = std::get<double>(v);
            return true;
        }
        return false;
    };

    double num_a, num_b;
    bool a_is_num = get_num(actual, num_a);
    bool b_is_num = get_num(target, num_b);

    if (a_is_num && b_is_num) {
        switch (op) {
            case FilterOp::EQ: return num_a == num_b;
            case FilterOp::NE: return num_a != num_b;
            case FilterOp::GT: return num_a > num_b;
            case FilterOp::GTE: return num_a >= num_b;
            case FilterOp::LT: return num_a < num_b;
            case FilterOp::LTE: return num_a <= num_b;
            default: return false;
        }
    }

    if (std::holds_alternative<std::string>(actual) && std::holds_alternative<std::string>(target)) {
        const auto& sa = std::get<std::string>(actual);
        const auto& sb = std::get<std::string>(target);
        switch (op) {
            case FilterOp::EQ: return sa == sb;
            case FilterOp::NE: return sa != sb;
            case FilterOp::GT: return sa > sb;
            case FilterOp::GTE: return sa >= sb;
            case FilterOp::LT: return sa < sb;
            case FilterOp::LTE: return sa <= sb;
            default: return false;
        }
    }

    if (std::holds_alternative<bool>(actual) && std::holds_alternative<bool>(target)) {
        bool ba = std::get<bool>(actual);
        bool bb = std::get<bool>(target);
        switch (op) {
            case FilterOp::EQ: return ba == bb;
            case FilterOp::NE: return ba != bb;
            default: return false;
        }
    }

    if (op == FilterOp::NE) return true;
    return false;
}

} // namespace

MetadataStore::MetadataStore(MetadataStore&& other) noexcept {
    *this = std::move(other);
}

MetadataStore& MetadataStore::operator=(MetadataStore&& other) noexcept {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        std::unique_lock<std::shared_mutex> other_lock(other.mutex_);
        store_ = std::move(other.store_);
        string_index_ = std::move(other.string_index_);
        int_index_ = std::move(other.int_index_);
    }
    return *this;
}

void MetadataStore::add_metadata(label_t label, const MetaMap& metadata) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto it = store_.find(label);
    if (it != store_.end()) {
        for (const auto& [k, v] : it->second) {
            if (std::holds_alternative<std::string>(v)) {
                auto& vec = string_index_[k][std::get<std::string>(v)];
                vec.erase(std::remove(vec.begin(), vec.end(), label), vec.end());
            } else if (std::holds_alternative<int64_t>(v)) {
                auto& vec = int_index_[k][std::get<int64_t>(v)];
                vec.erase(std::remove(vec.begin(), vec.end(), label), vec.end());
            }
        }
    }

    store_[label] = metadata;

    for (const auto& [k, v] : metadata) {
        if (std::holds_alternative<std::string>(v)) {
            string_index_[k][std::get<std::string>(v)].push_back(label);
        } else if (std::holds_alternative<int64_t>(v)) {
            int_index_[k][std::get<int64_t>(v)].push_back(label);
        }
    }
}

MetaMap MetadataStore::get_metadata(label_t label) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = store_.find(label);
    if (it == store_.end()) {
        return {};
    }
    return it->second;
}

bool MetadataStore::has_metadata(label_t label) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return store_.find(label) != store_.end();
}

void MetadataStore::remove_metadata(label_t label) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto it = store_.find(label);
    if (it != store_.end()) {
        for (const auto& [k, v] : it->second) {
            if (std::holds_alternative<std::string>(v)) {
                auto& vec = string_index_[k][std::get<std::string>(v)];
                vec.erase(std::remove(vec.begin(), vec.end(), label), vec.end());
            } else if (std::holds_alternative<int64_t>(v)) {
                auto& vec = int_index_[k][std::get<int64_t>(v)];
                vec.erase(std::remove(vec.begin(), vec.end(), label), vec.end());
            }
        }
        store_.erase(it);
    }
}

void MetadataStore::clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    store_.clear();
    string_index_.clear();
    int_index_.clear();
}

bool MetadataStore::evaluate_node(const MetaMap& meta, const FilterExpr& expr) const {
    switch (expr.op) {
        case FilterOp::EQ:
        case FilterOp::NE:
        case FilterOp::GT:
        case FilterOp::GTE:
        case FilterOp::LT:
        case FilterOp::LTE: {
            auto it = meta.find(expr.field);
            if (it == meta.end()) {
                return (expr.op == FilterOp::NE);
            }
            return compare_values(it->second, expr.op, expr.value);
        }
        case FilterOp::IN: {
            auto it = meta.find(expr.field);
            if (it == meta.end()) return false;
            for (const auto& candidate : expr.in_values) {
                if (compare_values(it->second, FilterOp::EQ, candidate)) {
                    return true;
                }
            }
            return false;
        }
        case FilterOp::AND: {
            for (const auto& child : expr.children) {
                if (!evaluate_node(meta, child)) return false;
            }
            return true;
        }
        case FilterOp::OR: {
            if (expr.children.empty()) return true;
            for (const auto& child : expr.children) {
                if (evaluate_node(meta, child)) return true;
            }
            return false;
        }
        case FilterOp::NOT: {
            if (expr.children.empty()) return true;
            return !evaluate_node(meta, expr.children[0]);
        }
    }
    return false;
}

bool MetadataStore::evaluate(label_t label, const FilterExpr& expr) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = store_.find(label);
    if (it == store_.end()) {
        MetaMap empty_meta;
        return evaluate_node(empty_meta, expr);
    }
    return evaluate_node(it->second, expr);
}

bool MetadataStore::can_use_inverted_index(const FilterExpr& expr) const {
    if (expr.op == FilterOp::EQ) {
        return std::holds_alternative<std::string>(expr.value) || std::holds_alternative<int64_t>(expr.value);
    }
    if (expr.op == FilterOp::IN) {
        return !expr.in_values.empty() &&
               (std::holds_alternative<std::string>(expr.in_values[0]) || std::holds_alternative<int64_t>(expr.in_values[0]));
    }
    return false;
}

std::vector<label_t> MetadataStore::get_matching_labels_from_index(const FilterExpr& expr) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::vector<label_t> result;

    if (expr.op == FilterOp::EQ) {
        if (std::holds_alternative<std::string>(expr.value)) {
            const auto& str_val = std::get<std::string>(expr.value);
            auto f_it = string_index_.find(expr.field);
            if (f_it != string_index_.end()) {
                auto v_it = f_it->second.find(str_val);
                if (v_it != f_it->second.end()) {
                    result = v_it->second;
                }
            }
        } else if (std::holds_alternative<int64_t>(expr.value)) {
            int64_t int_val = std::get<int64_t>(expr.value);
            auto f_it = int_index_.find(expr.field);
            if (f_it != int_index_.end()) {
                auto v_it = f_it->second.find(int_val);
                if (v_it != f_it->second.end()) {
                    result = v_it->second;
                }
            }
        }
    } else if (expr.op == FilterOp::IN) {
        for (const auto& val : expr.in_values) {
            if (std::holds_alternative<std::string>(val)) {
                const auto& str_val = std::get<std::string>(val);
                auto f_it = string_index_.find(expr.field);
                if (f_it != string_index_.end()) {
                    auto v_it = f_it->second.find(str_val);
                    if (v_it != f_it->second.end()) {
                        result.insert(result.end(), v_it->second.begin(), v_it->second.end());
                    }
                }
            } else if (std::holds_alternative<int64_t>(val)) {
                int64_t int_val = std::get<int64_t>(val);
                auto f_it = int_index_.find(expr.field);
                if (f_it != int_index_.end()) {
                    auto v_it = f_it->second.find(int_val);
                    if (v_it != f_it->second.end()) {
                        result.insert(result.end(), v_it->second.begin(), v_it->second.end());
                    }
                }
            }
        }
    }

    return result;
}

template <typename T>
static void append_raw_val(std::vector<uint8_t>& buffer, const T& val) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&val);
    buffer.insert(buffer.end(), p, p + sizeof(val));
}

static void append_str_val(std::vector<uint8_t>& buffer, const std::string& s) {
    uint32_t len = static_cast<uint32_t>(s.size());
    append_raw_val(buffer, len);
    if (len > 0) {
        buffer.insert(buffer.end(), s.begin(), s.end());
    }
}

void MetadataStore::serialize(std::vector<uint8_t>& buffer) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    uint64_t count = store_.size();
    append_raw_val(buffer, count);

    for (const auto& [label, meta] : store_) {
        append_raw_val(buffer, label);
        uint32_t num_keys = static_cast<uint32_t>(meta.size());
        append_raw_val(buffer, num_keys);

        for (const auto& [k, v] : meta) {
            append_str_val(buffer, k);
            uint8_t type_tag = static_cast<uint8_t>(v.index());
            append_raw_val(buffer, type_tag);

            if (type_tag == 0) { // int64_t
                int64_t val = std::get<int64_t>(v);
                append_raw_val(buffer, val);
            } else if (type_tag == 1) { // double
                double val = std::get<double>(v);
                append_raw_val(buffer, val);
            } else if (type_tag == 2) { // std::string
                append_str_val(buffer, std::get<std::string>(v));
            } else if (type_tag == 3) { // bool
                uint8_t val = std::get<bool>(v) ? 1 : 0;
                append_raw_val(buffer, val);
            }
        }
    }
}

template <typename T>
static void read_raw_val(const uint8_t* data, std::size_t length, std::size_t& offset, T& val) {
    if (offset + sizeof(T) > length) {
        throw CorruptFile("Corrupt metadata payload");
    }
    std::memcpy(&val, data + offset, sizeof(T));
    offset += sizeof(T);
}

static std::string read_str_val(const uint8_t* data, std::size_t length, std::size_t& offset) {
    if (offset + sizeof(uint32_t) > length) {
        throw CorruptFile("Corrupt metadata payload in string len");
    }
    uint32_t len = 0;
    std::memcpy(&len, data + offset, sizeof(len));
    offset += sizeof(len);
    if (offset + len > length) {
        throw CorruptFile("Corrupt metadata payload in string body");
    }
    std::string s(reinterpret_cast<const char*>(data + offset), len);
    offset += len;
    return s;
}

void MetadataStore::deserialize(const uint8_t* data, std::size_t length) {
    if (!data || length == 0) return;

    std::unique_lock<std::shared_mutex> lock(mutex_);
    store_.clear();
    string_index_.clear();
    int_index_.clear();

    std::size_t offset = 0;
    uint64_t count = 0;
    read_raw_val(data, length, offset, count);

    for (uint64_t i = 0; i < count; ++i) {
        label_t label = 0;
        read_raw_val(data, length, offset, label);
        uint32_t num_keys = 0;
        read_raw_val(data, length, offset, num_keys);

        MetaMap meta;
        for (uint32_t k = 0; k < num_keys; ++k) {
            std::string key = read_str_val(data, length, offset);
            uint8_t type_tag = 0;
            read_raw_val(data, length, offset, type_tag);

            if (type_tag == 0) {
                int64_t val = 0;
                read_raw_val(data, length, offset, val);
                meta[key] = val;
                int_index_[key][val].push_back(label);
            } else if (type_tag == 1) {
                double val = 0.0;
                read_raw_val(data, length, offset, val);
                meta[key] = val;
            } else if (type_tag == 2) {
                std::string val = read_str_val(data, length, offset);
                meta[key] = val;
                string_index_[key][val].push_back(label);
            } else if (type_tag == 3) {
                uint8_t b = 0;
                read_raw_val(data, length, offset, b);
                meta[key] = (b != 0);
            }
        }
        store_[label] = std::move(meta);
    }
}

} // namespace hnswdb
