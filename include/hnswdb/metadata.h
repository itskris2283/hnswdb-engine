#pragma once
#include "types.h"
#include <variant>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstdint>
#include <memory>
#include <shared_mutex>

namespace hnswdb {

using MetaValue = std::variant<int64_t, double, std::string, bool>;
using MetaMap = std::unordered_map<std::string, MetaValue>;

enum class FilterOp {
    EQ,
    NE,
    GT,
    GTE,
    LT,
    LTE,
    IN,
    AND,
    OR,
    NOT
};

struct FilterExpr {
    FilterOp op;
    std::string field;
    MetaValue value;
    std::vector<MetaValue> in_values;
    std::vector<FilterExpr> children;

    static FilterExpr Eq(const std::string& field, MetaValue val) {
        FilterExpr e;
        e.op = FilterOp::EQ;
        e.field = field;
        e.value = std::move(val);
        return e;
    }

    static FilterExpr Ne(const std::string& field, MetaValue val) {
        FilterExpr e;
        e.op = FilterOp::NE;
        e.field = field;
        e.value = std::move(val);
        return e;
    }

    static FilterExpr Gt(const std::string& field, MetaValue val) {
        FilterExpr e;
        e.op = FilterOp::GT;
        e.field = field;
        e.value = std::move(val);
        return e;
    }

    static FilterExpr Gte(const std::string& field, MetaValue val) {
        FilterExpr e;
        e.op = FilterOp::GTE;
        e.field = field;
        e.value = std::move(val);
        return e;
    }

    static FilterExpr Lt(const std::string& field, MetaValue val) {
        FilterExpr e;
        e.op = FilterOp::LT;
        e.field = field;
        e.value = std::move(val);
        return e;
    }

    static FilterExpr Lte(const std::string& field, MetaValue val) {
        FilterExpr e;
        e.op = FilterOp::LTE;
        e.field = field;
        e.value = std::move(val);
        return e;
    }

    static FilterExpr In(const std::string& field, const std::vector<MetaValue>& vals) {
        FilterExpr e;
        e.op = FilterOp::IN;
        e.field = field;
        e.in_values = vals;
        return e;
    }

    static FilterExpr And(const std::vector<FilterExpr>& exprs) {
        FilterExpr e;
        e.op = FilterOp::AND;
        e.children = exprs;
        return e;
    }

    static FilterExpr Or(const std::vector<FilterExpr>& exprs) {
        FilterExpr e;
        e.op = FilterOp::OR;
        e.children = exprs;
        return e;
    }

    static FilterExpr Not(const FilterExpr& expr) {
        FilterExpr e;
        e.op = FilterOp::NOT;
        e.children = {expr};
        return e;
    }
};

class MetadataStore {
public:
    MetadataStore() = default;
    ~MetadataStore() = default;

    MetadataStore(const MetadataStore&) = delete;
    MetadataStore& operator=(const MetadataStore&) = delete;

    MetadataStore(MetadataStore&& other) noexcept;
    MetadataStore& operator=(MetadataStore&& other) noexcept;

    void add_metadata(label_t label, const MetaMap& metadata);
    MetaMap get_metadata(label_t label) const;
    bool has_metadata(label_t label) const;
    void remove_metadata(label_t label);
    void clear();

    bool evaluate(label_t label, const FilterExpr& expr) const;

    // Fast inverted index lookup for exact/in matching
    bool can_use_inverted_index(const FilterExpr& expr) const;
    std::vector<label_t> get_matching_labels_from_index(const FilterExpr& expr) const;

    // Serialization
    void serialize(std::vector<uint8_t>& buffer) const;
    void deserialize(const uint8_t* data, std::size_t length);

private:
    bool evaluate_node(const MetaMap& meta, const FilterExpr& expr) const;

    std::unordered_map<label_t, MetaMap> store_;

    // Inverted index for string & int fields: field -> value -> sorted labels
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<label_t>>> string_index_;
    std::unordered_map<std::string, std::unordered_map<int64_t, std::vector<label_t>>> int_index_;

    mutable std::shared_mutex mutex_;
};

} // namespace hnswdb
