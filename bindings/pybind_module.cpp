#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "hnswdb/hnsw_index.h"
#include "hnswdb/flat_index.h"
#include "hnswdb/metadata.h"
#include "hnswdb/distance.h"
#include "hnswdb/types.h"
#include <string>
#include <vector>
#include <memory>
#include <iostream>

namespace py = pybind11;
using namespace hnswdb;

namespace {

MetaValue py_to_meta_value(const py::handle& obj) {
    if (py::isinstance<py::bool_>(obj)) {
        return obj.cast<bool>();
    }
    if (py::isinstance<py::int_>(obj)) {
        return obj.cast<int64_t>();
    }
    if (py::isinstance<py::float_>(obj)) {
        return obj.cast<double>();
    }
    if (py::isinstance<py::str>(obj)) {
        return obj.cast<std::string>();
    }
    throw InvalidArgument("Unsupported metadata value type: " + py::str(obj.get_type()).cast<std::string>());
}

py::object meta_value_to_py(const MetaValue& val) {
    if (std::holds_alternative<int64_t>(val)) {
        return py::cast(std::get<int64_t>(val));
    }
    if (std::holds_alternative<double>(val)) {
        return py::cast(std::get<double>(val));
    }
    if (std::holds_alternative<std::string>(val)) {
        return py::cast(std::get<std::string>(val));
    }
    if (std::holds_alternative<bool>(val)) {
        return py::cast(std::get<bool>(val));
    }
    return py::none();
}

MetaMap py_dict_to_metamap(const py::dict& d) {
    MetaMap map;
    for (auto item : d) {
        std::string k = item.first.cast<std::string>();
        map[k] = py_to_meta_value(item.second);
    }
    return map;
}

py::dict metamap_to_py_dict(const MetaMap& map) {
    py::dict d;
    for (const auto& [k, v] : map) {
        d[py::str(k)] = meta_value_to_py(v);
    }
    return d;
}

FilterExpr parse_filter_dict(const py::dict& d);

FilterExpr parse_single_field_filter(const std::string& field, const py::handle& val) {
    if (py::isinstance<py::dict>(val)) {
        py::dict op_dict = val.cast<py::dict>();
        std::vector<FilterExpr> field_exprs;
        for (auto item : op_dict) {
            std::string op = item.first.cast<std::string>();
            if (op == "$eq") {
                field_exprs.push_back(FilterExpr::Eq(field, py_to_meta_value(item.second)));
            } else if (op == "$ne") {
                field_exprs.push_back(FilterExpr::Ne(field, py_to_meta_value(item.second)));
            } else if (op == "$gt") {
                field_exprs.push_back(FilterExpr::Gt(field, py_to_meta_value(item.second)));
            } else if (op == "$gte") {
                field_exprs.push_back(FilterExpr::Gte(field, py_to_meta_value(item.second)));
            } else if (op == "$lt") {
                field_exprs.push_back(FilterExpr::Lt(field, py_to_meta_value(item.second)));
            } else if (op == "$lte") {
                field_exprs.push_back(FilterExpr::Lte(field, py_to_meta_value(item.second)));
            } else if (op == "$in") {
                py::list in_list = item.second.cast<py::list>();
                std::vector<MetaValue> in_vals;
                for (auto elem : in_list) {
                    in_vals.push_back(py_to_meta_value(elem));
                }
                field_exprs.push_back(FilterExpr::In(field, in_vals));
            } else {
                throw InvalidArgument("Unknown filter operator: " + op);
            }
        }
        if (field_exprs.size() == 1) return field_exprs[0];
        return FilterExpr::And(field_exprs);
    }
    // Direct equality: {'category': 'shoes'}
    return FilterExpr::Eq(field, py_to_meta_value(val));
}

FilterExpr parse_filter_dict(const py::dict& d) {
    std::vector<FilterExpr> and_children;

    for (auto item : d) {
        std::string key = item.first.cast<std::string>();

        if (key == "$and") {
            py::list list_and = item.second.cast<py::list>();
            std::vector<FilterExpr> children;
            for (auto elem : list_and) {
                children.push_back(parse_filter_dict(elem.cast<py::dict>()));
            }
            and_children.push_back(FilterExpr::And(children));
        } else if (key == "$or") {
            py::list list_or = item.second.cast<py::list>();
            std::vector<FilterExpr> children;
            for (auto elem : list_or) {
                children.push_back(parse_filter_dict(elem.cast<py::dict>()));
            }
            and_children.push_back(FilterExpr::Or(children));
        } else if (key == "$not") {
            FilterExpr child = parse_filter_dict(item.second.cast<py::dict>());
            and_children.push_back(FilterExpr::Not(child));
        } else {
            and_children.push_back(parse_single_field_filter(key, item.second));
        }
    }

    if (and_children.empty()) {
        return FilterExpr::And({});
    }
    if (and_children.size() == 1) {
        return and_children[0];
    }
    return FilterExpr::And(and_children);
}

// Convert SearchResult or vector<SearchResult> to Python tuple (ids, dists)
py::tuple results_to_numpy(const std::vector<SearchResult>& results, std::size_t n_queries, std::size_t k) {
    auto ids_arr = py::array_t<int64_t>({n_queries, k});
    auto dists_arr = py::array_t<float>({n_queries, k});

    auto ids_buf = ids_arr.mutable_unchecked<2>();
    auto dists_buf = dists_arr.mutable_unchecked<2>();

    for (std::size_t i = 0; i < n_queries; ++i) {
        for (std::size_t j = 0; j < k; ++j) {
            ids_buf(i, j) = (j < results[i].ids.size()) ? results[i].ids[j] : -1;
            dists_buf(i, j) = (j < results[i].dists.size()) ? results[i].dists[j] : std::numeric_limits<float>::infinity();
        }
    }

    return py::make_tuple(ids_arr, dists_arr);
}

} // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "HNSWDB C++ Core Engine";
    m.attr("__version__") = "0.1.0";

    // Exception Hierarchy
    auto pyHnswdbError = py::register_exception<HnswdbError>(m, "HnswdbError", PyExc_RuntimeError);
    py::register_exception<DimensionMismatch>(m, "DimensionMismatch", pyHnswdbError);
    py::register_exception<InvalidVector>(m, "InvalidVector", pyHnswdbError);
    py::register_exception<CapacityExceeded>(m, "CapacityExceeded", pyHnswdbError);
    py::register_exception<CorruptFile>(m, "CorruptFile", pyHnswdbError);
    py::register_exception<UnsupportedVersion>(m, "UnsupportedVersion", pyHnswdbError);
    py::register_exception<InvalidArgument>(m, "InvalidArgument", pyHnswdbError);




    // Distance utilities
    m.def("active_kernel_name", &distance::active_kernel_name, "Get name of currently active SIMD kernel ('avx2' or 'scalar')");
    m.def("set_force_scalar", &distance::set_force_scalar, py::arg("force"), "Force the scalar distance kernel");

    // HNSWIndex Binding
    py::class_<HNSWIndex>(m, "HNSWIndex")
        .def(py::init([](std::size_t dim,
                          const std::string& metric_str,
                          std::size_t max_elements,
                          std::size_t M,
                          std::size_t ef_construction,
                          std::size_t ef_search,
                          uint64_t seed) {
            Metric metric = string_to_metric(metric_str);
            return std::make_unique<HNSWIndex>(dim, metric, max_elements, M, ef_construction, ef_search, seed);
        }),
        py::arg("dim"),
        py::arg("metric") = "l2",
        py::arg("max_elements") = 1000000,
        py::arg("M") = 16,
        py::arg("ef_construction") = 200,
        py::arg("ef_search") = 50,
        py::arg("seed") = 42)

        .def("add", [](HNSWIndex& self,
                       py::array_t<float, py::array::c_style | py::array::forcecast> vectors,
                       py::array_t<int64_t, py::array::c_style | py::array::forcecast> ids,
                       py::object metadata,
                       std::size_t num_threads) {
            py::buffer_info v_buf = vectors.request();
            py::buffer_info id_buf = ids.request();

            std::size_t n = 0;
            std::size_t d = 0;

            if (v_buf.ndim == 1) {
                n = 1;
                d = v_buf.shape[0];
            } else if (v_buf.ndim == 2) {
                n = v_buf.shape[0];
                d = v_buf.shape[1];
            } else {
                throw InvalidArgument("vectors array must be 1-D or 2-D");
            }

            if (d != self.dim()) {
                throw DimensionMismatch("Vector dimension " + std::to_string(d) + " does not match index dimension " + std::to_string(self.dim()));
            }

            if (static_cast<std::size_t>(id_buf.size) != n) {
                throw InvalidArgument("ids length " + std::to_string(id_buf.size) + " does not match vectors length " + std::to_string(n));
            }

            const float* v_ptr = static_cast<const float*>(v_buf.ptr);
            const int64_t* id_ptr = static_cast<const int64_t*>(id_buf.ptr);

            // Parse metadata if present
            std::unique_ptr<std::vector<MetaMap>> meta_vec;
            if (!metadata.is_none()) {
                meta_vec = std::make_unique<std::vector<MetaMap>>();
                meta_vec->reserve(n);
                if (py::isinstance<py::list>(metadata)) {
                    py::list meta_list = metadata.cast<py::list>();
                    for (auto item : meta_list) {
                        meta_vec->push_back(py_dict_to_metamap(item.cast<py::dict>()));
                    }
                } else if (py::isinstance<py::dict>(metadata) && n == 1) {
                    meta_vec->push_back(py_dict_to_metamap(metadata.cast<py::dict>()));
                }
            }

            // Release GIL for batch insertion
            py::gil_scoped_release release;
            self.add_batch(v_ptr, id_ptr, n, meta_vec.get(), num_threads);
        },
        py::arg("vectors"),
        py::arg("ids"),
        py::arg("metadata") = py::none(),
        py::arg("num_threads") = 0)

        .def("search", [](const HNSWIndex& self,
                          py::array_t<float, py::array::c_style | py::array::forcecast> queries,
                          std::size_t k,
                          py::object ef_search,
                          py::object filter,
                          std::size_t num_threads) {
            py::buffer_info q_buf = queries.request();
            std::size_t n_queries = 0;
            std::size_t d = 0;

            if (q_buf.ndim == 1) {
                n_queries = 1;
                d = q_buf.shape[0];
            } else if (q_buf.ndim == 2) {
                n_queries = q_buf.shape[0];
                d = q_buf.shape[1];
            } else {
                throw InvalidArgument("queries array must be 1-D or 2-D");
            }

            if (d != self.dim()) {
                throw DimensionMismatch("Query dimension " + std::to_string(d) + " does not match index dimension " + std::to_string(self.dim()));
            }

            std::size_t ef_override = ef_search.is_none() ? 0 : ef_search.cast<std::size_t>();

            std::unique_ptr<FilterExpr> filter_expr;
            if (!filter.is_none()) {
                filter_expr = std::make_unique<FilterExpr>(parse_filter_dict(filter.cast<py::dict>()));
            }

            const float* q_ptr = static_cast<const float*>(q_buf.ptr);
            std::vector<SearchResult> results;

            {
                py::gil_scoped_release release;
                results = self.search_batch(q_ptr, n_queries, k, ef_override, filter_expr.get(), num_threads);
            }

            return results_to_numpy(results, n_queries, k);
        },
        py::arg("queries"),
        py::arg("k") = 10,
        py::arg("ef_search") = py::none(),
        py::arg("filter") = py::none(),
        py::arg("num_threads") = 0)

        .def("delete", [](HNSWIndex& self, py::object ids) -> std::size_t {
            std::vector<label_t> id_list;
            if (py::isinstance<py::array>(ids)) {
                auto arr = ids.cast<py::array_t<int64_t, py::array::c_style | py::array::forcecast>>();
                auto buf = arr.request();
                const int64_t* ptr = static_cast<const int64_t*>(buf.ptr);
                id_list.assign(ptr, ptr + buf.size);
            } else if (py::isinstance<py::list>(ids)) {
                id_list = ids.cast<std::vector<label_t>>();
            } else if (py::isinstance<py::int_>(ids)) {
                id_list.push_back(ids.cast<label_t>());
            } else {
                throw InvalidArgument("ids must be a numpy array, list of ints, or single int");
            }
            return self.delete_labels(id_list);
        }, py::arg("ids"))

        .def("save", [](const HNSWIndex& self, const std::string& path) {
            py::gil_scoped_release release;
            self.save(path);
        }, py::arg("path"))

        .def_static("load", [](const std::string& path) {
            py::gil_scoped_release release;
            return HNSWIndex::load(path);
        }, py::arg("path"))

        .def("set_ef_search", &HNSWIndex::set_ef_search, py::arg("ef"))
        .def("get_ef_search", &HNSWIndex::get_ef_search)
        .def("resize", &HNSWIndex::resize, py::arg("new_max_elements"))

        .def("compact", [](const HNSWIndex& self) {
            py::gil_scoped_release release;
            return self.compact();
        })

        .def("stats", [](const HNSWIndex& self) {
            auto st = self.stats();
            py::dict d;
            for (const auto& [k, v] : st) {
                if (k == "needs_compaction") {
                    d[py::str(k)] = (v != 0.0);
                } else if (k == "size" || k == "count" || k == "capacity" || k == "total_elements" ||
                           k == "deleted" || k == "tombstones" || k == "entry_point" || k == "dim" ||
                           k == "M" || k == "ef_construction" || k == "ef_search" ||
                           k == "max_level" || k == "memory_bytes") {
                    d[py::str(k)] = static_cast<int64_t>(v);
                } else {
                    d[py::str(k)] = v;
                }
            }
            d[py::str("metric")] = metric_to_string(self.metric());
            d[py::str("kernel")] = distance::active_kernel_name();
            return d;
        })

        .def("diagnostics", [](const HNSWIndex& self) {
            auto diag_map = self.diagnostics();
            py::dict d;
            for (const auto& [k, v] : diag_map) {
                d[py::str(k)] = v;
            }
            return d;
        })

        .def("__len__", &HNSWIndex::size)
        .def("size", &HNSWIndex::size)
        .def("capacity", &HNSWIndex::capacity)
        .def("dim", &HNSWIndex::dim)
        .def("contains", &HNSWIndex::contains, py::arg("label"))

        .def("get_vector", [](const HNSWIndex& self, label_t label) {
            auto vec = self.get_vector(label);
            auto arr = py::array_t<float>(vec.size());
            std::memcpy(arr.mutable_data(), vec.data(), vec.size() * sizeof(float));
            return arr;
        }, py::arg("label"))

        .def("get_metadata", [](const HNSWIndex& self, label_t label) {
            return metamap_to_py_dict(self.get_metadata(label));
        }, py::arg("label"))

        .def("set_metadata", [](HNSWIndex& self, label_t label, const py::dict& metadata) {
            self.set_metadata(label, py_dict_to_metamap(metadata));
        }, py::arg("label"), py::arg("metadata"))

        .def("bench_search_latency", [](const HNSWIndex& self,
                                        py::array_t<float, py::array::c_style | py::array::forcecast> queries,
                                        std::size_t k,
                                        std::size_t ef) {
            py::buffer_info q_buf = queries.request();
            std::size_t n_queries = (q_buf.ndim == 1) ? 1 : q_buf.shape[0];
            const float* ptr = static_cast<const float*>(q_buf.ptr);
            py::gil_scoped_release release;
            return self.bench_search_latency(ptr, n_queries, k, ef);
        }, py::arg("queries"), py::arg("k") = 10, py::arg("ef") = 0);

    // FlatIndex Binding
    py::class_<FlatIndex>(m, "FlatIndex")
        .def(py::init([](std::size_t dim, const std::string& metric_str) {
            Metric metric = string_to_metric(metric_str);
            return std::make_unique<FlatIndex>(dim, metric);
        }),
        py::arg("dim"),
        py::arg("metric") = "l2")

        .def("add", [](FlatIndex& self,
                       py::array_t<float, py::array::c_style | py::array::forcecast> vectors,
                       py::array_t<int64_t, py::array::c_style | py::array::forcecast> ids,
                       py::object /*metadata*/,
                       std::size_t num_threads) {
            py::buffer_info v_buf = vectors.request();
            py::buffer_info id_buf = ids.request();

            std::size_t n = (v_buf.ndim == 1) ? 1 : v_buf.shape[0];
            std::size_t d = (v_buf.ndim == 1) ? v_buf.shape[0] : v_buf.shape[1];

            if (d != self.dim()) {
                throw DimensionMismatch("Vector dimension " + std::to_string(d) + " does not match index dimension " + std::to_string(self.dim()));
            }

            const float* v_ptr = static_cast<const float*>(v_buf.ptr);
            const int64_t* id_ptr = static_cast<const int64_t*>(id_buf.ptr);

            py::gil_scoped_release release;
            self.add_batch(v_ptr, id_ptr, n, num_threads);
        },
        py::arg("vectors"),
        py::arg("ids"),
        py::arg("metadata") = py::none(),
        py::arg("num_threads") = 0)

        .def("search", [](const FlatIndex& self,
                          py::array_t<float, py::array::c_style | py::array::forcecast> queries,
                          std::size_t k,
                          py::object /*ef_search*/,
                          py::object /*filter*/,
                          std::size_t num_threads) {
            py::buffer_info q_buf = queries.request();
            std::size_t n_queries = (q_buf.ndim == 1) ? 1 : q_buf.shape[0];
            std::size_t d = (q_buf.ndim == 1) ? q_buf.shape[0] : q_buf.shape[1];

            if (d != self.dim()) {
                throw DimensionMismatch("Query dimension " + std::to_string(d) + " does not match index dimension " + std::to_string(self.dim()));
            }

            const float* q_ptr = static_cast<const float*>(q_buf.ptr);
            std::vector<SearchResult> results;
            {
                py::gil_scoped_release release;
                results = self.search_batch(q_ptr, n_queries, k, num_threads);
            }
            return results_to_numpy(results, n_queries, k);
        },
        py::arg("queries"),
        py::arg("k") = 10,
        py::arg("ef_search") = py::none(),
        py::arg("filter") = py::none(),
        py::arg("num_threads") = 0)

        .def("delete", [](FlatIndex& self, py::object ids) -> std::size_t {
            std::vector<label_t> id_list;
            if (py::isinstance<py::array>(ids)) {
                auto arr = ids.cast<py::array_t<int64_t, py::array::c_style | py::array::forcecast>>();
                auto buf = arr.request();
                const int64_t* ptr = static_cast<const int64_t*>(buf.ptr);
                id_list.assign(ptr, ptr + buf.size);
            } else if (py::isinstance<py::list>(ids)) {
                id_list = ids.cast<std::vector<label_t>>();
            } else if (py::isinstance<py::int_>(ids)) {
                id_list.push_back(ids.cast<label_t>());
            }
            return self.delete_labels(id_list);
        }, py::arg("ids"))

        .def("save", [](const FlatIndex& self, const std::string& path) {
            py::gil_scoped_release release;
            self.save(path);
        }, py::arg("path"))

        .def_static("load", [](const std::string& path) {
            py::gil_scoped_release release;
            return FlatIndex::load(path);
        }, py::arg("path"))

        .def("__len__", &FlatIndex::size)
        .def("size", &FlatIndex::size)
        .def("dim", &FlatIndex::dim)
        .def("contains", &FlatIndex::contains, py::arg("label"))

        .def("get_vector", [](const FlatIndex& self, label_t label) {
            auto vec = self.get_vector(label);
            auto arr = py::array_t<float>(vec.size());
            std::memcpy(arr.mutable_data(), vec.data(), vec.size() * sizeof(float));
            return arr;
        }, py::arg("label"));
}
