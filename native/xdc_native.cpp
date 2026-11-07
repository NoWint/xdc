// XDC native hot-path helpers.
//
// The extension is optional: the Python implementation remains canonical.  These
// routines fuse per-parameter validation, robust filtering, token-weighted
// averaging and the outer momentum update into a single pass over contiguous
// float32 buffers, and keep a native deterministic token sampler for the
// coordinator's per-token loop.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace {

using f32 = py::array_t<float, py::array::c_style | py::array::forcecast>;

struct SplitMix64 {
  uint64_t state;
  explicit SplitMix64(uint64_t seed) : state(seed + 0x9E3779B97F4A7C15ULL) {}
  uint64_t next() {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

template <typename Fn>
void parallel_chunks(size_t n, Fn fn) {
  unsigned hw = std::thread::hardware_concurrency();
  size_t threads = std::min<size_t>(std::max(1u, hw), 8);
  if (n < 65536 || threads == 1) {
    fn(0, n);
    return;
  }
  std::vector<std::thread> workers;
  workers.reserve(threads - 1);
  size_t chunk = (n + threads - 1) / threads;
  for (size_t t = 1; t < threads; ++t) {
    size_t begin = t * chunk;
    size_t end = std::min(n, begin + chunk);
    if (begin >= end) break;
    workers.emplace_back([=]() { fn(begin, end); });
  }
  fn(0, std::min(n, chunk));
  for (auto &worker : workers) worker.join();
}

py::tuple aggregate(py::array base_arr, py::list updates_list, py::array weights_arr, py::object momentum_obj,
                    double outer_lr, double momentum, bool nesterov, double max_norm_ratio) {
  auto base = f32::ensure(base_arr);
  auto weights = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(weights_arr);
  if (!base || !weights) throw std::invalid_argument("base/weights must be array-like");
  size_t n = static_cast<size_t>(base.size());
  size_t k = updates_list.size();
  if (k == 0) throw std::invalid_argument("no updates");
  if (static_cast<size_t>(weights.size()) != k) throw std::invalid_argument("weights length mismatch");

  std::vector<f32> keepalive;
  keepalive.reserve(k);
  std::vector<const float *> updates;
  updates.reserve(k);
  for (size_t i = 0; i < k; ++i) {
    auto arr = f32::ensure(updates_list[i]);
    if (!arr) throw std::invalid_argument("update is not array-like");
    if (static_cast<size_t>(arr.size()) != n) throw std::invalid_argument("update length mismatch");
    keepalive.push_back(arr);
    updates.push_back(arr.data());
  }
  const double *w = weights.data();
  for (size_t i = 0; i < k; ++i) {
    if (!std::isfinite(w[i]) || w[i] < 0) throw std::invalid_argument("invalid weight");
  }

  std::vector<double> norms(k, 0.0);
  for (size_t u = 0; u < k; ++u) {
    double sum = 0.0;
    const float *p = updates[u];
    for (size_t i = 0; i < n; ++i) {
      double v = static_cast<double>(p[i]);
      sum += v * v;
    }
    norms[u] = std::sqrt(sum);
    if (!std::isfinite(norms[u]))
      throw std::invalid_argument("update contains non-finite or unbounded values");
  }

  std::vector<size_t> keep(k);
  std::iota(keep.begin(), keep.end(), 0);
  if (k >= 3) {
    std::vector<double> sorted = norms;
    std::sort(sorted.begin(), sorted.end());
    double median = sorted[sorted.size() / 2];
    double threshold = std::max(median * max_norm_ratio, 1e-6);
    std::vector<size_t> filtered;
    for (size_t i = 0; i < k; ++i) {
      if (std::isfinite(norms[i]) && norms[i] > 0 && norms[i] <= threshold) filtered.push_back(i);
    }
    if (!filtered.empty()) keep = std::move(filtered);
  }
  double total_w = 0.0;
  for (size_t i : keep) total_w += w[i];
  if (!(total_w > 0)) throw std::invalid_argument("total weight is zero");

  py::array_t<float> out(n);
  py::array_t<float> mom(n);
  float *out_ptr = out.mutable_data();
  float *mom_ptr = mom.mutable_data();
  const float *base_ptr = base.data();
  const float *prev_ptr = nullptr;
  if (!momentum_obj.is_none() && momentum > 0) {
    auto prev = f32::ensure(momentum_obj);
    if (!prev || static_cast<size_t>(prev.size()) != n) throw std::invalid_argument("momentum length mismatch");
    keepalive.push_back(prev);
    prev_ptr = prev.data();
  }

  std::vector<float> scale(k, 0.0);
  for (size_t i : keep) scale[i] = static_cast<float>(w[i] / total_w);

  {
    py::gil_scoped_release release;
    parallel_chunks(n, [&](size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) {
        double avg = 0.0;
        for (size_t u : keep) avg += static_cast<double>(updates[u][i]) * scale[u];
        double buf = 0.0;
        double step = avg;
        if (momentum > 0) {
          double prev = prev_ptr ? static_cast<double>(prev_ptr[i]) : 0.0;
          buf = momentum * prev + avg;
          step = nesterov ? (avg + momentum * buf) : buf;
        }
        out_ptr[i] = static_cast<float>(static_cast<double>(base_ptr[i]) + outer_lr * step);
        mom_ptr[i] = static_cast<float>(buf);
      }
    });
  }

  py::dict stats;
  stats["updates_received"] = k;
  stats["updates_used"] = keep.size();
  stats["total_weight"] = total_w;
  py::list dropped;
  for (size_t i = 0; i < k; ++i) {
    if (std::find(keep.begin(), keep.end(), i) == keep.end()) dropped.append(i);
  }
  stats["dropped_indices"] = dropped;
  py::list norm_list;
  for (double v : norms) norm_list.append(v);
  stats["norms"] = norm_list;
  py::list weight_list;
  for (size_t i = 0; i < k; ++i) weight_list.append(w[i]);
  stats["weights"] = weight_list;
  return py::make_tuple(out, mom, stats);
}


py::tuple aggregate_tensors(py::list bases, py::list update_lists, py::array weights_arr, py::object momentum_obj,
                            double outer_lr, double momentum, bool nesterov, double max_norm_ratio) {
  size_t tensors = bases.size();
  size_t k = update_lists.size();
  if (tensors == 0 || k == 0) throw std::invalid_argument("empty tensor lists");
  auto weights = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(weights_arr);
  if (!weights || static_cast<size_t>(weights.size()) != k) throw std::invalid_argument("weights mismatch");
  std::vector<f32> base_arrays;
  base_arrays.reserve(tensors);
  std::vector<size_t> sizes;
  sizes.reserve(tensors);
  for (size_t t = 0; t < tensors; ++t) {
    auto arr = f32::ensure(bases[t]);
    if (!arr) throw std::invalid_argument("bad base tensor");
    base_arrays.push_back(arr);
    sizes.push_back(static_cast<size_t>(arr.size()));
  }
  std::vector<std::vector<f32>> upd_arrays(k);
  for (size_t u = 0; u < k; ++u) {
    py::list row = update_lists[u].cast<py::list>();
    if (row.size() != tensors) throw std::invalid_argument("update tensor count mismatch");
    upd_arrays[u].reserve(tensors);
    for (size_t t = 0; t < tensors; ++t) {
      auto arr = f32::ensure(row[t]);
      if (!arr || static_cast<size_t>(arr.size()) != sizes[t]) throw std::invalid_argument("update tensor size mismatch");
      upd_arrays[u].push_back(arr);
    }
  }
  const double *w = weights.data();
  std::vector<double> norms(k, 0.0);
  for (size_t u = 0; u < k; ++u) {
    double sum = 0.0;
    for (size_t t = 0; t < tensors; ++t) {
      const float *p = upd_arrays[u][t].data();
      size_t n = sizes[t];
      for (size_t i = 0; i < n; ++i) {
        double v = static_cast<double>(p[i]);
        sum += v * v;
      }
    }
    norms[u] = std::sqrt(sum);
    if (!std::isfinite(norms[u]))
      throw std::invalid_argument("update contains non-finite or unbounded values");
  }
  std::vector<size_t> keep(k);
  std::iota(keep.begin(), keep.end(), 0);
  if (k >= 3) {
    std::vector<double> sorted = norms;
    std::sort(sorted.begin(), sorted.end());
    double median = sorted[sorted.size() / 2];
    double threshold = std::max(median * max_norm_ratio, 1e-6);
    std::vector<size_t> filtered;
    for (size_t i = 0; i < k; ++i)
      if (std::isfinite(norms[i]) && norms[i] > 0 && norms[i] <= threshold) filtered.push_back(i);
    if (!filtered.empty()) keep = std::move(filtered);
  }
  double total_w = 0.0;
  for (size_t i : keep) total_w += w[i];
  if (!(total_w > 0)) throw std::invalid_argument("total weight is zero");
  std::vector<float> scale(k, 0.0f);
  for (size_t i : keep) scale[i] = static_cast<float>(w[i] / total_w);
  std::vector<f32> prev_arrays;
  if (!momentum_obj.is_none() && momentum > 0) {
    py::list prev_list = momentum_obj.cast<py::list>();
    if (prev_list.size() != tensors) throw std::invalid_argument("momentum tensor count mismatch");
    prev_arrays.reserve(tensors);
    for (size_t t = 0; t < tensors; ++t) {
      auto arr = f32::ensure(prev_list[t]);
      if (!arr || static_cast<size_t>(arr.size()) != sizes[t]) throw std::invalid_argument("momentum tensor size mismatch");
      prev_arrays.push_back(arr);
    }
  }
  py::list out_list, mom_list;
  std::vector<py::array_t<float>> out_arrays, mom_arrays;
  out_arrays.reserve(tensors);
  mom_arrays.reserve(tensors);
  for (size_t t = 0; t < tensors; ++t) {
    out_arrays.emplace_back(sizes[t]);
    mom_arrays.emplace_back(sizes[t]);
  }
  {
    py::gil_scoped_release release;
    for (size_t t = 0; t < tensors; ++t) {
      const float *bp = base_arrays[t].data();
      float *op = out_arrays[t].mutable_data();
      float *mp = mom_arrays[t].mutable_data();
      const float *pp = prev_arrays.empty() ? nullptr : prev_arrays[t].data();
      size_t n = sizes[t];
      for (size_t i = 0; i < n; ++i) {
        double avg = 0.0;
        for (size_t u : keep) avg += static_cast<double>(upd_arrays[u][t].data()[i]) * scale[u];
        double buf = 0.0;
        double step = avg;
        if (momentum > 0) {
          double prev = pp ? static_cast<double>(pp[i]) : 0.0;
          buf = momentum * prev + avg;
          step = nesterov ? (avg + momentum * buf) : buf;
        }
        op[i] = static_cast<float>(static_cast<double>(bp[i]) + outer_lr * step);
        mp[i] = static_cast<float>(buf);
      }
    }
  }
  for (size_t t = 0; t < tensors; ++t) {
    out_list.append(out_arrays[t]);
    mom_list.append(mom_arrays[t]);
  }
  py::dict stats;
  stats["updates_received"] = k;
  stats["updates_used"] = keep.size();
  stats["total_weight"] = total_w;
  py::list dropped;
  for (size_t i = 0; i < k; ++i)
    if (std::find(keep.begin(), keep.end(), i) == keep.end()) dropped.append(i);
  stats["dropped_indices"] = dropped;
  py::list norm_list;
  for (double v : norms) norm_list.append(v);
  stats["norms"] = norm_list;
  py::list weight_list;
  for (size_t i = 0; i < k; ++i) weight_list.append(w[i]);
  stats["weights"] = weight_list;
  return py::make_tuple(out_list, mom_list, stats);
}

int64_t topk_sample(py::array logits_arr, int top_k, double temperature, uint64_t seed) {
  auto logits = f32::ensure(logits_arr);
  if (!logits) throw std::invalid_argument("logits must be array-like");
  size_t n = static_cast<size_t>(logits.size());
  const float *p = logits.data();
  if (temperature <= 0.0) {
    size_t best = 0;
    for (size_t i = 1; i < n; ++i)
      if (p[i] > p[best]) best = i;
    return static_cast<int64_t>(best);
  }
  std::vector<size_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  size_t k = (top_k > 0 && static_cast<size_t>(top_k) < n) ? static_cast<size_t>(top_k) : n;
  auto cmp = [&](size_t a, size_t b) { return p[a] > p[b]; };
  if (k < n) {
    std::nth_element(idx.begin(), idx.begin() + static_cast<long>(k), idx.end(), cmp);
    idx.resize(k);
  }
  std::sort(idx.begin(), idx.end(), cmp);
  float threshold = p[idx.back()];
  double sum = 0.0;
  std::vector<double> probs;
  probs.reserve(idx.size());
  for (size_t i : idx) {
    double e = std::exp((static_cast<double>(p[i]) - threshold) / temperature);
    probs.push_back(e);
    sum += e;
  }
  SplitMix64 rng(seed);
  double r = rng.uniform() * sum;
  double acc = 0.0;
  for (size_t j = 0; j < idx.size(); ++j) {
    acc += probs[j];
    if (r <= acc) return static_cast<int64_t>(idx[j]);
  }
  return static_cast<int64_t>(idx.back());
}

py::array_t<int64_t> byte_batch(py::array data_arr, int batch_size, int seq_len, uint64_t seed) {
  auto data = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>::ensure(data_arr);
  if (!data) throw std::invalid_argument("data must be array-like bytes");
  size_t n = static_cast<size_t>(data.size());
  size_t need = static_cast<size_t>(seq_len) + 1;
  if (n < need) throw std::invalid_argument("data shard too small");
  size_t span = n - need + 1;
  py::array_t<int64_t> out({static_cast<py::ssize_t>(batch_size), static_cast<py::ssize_t>(need)});
  int64_t *op = out.mutable_data();
  const uint8_t *dp = data.data();
  SplitMix64 rng(seed);
  for (int b = 0; b < batch_size; ++b) {
    size_t start = static_cast<size_t>(rng.next() % span);
    for (size_t t = 0; t < need; ++t) op[b * need + t] = static_cast<int64_t>(dp[start + t]);
  }
  return out;
}

py::tuple quantize_int8(py::array data_arr, int block_size) {
  auto data = f32::ensure(data_arr);
  if (!data) throw std::invalid_argument("data must be array-like");
  if (block_size < 1) throw std::invalid_argument("block_size must be > 0");
  size_t n = static_cast<size_t>(data.size());
  size_t blocks = (n + block_size - 1) / block_size;
  py::array_t<int8_t> out(n);
  py::array_t<float> scales(blocks);
  const float *src = data.data();
  int8_t *dst = out.mutable_data();
  float *sc = scales.mutable_data();
  for (size_t b = 0; b < blocks; ++b) {
    size_t begin = b * block_size;
    size_t end = std::min(n, begin + block_size);
    float absmax = 0.0f;
    for (size_t i = begin; i < end; ++i) absmax = std::max(absmax, std::fabs(src[i]));
    float scale = absmax / 127.0f;
    sc[b] = scale == 0.0f ? 1.0f : scale;
    for (size_t i = begin; i < end; ++i) {
      float q = std::round(src[i] / sc[b]);
      q = std::max(-127.0f, std::min(127.0f, q));
      dst[i] = static_cast<int8_t>(q);
    }
  }
  return py::make_tuple(out, scales);
}

py::array_t<float> dequantize_int8(py::array q_arr, py::array s_arr, int block_size) {
  auto q = py::array_t<int8_t, py::array::c_style | py::array::forcecast>::ensure(q_arr);
  auto scales = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(s_arr);
  if (!q || !scales) throw std::invalid_argument("invalid quantized input");
  size_t n = static_cast<size_t>(q.size());
  size_t blocks = (n + block_size - 1) / block_size;
  if (static_cast<size_t>(scales.size()) != blocks) throw std::invalid_argument("scale count mismatch");
  py::array_t<float> out(n);
  const int8_t *qp = q.data();
  const float *sp = scales.data();
  float *op = out.mutable_data();
  for (size_t i = 0; i < n; ++i) op[i] = static_cast<float>(qp[i]) * sp[i / block_size];
  return out;
}

std::string version() { return "xdc-native 0.1.0"; }

}  // namespace

PYBIND11_MODULE(_native, m) {
  m.doc() = "XDC native acceleration core (optional)";
  m.def("aggregate", &aggregate, "Fused robust aggregation and outer optimizer");
  m.def("aggregate_tensors", &aggregate_tensors, "Per-tensor fused robust aggregation");
  m.def("topk_sample", &topk_sample, "Deterministic temperature/top-k token sampler");
  m.def("byte_batch", &byte_batch, "Sample byte-level training batches");
  m.def("quantize_int8", &quantize_int8, "Block-wise int8 quantizer");
  m.def("dequantize_int8", &dequantize_int8, "Block-wise int8 dequantizer");
  m.def("version", &version);
}
