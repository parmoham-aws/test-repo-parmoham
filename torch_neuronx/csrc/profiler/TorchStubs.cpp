/*
 * Neuron Profiler Stubs Implementation
 *
 * Implements torch::profiler::impl::ProfilerStubs interface for
 * PrivateUse1 device (Neuron).
 *
 * Note: This is separate from the Kineto activity profiler
 * (Profiler.cpp).
 * PyTorch has two profiler registration systems:
 * 1. Kineto Activity Profiler - for trace collection (Profiler.cpp)
 * 2. ProfilerStubs interface - for device-specific timing/sync (this file)
 */

#include <c10/core/DeviceType.h>
#include <torch/csrc/profiler/stubs/base.h>

#include <chrono>
#include <functional>

namespace at::neuron {

struct TorchStubs : public torch::profiler::impl::ProfilerStubs {
  /**
   * Record a timestamp event for profiling.
   * Called by PyTorch profiler to capture timing information.
   */
  void record(c10::DeviceIndex* device, torch::profiler::impl::ProfilerVoidEventStub* event,
              int64_t* cpu_ns) const override {
    // Record current device (0 for now - TODO: get actual current device)
    if (device) {
      *device = 0;
    }

    // Record CPU timestamp in nanoseconds
    if (cpu_ns) {
      *cpu_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now().time_since_epoch())
                    .count();
    }

    // No device event for now - CPU timestamps are sufficient for initial impl
    if (event) {
      *event = nullptr;
    }
  }

  /**
   * Calculate elapsed time between two events.
   * Returns time in milliseconds.
   */
  float elapsed(const torch::profiler::impl::ProfilerVoidEventStub* /*event*/,
                const torch::profiler::impl::ProfilerVoidEventStub* /*event2*/) const override {
    // TODO: Implement with actual device events when available
    // For now return 0 since we don't create device events
    return 0.0f;
  }

  /**
   * Mark an event with a name (NVTX-style annotation).
   * Optional - no-op for now.
   */
  void mark(const char* /*name*/) const override {
    // No-op for now - could integrate with Neuron tracing in future
  }

  /**
   * Push a range onto the profiler stack.
   * Optional - no-op for now.
   */
  void rangePush(const char* /*name*/) const override {
    // No-op for now - could integrate with Neuron tracing in future
  }

  /**
   * Pop a range from the profiler stack.
   * Optional - no-op for now.
   */
  void rangePop() const override {
    // No-op for now - could integrate with Neuron tracing in future
  }

  /**
   * Returns true to indicate that profiler stubs are available for this device.
   * Must return true to prevent "PrivateUse1 used in profiler but
   * not enabled" error.
   */
  bool enabled() const override { return true; }

  /**
   * Iterate over each device and call the provided callback.
   * Used by profiler to collect per-device information.
   */
  void onEachDevice(std::function<void(int)> op) const override {
    // TODO: Iterate over actual available Neuron devices
    // For now, call for device 0 only
    op(0);
  }

  /**
   * Synchronize all device operations.
   * Called by profiler to ensure all operations complete before collecting
   * traces.
   */
  void synchronize() const override {
    // TODO: Call NRT synchronization when available
    // This should wait for all pending Neuron operations to complete
  }
};

static TorchStubs neuron_profiler_stubs;

// Auto-register profiler stubs at module load time
struct RegisterTorchStubs {
  RegisterTorchStubs() {
    torch::profiler::impl::registerPrivateUse1Methods(&neuron_profiler_stubs);
  }
} register_neuron_profiler_stubs;

}  // namespace at::neuron
