#pragma once

#include <c10/core/Device.h>
#include <c10/util/Exception.h>
#include <torch/torch.h>

// Include NRT headers
extern "C" {
#include <nrt/nrt.h>
#include <nrt/nrt_async.h>
#include <nrt/nrt_experimental.h>
#include <nrt/nrt_profile.h>
}

#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Forward declarations
namespace c10_neuron {
namespace NeuronCachingAllocator {
nrt_tensor_t* findTensor(void* ptr);
}  // namespace NeuronCachingAllocator
}  // namespace c10_neuron

namespace at::neuron::nrt {

using SequenceId = nrta_seq_t;

// Number of device kernel types that need completion tracking
// Corresponds to: kHLO, kCollective, kCopy, kWrite, kRead
// Excludes host-side operations: kEvent, kHint
constexpr size_t kDeviceKernelTypeCount = 5;

// State for tracking async NRT operation scheduling
struct AsyncSchedulingState {
  SequenceId sequence_id{0};      // Sequence number from ExecuteOrSchedule()
  bool is_scheduled{false};       // Whether async scheduling was called (in-flight)
  bool is_pending_signal{false};  // kSignal event waiting for prior op to complete

  void Reset() {
    sequence_id = 0;
    is_scheduled = false;
    is_pending_signal = false;
  }
};

// Completion tracking state for async NRT operations
struct CompletionState {
  std::array<SequenceId, kDeviceKernelTypeCount> last_completed_seq{};
  std::array<int, kDeviceKernelTypeCount> inflight_count{};
  uint32_t lnc_idx{0};
};

struct AsyncError {
  SequenceId seq_id;
  uint64_t error_code;
};

class ErrorTracker {
 public:
  explicit ErrorTracker(uint32_t lnc_idx);
  ~ErrorTracker() noexcept;

  ErrorTracker(const ErrorTracker&) = delete;
  ErrorTracker& operator=(const ErrorTracker&) = delete;
  ErrorTracker(ErrorTracker&& other) noexcept;
  ErrorTracker& operator=(ErrorTracker&& other) noexcept;

  nrta_error_tracker_t* get() const noexcept { return tracker_; }
  std::vector<AsyncError> GetAndClearErrors();
  bool IsValid() const noexcept { return tracker_ != nullptr; }

 private:
  nrta_error_tracker_t* tracker_;
};

// Provides automatic resource management for NRT tensor sets with
// exception-safe operations and move semantics. Automatically allocates
// the tensor set on construction and destroys it on destruction.
class TensorSet {
 public:
  TensorSet();
  ~TensorSet() noexcept;

  // Non-copyable but movable for efficiency
  TensorSet(const TensorSet&) = delete;
  TensorSet& operator=(const TensorSet&) = delete;

  TensorSet(TensorSet&& other) noexcept;
  TensorSet& operator=(TensorSet&& other) noexcept;

  // Get the underlying NRT tensor set pointer
  nrt_tensor_set_t* get() const noexcept { return tensor_set_; }

  NRT_STATUS AddTensor(std::string_view name, nrt_tensor_t* data_ptr);

  bool IsValid() const noexcept { return tensor_set_ != nullptr; }

  explicit operator bool() const noexcept { return IsValid(); }

 private:
  nrt_tensor_set_t* tensor_set_;

  void validate() const;
  void reset() noexcept;
};

// Handles NEFF loading and automatic cleanup with comprehensive
// error handling and resource tracking.
class Model {
 public:
  Model() noexcept;
  ~Model() noexcept;

  // Load NEFF data into the model
  NRT_STATUS Load(const std::vector<uint8_t>& neff_bytes, int device_id = 0, int num_cores = 1);

  // Load NEFF data with collective operations
  NRT_STATUS LoadCollectives(const std::vector<uint8_t>& neff_bytes, int device_id = 0,
                             int num_cores = 1);

  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  Model(Model&& other) noexcept;
  Model& operator=(Model&& other) noexcept;

  // Access the underlying model
  nrt_model_t* get() const noexcept { return model_; }

  // Dispatch model execution. When err_tracker is nullptr, executes synchronously.
  // When err_tracker is provided, schedules asynchronously and returns sequence in req_sequence.
  NRT_STATUS DispatchExecution(nrt_tensor_set_t* input_set, nrt_tensor_set_t* output_set,
                               int queue = 0, ErrorTracker* err_tracker = nullptr,
                               SequenceId* req_sequence = nullptr);

  bool IsValid() const noexcept { return model_ != nullptr; }
  explicit operator bool() const noexcept { return IsValid(); }

  // Check if model is loaded (alias for IsValid for clarity)
  bool IsLoaded() const noexcept { return IsValid(); }

 private:
  nrt_model_t* model_;

  void validate() const;
  void reset() noexcept;
};

// Standalone NRT tensor allocation/deallocation functions
// These provide direct access to NRT tensor operations

/**
 * Allocate an NRT tensor
 * @param placement Tensor placement (device or host)
 * @param vnc_id Virtual NeuronCore ID
 * @param size_bytes Size in bytes
 * @param name Tensor name for debugging
 * @param tensor Output pointer to allocated tensor
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS AllocateTensor(nrt_tensor_placement_t placement, int vnc_id, size_t size_bytes,
                                 const char* name, nrt_tensor_t** tensor) {
  return nrt_tensor_allocate(placement, vnc_id, size_bytes, name, tensor);
}

/**
 * Free an NRT tensor
 * @param tensor Pointer to tensor pointer (will be set to nullptr)
 */
inline void FreeTensor(nrt_tensor_t** tensor) { nrt_tensor_free(tensor); }

/**
 * Copy data between NRT tensors
 * @param src Source tensor
 * @param src_offset Offset in source tensor
 * @param dst Destination tensor
 * @param dst_offset Offset in destination tensor
 * @param size_bytes Number of bytes to copy
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS CopyTensor(nrt_tensor_t* src, size_t src_offset, nrt_tensor_t* dst,
                             size_t dst_offset, size_t size_bytes) {
  return nrt_tensor_copy(src, src_offset, dst, dst_offset, size_bytes);
}

/**
 * Write data from CPU memory to NRT tensor
 * @param dst Destination NRT tensor
 * @param src Source CPU memory pointer
 * @param dst_offset Offset in destination tensor
 * @param size_bytes Number of bytes to write
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS WriteTensor(nrt_tensor_t* dst, void* src, size_t dst_offset, size_t size_bytes) {
  return nrt_tensor_write(dst, src, dst_offset, size_bytes);
}

/**
 * Read data from NRT tensor to CPU memory
 * @param src Source NRT tensor
 * @param dst Destination CPU memory pointer
 * @param src_offset Offset in source tensor
 * @param size_bytes Number of bytes to read
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS ReadTensor(nrt_tensor_t* src, void* dst, size_t src_offset, size_t size_bytes) {
  return nrt_tensor_read(src, dst, src_offset, size_bytes);
}

/**
 * Execute a barrier across devices
 * @param vnc Virtual neuron core id
 * @param global_device_id Global device id for barrier coordination
 * @param global_device_count Total number of devices participating
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS Barrier(int32_t vnc, uint32_t global_device_id, uint32_t global_device_count) {
  return nrt_barrier(vnc, global_device_id, global_device_count);
}

/**
 * Get the size of an NRT tensor
 * @param tensor Tensor to query
 * @return Size in bytes
 */
inline size_t GetTensorSize(nrt_tensor_t* tensor) { return nrt_tensor_get_size(tensor); }

/**
 * Allocate a slice of an existing NRT tensor
 * @param base_tensor Base tensor to slice from
 * @param offset Offset in base tensor
 * @param size Size of slice
 * @param name Name for the slice tensor
 * @param slice_tensor Output pointer to slice tensor
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS AllocateSliceTensor(nrt_tensor_t* base_tensor, size_t offset, size_t size,
                                      const char* name, nrt_tensor_t** slice_tensor) {
  return nrt_tensor_allocate_slice(base_tensor, offset, size, name, slice_tensor);
}

/**
 * Get virtual address of a tensor
 */
inline void* GetTensorVA(nrt_tensor_t* tensor) { return nrt_tensor_get_va(tensor); }

// NRT Profiling/Tracing Functions
// These provide access to NRT inspect functionality for comprehensive profiling

/**
 * Stop tracing/profiling and dump profile data
 * Dumps captured profile data to the configured output directory
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS StopInspect() { return nrt_inspect_stop(); }

// NRT Profiling with Options Functions
// These provide wrapper functions for the advanced NRT inspect configuration APIs

/**
 * Allocate memory for the options structure for configurable profiling
 * @param options Output pointer to allocated config structure
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS AllocateInspectConfig(nrt_inspect_config_t** options) {
  return nrt_inspect_config_allocate(options);
}

/**
 * Free memory for the options structure
 * @param options Pointer to config structure to free
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS FreeInspectConfig(nrt_inspect_config_t* options) {
  return nrt_inspect_config_free(options);
}

/**
 * Set all fields of the config structure to their default values
 * @param options Pointer to config structure
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS SetInspectConfigDefaults(nrt_inspect_config_t* options) {
  return nrt_inspect_config_set_defaults(options);
}

/**
 * Set the output directory for profiling results
 * @param options Pointer to config structure
 * @param output_dir Path to the output directory
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS SetInspectConfigOutputDir(nrt_inspect_config_t* options, const char* output_dir) {
  return nrt_inspect_config_set_output_dir(options, output_dir);
}

/**
 * Enable or disable inspect profiling for normal execution
 * @param options Pointer to config structure
 * @param enable_inspect Boolean to enable or disable profiling
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS SetInspectConfigEnableInspect(nrt_inspect_config_t* options,
                                                bool enable_inspect) {
  return nrt_inspect_config_set_enable_inspect(options, enable_inspect);
}

/**
 * Set system trace capture enabled for a specific NeuronCore
 * @param options Pointer to config structure
 * @param nc_idx Index of the NeuronCore
 * @param enabled Boolean to enable or disable capture
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS SetInspectConfigCaptureEnabledForNC(nrt_inspect_config_t* options,
                                                      uint32_t nc_idx, bool enabled) {
  return nrt_inspect_config_set_capture_enabled_for_nc(options, nc_idx, enabled);
}

/**
 * Set system trace capture enabled for a specific event type
 * @param options Pointer to config structure
 * @param event_type String name of the event type
 * @param enabled Boolean to enable or disable capture
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS SetInspectConfigCaptureEnabledForEventType(nrt_inspect_config_t* options,
                                                             const char* event_type, bool enabled) {
  // TODO: linking to this API fails fix merged update once cherry-picked to 2.30
  //  return nrt_inspect_config_set_capture_enabled_for_event_type_string(options, event_type,
  //  enabled); (SIM ID: P372975680)
  return NRT_STATUS(0);
}

/**
 * Get all available activity type strings
 * @param activity_types Output pointer to array of activity type strings
 * @param count Output pointer to number of activity types
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS GetAllActivityTypes(const char*** activity_types, size_t* count) {
  return nrt_inspect_config_get_all_activity_types(activity_types, count);
}

/**
 * Free activity types array allocated by GetAllActivityTypes
 * @param activity_types Pointer to activity types array
 * @param count Number of activity types
 */
inline void FreeActivityTypes(const char** activity_types, size_t count) {
  nrt_inspect_config_free_activity_types(activity_types, count);
}

/**
 * Begin tracing/profiling with configurable options
 * @param options Pointer to config structure with profiling options
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS BeginInspectWithOptions(nrt_inspect_config_t* options) {
  return nrt_inspect_begin_with_options(options);
}

// ============================================================================
// Asynchronous NRT operations
// ============================================================================

/**
 * Enqueue an asynchronous tensor write request
 * @param tensor Destination tensor
 * @param buf Host buffer containing source data
 * @param offset Offset into the tensor
 * @param size Number of bytes to write
 * @param queue XU queue to use
 * @param err_tracker Error tracker for capturing async errors
 * @param req_sequence Output sequence number of the scheduled request
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS ScheduleTensorWrite(nrt_tensor_t* tensor, const void* buf, uint64_t offset,
                                      uint64_t size, int queue, ErrorTracker* err_tracker,
                                      SequenceId* req_sequence) {
  return nrta_tensor_write(tensor, buf, offset, size, queue,
                           err_tracker ? err_tracker->get() : nullptr, req_sequence);
}

/**
 * Enqueue an asynchronous tensor read request
 * @param buf Destination host buffer
 * @param tensor Source tensor
 * @param offset Offset into the tensor
 * @param size Number of bytes to read
 * @param queue XU queue to use
 * @param err_tracker Error tracker for capturing async errors
 * @param req_sequence Output sequence number of the scheduled request
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS ScheduleTensorRead(void* buf, nrt_tensor_t* tensor, uint64_t offset,
                                     uint64_t size, int queue, ErrorTracker* err_tracker,
                                     SequenceId* req_sequence) {
  return nrta_tensor_read(buf, tensor, offset, size, queue,
                          err_tracker ? err_tracker->get() : nullptr, req_sequence);
}

/**
 * Enqueue an asynchronous tensor copy request
 * @param src Source tensor
 * @param src_offset Offset in source tensor
 * @param dst Destination tensor
 * @param dst_offset Offset in destination tensor
 * @param size_bytes Number of bytes to copy
 * @param queue XU queue to use
 * @param err_tracker Error tracker for capturing async errors
 * @param req_sequence Output sequence number of the scheduled request
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS ScheduleTensorCopy(nrt_tensor_t* src, uint64_t src_offset, nrt_tensor_t* dst,
                                     uint64_t dst_offset, uint64_t size_bytes, int queue,
                                     ErrorTracker* err_tracker, SequenceId* req_sequence) {
  return nrta_tensor_copy(src, src_offset, dst, dst_offset, size_bytes, queue,
                          err_tracker ? err_tracker->get() : nullptr, req_sequence);
}

/**
 * Check completion status of a scheduled async request
 * @param sequence_id Scheduled request sequence id
 * @param is_completed Output: true if the request is completed, false otherwise
 * @return NRT_SUCCESS if the request is valid, NRT_INVALID if the sequence_id is not valid
 */
inline NRT_STATUS IsNRTAsyncRequestCompleted(SequenceId sequence_id, bool* is_completed) {
  return nrta_is_completed(sequence_id, is_completed);
}

/**
 * Get sequence number of the last completed request for an execution unit queue
 * @param lnc LNC id
 * @param kernel_type_index Index from GetDeviceKernelTypeIndex() (0-4)
 * @param queue XU's queue
 * @param sequence_id Output: last completed sequence number
 * @return NRT_STATUS indicating success or failure
 */
inline NRT_STATUS GetLastCompletedRequest(uint32_t lnc, size_t kernel_type_index, int queue,
                                          SequenceId* sequence_id) {
  // Mapping from KernelTypeEnum index to NRT's nrta_xu_t
  static constexpr nrta_xu_t kKernelTypeIndexToXU[] = {
      NRTA_XU_COMPUTE,       // kHLO (index 0)
      NRTA_XU_COLLECTIVES,   // kCollective (index 1)
      NRTA_XU_TENSOR_OP,     // kCopy (index 2)
      NRTA_XU_TENSOR_WRITE,  // kWrite (index 3)
      NRTA_XU_TENSOR_READ,   // kRead (index 4)
  };
  return nrta_get_sequence(lnc, kKernelTypeIndexToXU[kernel_type_index], queue, sequence_id);
}

}  // namespace at::neuron::nrt
