#include "KernelExecution.h"

#include <c10/core/DynamicCast.h>
#include <sys/mman.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

#include "torch_neuronx/csrc/c10/neuron/NeuronStream.h"
#include "torch_neuronx/csrc/core/NeuronDevice.h"
#include "torch_neuronx/csrc/core/NeuronLogging.h"
#include "torch_neuronx/csrc/core/allocator/NeuronCachingAllocator.h"
#include "torch_neuronx/csrc/core/compilation/NeuronCompiler.h"
#include "torch_neuronx/csrc/core/runtime/NRTHandler.h"
#include "torch_neuronx/csrc/core/runtime/NRTUtils.h"
#include "torch_neuronx/csrc/core/utils/NeuronExceptions.h"
#include "torch_neuronx/csrc/core/utils/PlatformUtils.h"
#include "torch_neuronx/csrc/utils/NonTemporalMemcpy.h"

namespace at::neuron {

// CompilableKernelExecution implementation

std::string CompilableKernelExecution::CreateCompilerExtensionHash(
    const std::string& additional_args, const std::string& optimization_level) {
  // Build extension from compiler-specific parameters (no hashing for readability)
  std::string platform_target = utils::GetPlatformTarget();
  std::string logical_cores = utils::GetLogicalNeuronCores();

  // Remove leading dash from optimization level for cleaner key
  std::string opt_level_suffix = optimization_level;
  if (!opt_level_suffix.empty() && opt_level_suffix[0] == '-') {
    opt_level_suffix = opt_level_suffix.substr(1);
  }

  // Sanitize additional_args for use in cache key (replace spaces with underscores)
  std::string sanitized_args = additional_args;
  std::replace(sanitized_args.begin(), sanitized_args.end(), ' ', '_');
  std::replace(sanitized_args.begin(), sanitized_args.end(), '=', '-');

  // Build extension: <target>_<lnc>_<opt_level>_<additional_args>
  std::string extension = platform_target + "_" + logical_cores + "_" + opt_level_suffix;

  if (!sanitized_args.empty()) {
    extension += "_" + sanitized_args;
  }

  return extension;
}

std::string CompilableKernelExecution::GenerateCacheKey(const std::string& base_key,
                                                        const std::string& additional_args,
                                                        const std::string& optimization_level) {
  std::string compiler_extension = CreateCompilerExtensionHash(additional_args, optimization_level);
  return base_key + "_" + compiler_extension;
}

bool CompilableKernelExecution::ValidateImpl() const {
  return !cache_key_.empty() && cache_key_[0] != '_';
}

// XLACompilableKernelExecution implementation

std::string XLACompilableKernelExecution::GetXLAAdditionalArgs() {
  // Base XLA compilation arguments
  std::string args = "--model-type transformer --auto-cast=none";

  // Append user-provided XLA flags from environment
  if (const char* cc_flags_env = std::getenv("NEURON_CC_FLAGS")) {
    args += " " + std::string(cc_flags_env);
  }

  return args;
}

std::string XLACompilableKernelExecution::GetXLAOptimizationLevel() {
  // Get optimization level from environment or use default
  if (const char* opt_level_env = std::getenv("NEURON_COMPILER_OPT_LEVEL")) {
    return std::string(opt_level_env);
  }
  return "-O1";
}

XLACompilableKernelExecution::XLACompilableKernelExecution(
    const std::string& op_name, std::vector<TensorDataRef>&& input_refs,
    std::vector<TensorDataRef>&& output_refs, const std::vector<TensorContext>& input_contexts,
    const std::vector<TensorContext>& output_contexts, const std::string& base_cache_key,
    const std::vector<uint8_t>& hlo_bytes, bool has_collectives, int device_id)
    : CompilableKernelExecution(
          GenerateCacheKey(base_cache_key, GetXLAAdditionalArgs(), GetXLAOptimizationLevel()),
          GetXLAAdditionalArgs(), GetXLAOptimizationLevel(), has_collectives),
      NeuronKernelExecutionTyped<KernelTypeEnum::kHLO, c10::DeviceType::PrivateUse1,
                                 c10::DeviceType::PrivateUse1>(
          op_name, std::move(input_refs), std::move(output_refs), input_contexts, output_contexts,
          device_id),
      hlo_bytes_(hlo_bytes) {}

bool XLACompilableKernelExecution::ValidateImpl() const {
  return CompilableKernelExecution::ValidateImpl() && !hlo_bytes_.empty();
}
void XLACompilableKernelExecution::ExecuteOrSchedule(nrt::ErrorTracker* err_tracker,
                                                     nrt::SequenceId* sequence_id,
                                                     c10::StreamId stream_id) const {
  const std::string& operation_name = GetOpName();
  bool is_async = (err_tracker != nullptr);
  TORCH_NEURONX_DEBUG("Executing compilable kernel", "operation=", operation_name,
                      "async=", is_async);

  const CompilableKernelExecution& compilable_kernel =
      static_cast<const CompilableKernelExecution&>(*this);
  int device_id = GetDeviceId();
  NRTHandler::ExecutionConfig config(device_id, 1 /* num_cores */);
  auto* model = NRTHandler::GetOrLoadModel(compilable_kernel, config);

  // Create slices for tensors with storage_offset > 0
  const auto src_data_ptrs = HandleContiguousSlicing(GetSrcDataPtrs(), src_tensor_ctxs_);
  const auto dst_data_ptrs = HandleContiguousSlicing(GetDstDataPtrs(), dst_tensor_ctxs_);

  NRTHandler::DispatchModelExecution(model, src_data_ptrs, dst_data_ptrs, config, err_tracker,
                                     sequence_id, stream_id);

  TORCH_NEURONX_DEBUG("NRT execution completed", "operation=", operation_name);
}

std::vector<uint8_t> XLACompilableKernelExecution::CompileToNeff() const {
  if (hlo_bytes_.empty()) {
    throw std::invalid_argument("Cannot compile: no HLO bytes available");
  }

  try {
    // Determine IR type: collectives always use XLA (HLO protobuf),
    // non-collectives may use StableHLO (MLIR) if environment variable is set
    std::string ir_type =
        (!HasCollectives() && NeuronCompiler::IsStableHLOEnabled()) ? "StableHLO" : "XLA";

    CompilationConfig config{.framework = "XLA",
                             .platform_target = utils::GetPlatformTarget(),
                             .logical_neuron_cores = utils::GetLogicalNeuronCores(),
                             .optimization_level = GetOptimizationLevel(),
                             .additional_args = GetAdditionalArgs()};
    return NeuronCompiler::CompileHloToNeff(hlo_bytes_, config, ir_type, GetOpName(), GetCacheKey(),
                                            "XLACompilableKernelExecution");
  } catch (const std::exception& e) {
    throw std::runtime_error("HLO to NEFF compilation failed for op '" + GetOpName() +
                             "': " + e.what());
  }
}

// CompileOnlyKernelExecution implementation
std::string CompileOnlyKernelExecution::GetCompileAdditionalArgs() {
  // Base torch.compile compilation arguments
  std::string args = "--model-type transformer --auto-cast=none";

  // Append user-provided flags from environment
  if (const char* cc_flags_env = std::getenv("NEURON_CC_FLAGS")) {
    args += " " + std::string(cc_flags_env);
  }

  return args;
}

std::string CompileOnlyKernelExecution::GetCompileOptimizationLevel() {
  // Get optimization level from environment or use default
  if (const char* opt_level_env = std::getenv("NEURON_COMPILER_OPT_LEVEL")) {
    return std::string(opt_level_env);
  }
  return "-O2";
}

CompileOnlyKernelExecution::CompileOnlyKernelExecution(const std::string& base_cache_key,
                                                       const std::vector<uint8_t>& stablehlo_bytes,
                                                       bool has_collectives)
    : CompilableKernelExecution(GenerateCacheKey(base_cache_key, GetCompileAdditionalArgs(),
                                                 GetCompileOptimizationLevel()),
                                GetCompileAdditionalArgs(), GetCompileOptimizationLevel(),
                                has_collectives),
      NeuronKernelExecution("torch_compile", -1),  // Compilation-only, no actual execution
      stablehlo_bytes_(stablehlo_bytes) {}

bool CompileOnlyKernelExecution::ValidateImpl() const {
  return CompilableKernelExecution::ValidateImpl() && !stablehlo_bytes_.empty();
}

std::vector<uint8_t> CompileOnlyKernelExecution::CompileToNeff() const {
  if (stablehlo_bytes_.empty()) {
    throw std::invalid_argument("Cannot compile: no StableHLO bytes available");
  }

  try {
    // torch.compile always uses StableHLO (MLIR) format
    std::string ir_type = "StableHLO";

    CompilationConfig config{.framework = "XLA",
                             .platform_target = utils::GetPlatformTarget(),
                             .logical_neuron_cores = utils::GetLogicalNeuronCores(),
                             .optimization_level = GetOptimizationLevel(),
                             .additional_args = GetAdditionalArgs()};
    return NeuronCompiler::CompileHloToNeff(stablehlo_bytes_, config, ir_type, "torch_compile",
                                            GetCacheKey(), "CompileOnlyKernelExecution");
  } catch (const std::exception& e) {
    throw std::runtime_error("StableHLO to NEFF compilation failed for torch.compile: " +
                             std::string(e.what()));
  }
}

// NeffDirectKernelExecution implementation
NeffDirectKernelExecution::NeffDirectKernelExecution(
    const std::string& graph_name, const std::string& cache_key,
    std::vector<TensorDataRef>&& input_refs, std::vector<TensorDataRef>&& output_refs,
    const std::vector<TensorContext>& input_contexts,
    const std::vector<TensorContext>& output_contexts, int device_id, bool has_collectives)
    : NeuronKernelExecutionTyped<KernelTypeEnum::kHLO, c10::DeviceType::PrivateUse1,
                                 c10::DeviceType::PrivateUse1>(
          graph_name, std::move(input_refs), std::move(output_refs), input_contexts,
          output_contexts, device_id),
      CompilableKernelExecution(cache_key, "", "", has_collectives) {}

bool NeffDirectKernelExecution::ValidateImpl() const { return HasOutputs(); }

std::vector<uint8_t> NeffDirectKernelExecution::CompileToNeff() const {
  // NEFF is pre-compiled. Must always result in a cache hit.
  throw std::logic_error("NeffDirectKernelExecution expects pre-compiled graphs.");
}

void NeffDirectKernelExecution::ExecuteOrSchedule(nrt::ErrorTracker* err_tracker,
                                                  nrt::SequenceId* sequence_id,
                                                  c10::StreamId stream_id) const {
  // Create slices for tensors with storage_offset > 0
  const auto src_data_ptrs = HandleContiguousSlicing(GetSrcDataPtrs(), src_tensor_ctxs_);
  const auto dst_data_ptrs = HandleContiguousSlicing(GetDstDataPtrs(), dst_tensor_ctxs_);

  const CompilableKernelExecution& compilable_kernel =
      static_cast<const CompilableKernelExecution&>(*this);
  int device_id = GetDeviceId();
  NRTHandler::ExecutionConfig config(device_id, 1);
  auto* model = NRTHandler::GetOrLoadModel(compilable_kernel, config);
  NRTHandler::DispatchModelExecution(model, src_data_ptrs, dst_data_ptrs, config, err_tracker,
                                     sequence_id, stream_id);
}

// CollectiveDirectKernelExecution implementation

CollectiveDirectKernelExecution::CollectiveDirectKernelExecution(
    const std::string& op_name, CollectiveType collective_type,
    std::vector<TensorDataRef>&& src_refs, std::vector<TensorDataRef>&& dst_refs,
    c10d::ReduceOp reduce_op, int device_id)
    : NeuronKernelExecutionTyped<KernelTypeEnum::kCollective, c10::DeviceType::PrivateUse1,
                                 c10::DeviceType::PrivateUse1>(op_name, std::move(src_refs),
                                                               std::move(dst_refs), device_id),
      collective_type_(collective_type),
      reduce_op_(reduce_op) {}

bool CollectiveDirectKernelExecution::ValidateImpl() const { return HasInputs() && HasOutputs(); }

void CollectiveDirectKernelExecution::Prepare() const {
  // TODO: Implement collective preparation
}

void CollectiveDirectKernelExecution::ExecuteOrSchedule(nrt::ErrorTracker* err_tracker,
                                                        nrt::SequenceId* sequence_id,
                                                        c10::StreamId stream_id) const {
  // TODO: Implement async collective scheduling.
  Execute();
}

void CollectiveDirectKernelExecution::Execute() const {
  TORCH_CHECK(false, "CollectiveDirectKernelExecution::Execute() not yet implemented");
}

HintDirectKernelExecution::HintDirectKernelExecution(const std::string& op_name, void* ptr,
                                                     size_t size_bytes, int device_id,
                                                     HintType hint_type)
    : NeuronKernelExecution(op_name, device_id),
      ptr_(ptr),
      size_bytes_(size_bytes),
      hint_type_(hint_type) {}

CopyDirectKernelExecution::CopyDirectKernelExecution(const std::string& op_name,
                                                     TensorDataRef src_ref, TensorDataRef dst_ref,
                                                     size_t src_offset_bytes,
                                                     size_t dst_offset_bytes, size_t size_bytes,
                                                     int device_id)
    : NeuronKernelExecutionTyped<KernelTypeEnum::kCopy, c10::DeviceType::PrivateUse1,
                                 c10::DeviceType::PrivateUse1>(op_name, {std::move(src_ref)},
                                                               {std::move(dst_ref)}, device_id),
      src_offset_bytes_(src_offset_bytes),
      dst_offset_bytes_(dst_offset_bytes),
      size_bytes_(size_bytes) {}

bool CopyDirectKernelExecution::ValidateImpl() const { return size_bytes_ > 0; }

void CopyDirectKernelExecution::ExecuteOrSchedule(nrt::ErrorTracker* err_tracker,
                                                  nrt::SequenceId* sequence_id,
                                                  c10::StreamId stream_id) const {
  nrt_tensor_t* src_ptr = GetSrcDataPtrs()[0];
  nrt_tensor_t* dst_ptr = GetDstDataPtrs()[0];

  if (err_tracker == nullptr) {
    NRT_STATUS status =
        nrt::CopyTensor(src_ptr, src_offset_bytes_, dst_ptr, dst_offset_bytes_, size_bytes_);
    if (status != NRT_SUCCESS) {
      std::string error_msg = "Failed to copy tensor data. Status: " + std::to_string(status);
      throw torch_neuronx::ExecutionRuntimeException(error_msg, status);
    }
  } else {
    // Queue parameter is not used yet; always use queue 0
    constexpr int kDefaultQueue = 0;
    NRT_STATUS status =
        nrt::ScheduleTensorCopy(src_ptr, src_offset_bytes_, dst_ptr, dst_offset_bytes_, size_bytes_,
                                kDefaultQueue, err_tracker, sequence_id);
    if (status != NRT_SUCCESS) {
      std::string error_msg =
          "Failed to schedule async tensor copy. Status: " + std::to_string(status);
      throw torch_neuronx::ExecutionRuntimeException(error_msg, status);
    }
  }
}

WriteDirectKernelExecution::WriteDirectKernelExecution(const std::string& op_name, void* src_ptr,
                                                       TensorDataRef dst_ref,
                                                       size_t dst_offset_bytes,
                                                       size_t dst_size_bytes, int device_id)
    : NeuronKernelExecutionTyped<KernelTypeEnum::kWrite, c10::DeviceType::CPU,
                                 c10::DeviceType::PrivateUse1>(op_name, {TensorDataRef{src_ptr}},
                                                               {std::move(dst_ref)}, device_id),
      dst_offset_bytes_(dst_offset_bytes),
      dst_size_bytes_(dst_size_bytes) {}

bool WriteDirectKernelExecution::ValidateImpl() const { return dst_size_bytes_ > 0; }

void* WriteDirectKernelExecution::AllocateBounceBuffer(size_t size_bytes) {
  bounce_buffer_.resize((size_bytes + 63) / 64);
  void* bounce = bounce_buffer_.data()->data;
  torch_neuronx::utils::non_temporal_memcpy(bounce, src_ptrs_[0], size_bytes);
  torch_neuronx::utils::non_temporal_sfence();
  src_ptrs_[0] = bounce;
  src_data_ptrs_[0].ptr = bounce;
  return bounce;
}

void WriteDirectKernelExecution::ExecuteOrSchedule(nrt::ErrorTracker* err_tracker,
                                                   nrt::SequenceId* sequence_id,
                                                   c10::StreamId stream_id) const {
  nrt_tensor_t* dst_data_ptr = GetDstDataPtrs()[0];
  void* src_data_ptr =
      bounce_buffer_.empty() ? GetSrcDataPtrs()[0] : const_cast<char*>(bounce_buffer_.data()->data);

  if (err_tracker == nullptr) {
    NRT_STATUS status = nrt::WriteTensor(dst_data_ptr, const_cast<void*>(src_data_ptr),
                                         dst_offset_bytes_, dst_size_bytes_);
    if (status != NRT_SUCCESS) {
      throw torch_neuronx::ExecutionRuntimeException(
          "Failed to write tensor data. Status: " + std::to_string(status), status);
    }
  } else {
    // Queue parameter is not used yet; always use queue 0
    constexpr int kDefaultQueue = 0;
    NRT_STATUS status =
        nrt::ScheduleTensorWrite(dst_data_ptr, src_data_ptr, dst_offset_bytes_, dst_size_bytes_,
                                 kDefaultQueue, err_tracker, sequence_id);
    if (status != NRT_SUCCESS) {
      std::string error_msg =
          "Failed to schedule async tensor write. Status: " + std::to_string(status);
      throw torch_neuronx::ExecutionRuntimeException(error_msg, status);
    }
  }
}

ReadDirectKernelExecution::ReadDirectKernelExecution(const std::string& op_name,
                                                     TensorDataRef src_ref, void* dst_ptr,
                                                     size_t src_offset_bytes, size_t src_size_bytes,
                                                     int device_id)
    : NeuronKernelExecutionTyped<KernelTypeEnum::kRead, c10::DeviceType::PrivateUse1,
                                 c10::DeviceType::CPU>(op_name, {std::move(src_ref)},
                                                       {TensorDataRef{dst_ptr}}, device_id),
      src_offset_bytes_(src_offset_bytes),
      src_size_bytes_(src_size_bytes) {}

bool ReadDirectKernelExecution::ValidateImpl() const { return src_size_bytes_ > 0; }

void ReadDirectKernelExecution::ExecuteOrSchedule(nrt::ErrorTracker* err_tracker,
                                                  nrt::SequenceId* sequence_id,
                                                  c10::StreamId stream_id) const {
  nrt_tensor_t* src_ptr = GetSrcDataPtrs()[0];
  void* dst_ptr = GetDstDataPtrs()[0];

  if (err_tracker == nullptr) {
    NRT_STATUS status = nrt::ReadTensor(src_ptr, dst_ptr, src_offset_bytes_, src_size_bytes_);
    if (status != NRT_SUCCESS) {
      std::string error_msg = "Failed to read tensor data. Status: " + std::to_string(status);
      throw torch_neuronx::ExecutionRuntimeException(error_msg, status);
    }
  } else {
    // Queue parameter is not used yet; always use queue 0
    constexpr int kDefaultQueue = 0;
    NRT_STATUS status =
        nrt::ScheduleTensorRead(dst_ptr, src_ptr, src_offset_bytes_, src_size_bytes_, kDefaultQueue,
                                err_tracker, sequence_id);
    if (status != NRT_SUCCESS) {
      std::string error_msg =
          "Failed to schedule async tensor read. Status: " + std::to_string(status);
      throw torch_neuronx::ExecutionRuntimeException(error_msg, status);
    }
  }
}

EventDirectKernelExecution::EventDirectKernelExecution(const std::string& op_name,
                                                       const NeuronEvent& event, EventAction action,
                                                       int device_id)
    : NeuronKernelExecution(op_name, device_id), event_(event), action_(action) {}

bool EventDirectKernelExecution::ValidateImpl() const {
  // Event primitives don't require inputs/outputs
  return true;
}

bool EventDirectKernelExecution::IsReady() const {
  if (action_ == EventAction::kSignal) {
    return true;
  }
  return event_.query();
}

void EventDirectKernelExecution::ExecuteOrSchedule(nrt::ErrorTracker* err_tracker,
                                                   nrt::SequenceId* sequence_id,
                                                   c10::StreamId stream_id) const {
  // Events are host-side operations - always execute synchronously
  if (action_ == EventAction::kWait) {
    // At this point, the event is assumed to have completed
    TORCH_CHECK(IsReady(), "Event not ready for wait");
    return;
  }
  event_.complete_event();
}

}  // namespace at::neuron
