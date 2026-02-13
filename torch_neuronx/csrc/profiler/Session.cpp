#include "Session.h"

#include <libkineto.h>
#include <nrt/nrt_profile.h>
#include <sys/stat.h>

#include <algorithm>
#include <chrono>

#include "LogWrapper.h"
#include "NConfigParser.h"

// Forward declaration for Neuron runtime lazy initialization
namespace torch_neuronx {
void maybe_lazy_init();
}

namespace at::neuron {

namespace {

using namespace torch_neuronx::profiler;

// Helper to cast context to NRT config
inline nrt_inspect_config_t* toNrtConfig(void* ctx) {
  return static_cast<nrt_inspect_config_t*>(ctx);
}

ConfigRegistry createNeuronConfigRegistry() {
  return {
      {"max_events_per_nc", NConfigParser::parseUint64,
       [](void* ctx, const ConfigValue& v) {
         NRT_STATUS status = nrt_inspect_config_set_sys_trace_max_events_per_nc(
             toNrtConfig(ctx), std::get<uint64_t>(v));
         if (status != NRT_SUCCESS) {
           NPROF_WARN("  Failed to set max_events_per_nc (status:", status, ")");
           return false;
         }
         NPROF_INFO("  Set max_events_per_nc:", std::get<uint64_t>(v));
         return true;
       }},

      {"capture_enabled_for_nc", NConfigParser::parseNcIndexList,
       [](void* ctx, const ConfigValue& v) {
         for (uint32_t ncIdx : std::get<std::vector<uint32_t>>(v)) {
           NRT_STATUS status =
               nrt_inspect_config_set_capture_enabled_for_nc(toNrtConfig(ctx), ncIdx, true);
           if (status != NRT_SUCCESS) {
             NPROF_WARN("  Failed to enable capture for NC:", ncIdx);
           } else {
             NPROF_INFO("  Enabled capture for NeuronCore:", ncIdx);
           }
         }
         return true;
       }},

      {"host_memory", NConfigParser::parseBool,
       [](void* ctx, const ConfigValue& v) {
         bool enabled = std::get<bool>(v);
         nrt_inspect_config_set_activity(toNrtConfig(ctx), "host_memory", enabled);
         NPROF_INFO("  Set host_memory:", enabled ? "enabled" : "disabled");
         return true;
       }},

      {"profile_output_dir", NConfigParser::parseString,
       [](void* ctx, const ConfigValue& v) {
         NRT_STATUS status =
             nrt_inspect_config_set_output_dir(toNrtConfig(ctx), std::get<std::string>(v).c_str());
         if (status != NRT_SUCCESS) {
           NPROF_WARN("  Failed to set profile_output_dir (status:", status, ")");
           return false;
         }
         NPROF_INFO("  Set profile_output_dir:", std::get<std::string>(v));
         return true;
       }},

      // Example of new config:
      // {"session_id", NConfigParser::parseInt64,
      //  [](void* ctx, const ConfigValue& v) {
      //      return nrt_inspect_config_set_session_id(
      //          toNrtConfig(ctx), std::get<int64_t>(v)) == NRT_SUCCESS;
      //  }},
  };
}

}  // namespace

// TODO: Block starting multiple parallel sessions. Verify pytorch behavior.
NeuronActivityProfilerSession::NeuronActivityProfilerSession(
    const std::set<libkineto::ActivityType>& activityTypes, const libkineto::Config& config)
    : activityTypes_(activityTypes), config_(config.clone()) {
  NPROF_INFO("Profiler session created.");
  NPROF_INFO("  Log file:", config_->activitiesLogFile());
}

void NeuronActivityProfilerSession::initNrtConfig() {
  NRT_STATUS status = nrt_inspect_config_allocate(&nrtConfig_);

  if (status != NRT_SUCCESS) {
    NPROF_ERROR(errors_, "Failed to allocate NRT config:", std::to_string(status));
    nrtConfig_ = nullptr;
    return;
  }

  nrt_inspect_config_set_defaults(nrtConfig_);
  nrt_inspect_config_set_enable_inspect(nrtConfig_, true);

  applyActivityTypes();
  applyCustomConfig();
}

void NeuronActivityProfilerSession::applyActivityTypes() {
  if (!nrtConfig_) {
    return;
  }

  // First disable all activities, then selectively enable based on Kineto request
  nrt_inspect_config_set_activity(nrtConfig_, "all", false);

  for (const auto& activityType : activityTypes_) {
    switch (activityType) {
      case libkineto::ActivityType::PRIVATEUSE1_RUNTIME:
        nrt_inspect_config_set_activity(nrtConfig_, "system_profile", true);
        NPROF_INFO("  Enabled NRT activity: system_profile (from PRIVATEUSE1_RUNTIME)");
        break;
      case libkineto::ActivityType::PRIVATEUSE1_DRIVER:
        nrt_inspect_config_set_activity(nrtConfig_, "device_profile", true);
        nrt_inspect_config_set_inspect_device_profile_mode(nrtConfig_,
                                                           NRT_INSPECT_DEVICE_PROFILE_MODE_SESSION);
        NPROF_INFO("  Enabled NRT activity: device_profile (from PRIVATEUSE1_DRIVER)");
        break;
      case libkineto::ActivityType::CPU_OP:
        nrt_inspect_config_set_activity(nrtConfig_, "cpu_util", true);
        NPROF_INFO("  Enabled NRT activity: cpu_util (from CPU_OP)");
        break;
      default:
        NPROF_INFO("  Ignoring unsupported activity type:", static_cast<int>(activityType));
        break;
    }
  }
}

void NeuronActivityProfilerSession::applyCustomConfig() {
  if (!config_ || !nrtConfig_) {
    return;
  }

  const std::string& customConfig = config_->getCustomConfig();
  if (customConfig.empty()) {
    NPROF_INFO("  No custom config provided");
    return;
  }

  NPROF_INFO("  Custom config:", customConfig);

  static const auto registry = createNeuronConfigRegistry();
  torch_neuronx::profiler::NConfigParser::parseAndApply(customConfig, registry, nrtConfig_);
}

NeuronActivityProfilerSession::~NeuronActivityProfilerSession() {
  NPROF_INFO("Session DESTROYED");
  if (isActive_) {
    // Best-effort cleanup - stop profiling before destroying config
    nrt_inspect_stop();
    isActive_ = false;
  }
  cleanupNrtConfig();
}

void NeuronActivityProfilerSession::cleanupNrtConfig() {
  if (nrtConfig_) {
    nrt_inspect_config_free(nrtConfig_);
    nrtConfig_ = nullptr;
  }
}

void NeuronActivityProfilerSession::start() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (isActive_) {
    NPROF_INFO("Already started, ignoring duplicate start()");
    return;
  }

  NPROF_INFO("Starting profiler session..");

  // Neuron runtime will be initialized at the first device access, say
  // but the profiler session can be started before that for example:
  // ```cpp
  //.     with pytorch.profiler.profile(activities=..) as prof:
  //        torch.randn(10, 5, device="neuron")
  //        ...
  // ```
  // But profiling does not work for neuron unless NRT is initialized.
  // So we initialize runtime if it is not initialized.
  torch_neuronx::maybe_lazy_init();

  if (!nrtConfig_) {
    initNrtConfig();
  }

  if (!nrtConfig_) {
    NPROF_ERROR(errors_, "Cannot start: NRT config not initialized");
    return;
  }

  profileStartTime_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now().time_since_epoch())
                          .count();

  NRT_STATUS status = nrt_inspect_begin_with_options(nrtConfig_);

  if (status != NRT_SUCCESS) {
    NPROF_ERROR(errors_, "Failed to start NRT profiling:", std::to_string(status));
    return;
  }

  isActive_ = true;
  NPROF_INFO("Session started.");
}

void NeuronActivityProfilerSession::stop() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!isActive_) {
    NPROF_INFO("Session not active; nothing to stop");
    return;
  }

  profileEndTime_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();

  // Stop profiling first - this writes trace data to the output directory
  NRT_STATUS status = nrt_inspect_stop();
  if (status != NRT_SUCCESS) {
    NPROF_ERROR(errors_, "Failed to stop NRT profiling:", std::to_string(status));
  }

  isActive_ = false;
  NPROF_INFO("NRT profiling STOPPED");
}

std::vector<std::string> NeuronActivityProfilerSession::errors() {
  std::lock_guard<std::mutex> lock(mutex_);
  return errors_;
}

void NeuronActivityProfilerSession::processTrace(libkineto::ActivityLogger& logger) {
  NPROF_INFO("Process trace `sync` called.");
}

void NeuronActivityProfilerSession::processTrace(
    libkineto::ActivityLogger& logger, libkineto::getLinkedActivityCallback /*getLinkedActivity*/,
    int64_t captureWindowStartTime, int64_t captureWindowEndTime) {
  NPROF_INFO("Capture window:", captureWindowStartTime, "to", captureWindowEndTime);
}

std::unique_ptr<libkineto::DeviceInfo> NeuronActivityProfilerSession::getDeviceInfo() {
  NPROF_INFO("getDeviceInfo called");
  // FIXME:Return nullptr for now - minimal implementation
  return nullptr;
}

std::vector<libkineto::ResourceInfo> NeuronActivityProfilerSession::getResourceInfos() {
  NPROF_INFO("getResourceInfos called");
  // FIXME: Return empty vector for now - minimal implementation
  return {};
}

std::unique_ptr<libkineto::CpuTraceBuffer> NeuronActivityProfilerSession::getTraceBuffer() {
  NPROF_INFO("getTraceBuffer called");
  // FIXME: Return nullptr for now - minimal implementation
  return nullptr;
}

}  // namespace at::neuron
