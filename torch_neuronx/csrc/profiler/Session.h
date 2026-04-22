#pragma once

#include <ActivityType.h>
#include <Config.h>
#include <IActivityProfiler.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

struct nrt_inspect_config;
typedef struct nrt_inspect_config nrt_inspect_config_t;

namespace at::neuron {
class NeuronActivityProfilerSession : public libkineto::IActivityProfilerSession {
 public:
  explicit NeuronActivityProfilerSession(const std::set<libkineto::ActivityType>& activityTypes,
                                         const libkineto::Config& config);

  ~NeuronActivityProfilerSession() override;

  void start() override;
  void stop() override;
  [[nodiscard]] std::vector<std::string> errors() override;

  void processTrace(libkineto::ActivityLogger& logger) override;

  void processTrace(libkineto::ActivityLogger& logger,
                    libkineto::getLinkedActivityCallback getLinkedActivity,
                    int64_t captureWindowStartTime, int64_t captureWindowEndTime) override;

  std::unique_ptr<libkineto::DeviceInfo> getDeviceInfo() override;
  std::vector<libkineto::ResourceInfo> getResourceInfos() override;
  std::unique_ptr<libkineto::CpuTraceBuffer> getTraceBuffer() override;

 private:
  std::set<libkineto::ActivityType> activityTypes_;

  std::unique_ptr<const libkineto::Config> config_;

  std::vector<std::string> errors_;

  nrt_inspect_config_t* nrtConfig_{nullptr};
  int64_t profileStartTime_{0};
  int64_t profileEndTime_{0};
  bool isActive_{false};

  mutable std::mutex mutex_;

  void initNrtConfig();
  void cleanupNrtConfig();
  void applyActivityTypes();
  void applyCustomConfig();
};
}  // namespace at::neuron
