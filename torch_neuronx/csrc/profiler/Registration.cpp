#include <libkineto.h>

#include <mutex>

#include "LogWrapper.h"
#include "Profiler.h"

namespace at::neuron {

namespace {
std::once_flag g_registration_flag;
bool g_profiler_registered = false;
}  // namespace

bool registerNeuronProfiler() {
  std::call_once(g_registration_flag, []() {
    // Check if libkineto's main profiler is registered
    if (!libkineto::api().isProfilerRegistered()) {
      NPROF_INFO("libkineto profiler not registered.",
                 "This may be expected if PyTorch profiling is not initialized yet.");
    }

    // Register the Neuron profiler factory
    libkineto::api().registerProfilerFactory([]() -> std::unique_ptr<libkineto::IActivityProfiler> {
      NPROF_INFO("Factory called - creating profiler instance");
      return std::make_unique<NeuronActivityProfiler>();
    });

    g_profiler_registered = true;
    NPROF_INFO("Registration complete");
  });

  if (g_profiler_registered) {
    return true;
  }

  // If we get here, registration was attempted but failed
  NPROF_INFO("Already registered");
  return true;
}

}  // namespace at::neuron
