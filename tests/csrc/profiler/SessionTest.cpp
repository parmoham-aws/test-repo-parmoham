#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <libkineto.h>

#include <thread>
#include <vector>

#include "tests/csrc/mocks/MockNRT.h"
#include "torch_neuronx/csrc/profiler/Session.h"

using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::StrEq;

namespace at::neuron {
namespace {

// Fake nrt_inspect_config for testing
struct FakeNrtInspectConfig {
  int dummy;
};

class SessionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_session_ = std::make_unique<torch_neuronx::testing::MockNRTSession>();
    mock_nrt_ = torch_neuronx::testing::MockNRT::GetInstance();
  }

  void TearDown() override { mock_session_.reset(); }

  // Helper: expects successful config allocation sequence.
  void expectConfigAllocation() {
    EXPECT_CALL(*mock_nrt_, nrt_inspect_config_allocate(_))
        .WillOnce(DoAll(SetArgPointee<0>(reinterpret_cast<nrt_inspect_config_t*>(&fake_config_)),
                        Return(NRT_SUCCESS)));
    EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_defaults(_)).WillOnce(Return(NRT_SUCCESS));
    EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_enable_inspect(_, true))
        .WillOnce(Return(NRT_SUCCESS));
  }

  // Helper: expect activity type setup calls
  void expectActivityTypeSetup() {
    EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("all"), false))
        .WillOnce(Return(NRT_SUCCESS));
  }

  // Helper: expect full successful start sequence
  void expectSuccessfulStart() {
    expectConfigAllocation();
    expectActivityTypeSetup();
    EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  }

  // Helper: expect cleanup
  void expectCleanup() {
    EXPECT_CALL(*mock_nrt_, nrt_inspect_config_free(_)).WillOnce(Return(NRT_SUCCESS));
  }

  // Helper: expect stop
  void expectStop() { EXPECT_CALL(*mock_nrt_, nrt_inspect_stop()).WillOnce(Return(NRT_SUCCESS)); }

  std::unique_ptr<torch_neuronx::testing::MockNRTSession> mock_session_;
  torch_neuronx::testing::MockNRT* mock_nrt_;
  FakeNrtInspectConfig fake_config_;
};

TEST_F(SessionTest, StartStopSuccess) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  expectSuccessfulStart();
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  EXPECT_TRUE(session.errors().empty());
}

// Ensues that if system_profile is in the list of activityTypes, then it does not call
// the device_profile.
TEST_F(SessionTest, StartWithActivityTypeRuntime) {
  std::set<libkineto::ActivityType> activityTypes = {libkineto::ActivityType::PRIVATEUSE1_RUNTIME};
  libkineto::Config config;

  expectConfigAllocation();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("all"), false))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("system_profile"), true))
      .WillOnce(Return(NRT_SUCCESS));

  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("device_profile"), true))
      .Times(
          0);  // makes sure that device profiling is not set as activityTypes does not mention it.

  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  EXPECT_TRUE(session.errors().empty());
}

TEST_F(SessionTest, StartWithActivityTypeDriver) {
  std::set<libkineto::ActivityType> activityTypes = {libkineto::ActivityType::PRIVATEUSE1_DRIVER};
  libkineto::Config config;

  expectConfigAllocation();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("all"), false))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("device_profile"), true))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_inspect_device_profile_mode(
                              _, NRT_INSPECT_DEVICE_PROFILE_MODE_SESSION))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("system_profile"), true))
      .Times(0);
  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  EXPECT_TRUE(session.errors().empty());
}

TEST_F(SessionTest, StartWithActivityTypeCpuOp) {
  std::set<libkineto::ActivityType> activityTypes = {libkineto::ActivityType::CPU_OP};
  libkineto::Config config;

  expectConfigAllocation();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("all"), false))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("cpu_util"), true))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("system_profile"), true))
      .Times(0);
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("device_profile"), true))
      .Times(0);
  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  EXPECT_TRUE(session.errors().empty());
}

TEST_F(SessionTest, StartWithMultipleActivityTypes) {
  // All three supported activity types should each enable their corresponding NRT activity.
  std::set<libkineto::ActivityType> activityTypes = {libkineto::ActivityType::PRIVATEUSE1_RUNTIME,
                                                     libkineto::ActivityType::PRIVATEUSE1_DRIVER,
                                                     libkineto::ActivityType::CPU_OP};
  libkineto::Config config;

  expectConfigAllocation();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("all"), false))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("system_profile"), true))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("device_profile"), true))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_inspect_device_profile_mode(
                              _, NRT_INSPECT_DEVICE_PROFILE_MODE_SESSION))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("cpu_util"), true))
      .WillOnce(Return(NRT_SUCCESS));
  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  EXPECT_TRUE(session.errors().empty());
}

TEST_F(SessionTest, ErrorsReturnsAccumulatedErrors) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_allocate(_)).WillOnce(Return(NRT_FAILURE));

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();

  std::vector<std::string> errors = session.errors();
  EXPECT_FALSE(errors.empty());
}

TEST_F(SessionTest, StartInspectBeginFailure) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  expectConfigAllocation();
  expectActivityTypeSetup();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_FAILURE));
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();

  EXPECT_FALSE(session.errors().empty());
}

TEST_F(SessionTest, StopInspectStopFailure) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  expectSuccessfulStart();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_stop()).WillOnce(Return(NRT_FAILURE));
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  EXPECT_FALSE(session.errors().empty());
}

// This establishes that starting another session when one is
// active is possible today.
TEST_F(SessionTest, StartDuplicateIgnored) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  // Only expect one start sequence
  expectSuccessfulStart();
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.start();  // Second start should be ignored
  session.stop();

  EXPECT_TRUE(session.errors().empty());
}

TEST_F(SessionTest, StopWhenNotActive) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  // No NRT calls expected - session never started

  NeuronActivityProfilerSession session(activityTypes, config);
  session.stop();  // Should be no-op

  EXPECT_TRUE(session.errors().empty());
}

TEST_F(SessionTest, DestructorCleansUpActiveSession) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  expectSuccessfulStart();
  // Destructor should call stop and cleanup
  expectStop();
  expectCleanup();

  {
    NeuronActivityProfilerSession session(activityTypes, config);
    session.start();
    // Session destroyed without explicit stop()
  }
}

TEST_F(SessionTest, DestructorDoesNotStopInactiveSession) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  // No NRT calls expected for inactive session destruction

  {
    NeuronActivityProfilerSession session(activityTypes, config);
  }
}

TEST_F(SessionTest, StartWithUnsupportedActivityType) {
  // Use an activity type that's not mapped
  std::set<libkineto::ActivityType> activityTypes = {libkineto::ActivityType::CUDA_RUNTIME};
  libkineto::Config config;

  expectConfigAllocation();
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_activity(_, StrEq("all"), false))
      .WillOnce(Return(NRT_SUCCESS));
  // No additional activity set calls - unsupported type is just logged
  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  EXPECT_TRUE(session.errors().empty());
}

TEST_F(SessionTest, StartWithCustomConfigMaxEventsPerNc) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;
  std::string customConfig = "max_events_per_nc:100000";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  expectConfigAllocation();
  expectActivityTypeSetup();

  // Expect the custom config NRT call
  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_sys_trace_max_events_per_nc(_, 100000))
      .WillOnce(Return(NRT_SUCCESS));

  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  EXPECT_TRUE(session.errors().empty());
}

TEST_F(SessionTest, StartWithCustomConfigProfileOutputDir) {
  // Custom config profile_output_dir should call nrt_inspect_config_set_output_dir.
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;
  std::string customConfig = "profile_output_dir:/tmp/neuron_traces";
  config.handleOption("CUSTOM_CONFIG", customConfig);

  expectConfigAllocation();
  expectActivityTypeSetup();

  EXPECT_CALL(*mock_nrt_, nrt_inspect_config_set_output_dir(_, StrEq("/tmp/neuron_traces")))
      .WillOnce(Return(NRT_SUCCESS));

  EXPECT_CALL(*mock_nrt_, nrt_inspect_begin_with_options(_)).WillOnce(Return(NRT_SUCCESS));
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);
  session.start();
  session.stop();

  EXPECT_TRUE(session.errors().empty());
}

TEST_F(SessionTest, AccessorsReturnDefaults) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  NeuronActivityProfilerSession session(activityTypes, config);

  EXPECT_EQ(session.getDeviceInfo(), nullptr);
  EXPECT_TRUE(session.getResourceInfos().empty());
  EXPECT_EQ(session.getTraceBuffer(), nullptr);
}

// Concurrent session starts.
TEST_F(SessionTest, ConcurrentStartCalls) {
  std::set<libkineto::ActivityType> activityTypes;
  libkineto::Config config;

  // Only expect one full start sequence
  expectSuccessfulStart();
  expectStop();
  expectCleanup();

  NeuronActivityProfilerSession session(activityTypes, config);

  std::vector<std::thread> threads;
  for (int i = 0; i < 5; ++i) {
    threads.emplace_back([&session]() { session.start(); });
  }

  for (auto& t : threads) {
    t.join();
  }

  session.stop();
  EXPECT_TRUE(session.errors().empty());
}
}  // namespace
}  // namespace at::neuron
