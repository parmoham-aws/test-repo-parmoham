"""
Integration test for Neuron Profiler with PyTorch profiler.
Verifies end-to-end profiling: NRT trace output + chrome trace export.
"""

import os
import shutil
import tempfile
import unittest

import torch
from torch._C._profiler import _ExperimentalConfig
from torch.profiler import ProfilerActivity, profile, record_function

import torch_neuronx


class TestNeuronProfilerIntegration(unittest.TestCase):
    """Integration test for Neuron profiler with PyTorch profiler API."""

    def setUp(self):
        """Create temp directory for profiler output."""
        self.test_dir = tempfile.mkdtemp(prefix="neuron_profiler_test_")

    def tearDown(self):
        """Clean up temp directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _validate_nrt_output_structure(self, output_dir):
        """Validate that NRT created the expected output files.

        Expected structure:
            {output_dir}/
            └── {instance_id}_pid_{process_id}/
                ├── *_instid_*_vnc_*.ntff    # Trace files per VNC
                ├── neff_*.neff              # Compiled model binaries
                ├── cpu_util.pb              # CPU utilization data
                ├── host_mem.pb              # Host memory data
                ├── ntrace.pb                # System trace events
                └── trace_info.pb            # Trace metadata

        Returns:
            str: Path to the instance subdirectory
        """
        # Find the instance subdirectory (pattern: *_pid_*)
        subdirs = [
            d
            for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and "_pid_" in d
        ]
        self.assertEqual(
            len(subdirs), 1, f"Expected exactly one instance subdirectory, found: {subdirs}"
        )

        instance_dir = os.path.join(output_dir, subdirs[0])
        files = os.listdir(instance_dir)

        # Check all required protobuf files (host_memory:true enables cpu_util and host_mem)
        required_pb_files = ["trace_info.pb", "ntrace.pb", "cpu_util.pb", "host_mem.pb"]
        for pb_file in required_pb_files:
            self.assertIn(pb_file, files, f"Missing required file: {pb_file}")

        # Check for at least one .ntff trace file
        ntff_files = [f for f in files if f.endswith(".ntff")]
        self.assertGreater(len(ntff_files), 0, "No .ntff trace files found")

        # Check for at least one neff file
        neff_files = [f for f in files if f.endswith(".neff")]
        self.assertGreater(len(neff_files), 0, "No .neff files found")

    def test_profiler_end_to_end(self):
        """
        Tests that the torch.profiler.profile() API is able to generate:
        - Neuron runtime and device trace.

        Verifies:
        1. Profiler starts/stops without error
        2. Custom config (host_memory, max_events_per_nc, capture_enabled_for_nc,
          profile_output_dir) works
        3. record_function annotations are captured
        4. NRT creates output in specified profile_output_dir
        5. Chrome trace export produces valid JSON in the same directory
        """
        # Configure profiler with custom options including output directory
        activities = [ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
        exp_config = _ExperimentalConfig(
            custom_profiler_config=f"profile_output_dir:{self.test_dir};host_memory:true;max_events_per_nc:100000;capture_enabled_for_nc:0,1"
        )

        # Run profiling with record_function annotation
        with (
            profile(activities=activities, experimental_config=exp_config) as prof,
            record_function("model_inference"),
        ):
            # Initialize NRT and run operations
            x = torch.randn(10, 5, device="neuron")
            linear = torch.nn.Linear(5, 3).to("neuron")
            _ = linear(x)
            torch.neuron.synchronize()

        # Verify 1: Events were captured
        events = prof.events()
        self.assertIsNotNone(events, "prof.events() returned None")

        # Verify 2: record_function annotation appears in trace
        event_names = [e.name for e in events]
        self.assertIn(
            "model_inference",
            event_names,
            f"'model_inference' not found in events: {event_names[:10]}",
        )

        # Verify 3: NRT created output in specified directory
        self.assertTrue(
            os.path.isdir(self.test_dir), f"Output directory not created at {self.test_dir}"
        )

        # Verify 4: Output directory contains expected NRT trace files
        self._validate_nrt_output_structure(self.test_dir)

        # Verify 5: Chrome trace export to same directory
        trace_file = os.path.join(self.test_dir, "neuron_trace.json")
        prof.export_chrome_trace(trace_file)
        self.assertTrue(os.path.exists(trace_file), f"Chrome trace not created at {trace_file}")
        self.assertGreater(os.path.getsize(trace_file), 0, "Chrome trace file is empty")


if __name__ == "__main__":
    unittest.main()
