import unittest
from unittest import mock

from accvlab_build_config.helpers import build_utils


class CudaArchSelectionTest(unittest.TestCase):
    def _mock_supported_architectures(self, supported_architectures):
        return mock.patch.object(
            build_utils,
            "_detect_nvcc_supported_architectures",
            return_value=supported_architectures,
        )

    def test_exact_supported_architecture_uses_real_target(self):
        with self._mock_supported_architectures(["80", "90", "100"]):
            selection = build_utils.select_cuda_architectures_for_nvcc(["90"])

        self.assertEqual(selection.architectures, ["90"])
        self.assertEqual(selection.ptx_architectures, [])

    def test_unsupported_hole_uses_base_ptx_not_nearby_real(self):
        with self._mock_supported_architectures(["100", "120"]):
            selection = build_utils.select_cuda_architectures_for_nvcc(["103"])

        self.assertEqual(selection.architectures, [])
        self.assertEqual(selection.ptx_architectures, ["100"])

    def test_future_gpu_uses_supported_base_ptx_below_detection(self):
        with self._mock_supported_architectures(["80", "90", "100", "103", "120"]):
            selection = build_utils.select_cuda_architectures_for_nvcc(["121"])

        self.assertEqual(selection.architectures, [])
        self.assertEqual(selection.ptx_architectures, ["120"])

    def test_unsupported_detection_uses_greatest_supported_ptx_without_base(self):
        with self._mock_supported_architectures(["75", "86", "89"]):
            selection = build_utils.select_cuda_architectures_for_nvcc(["88"])

        self.assertEqual(selection.architectures, [])
        self.assertEqual(selection.ptx_architectures, ["86"])

    def test_unsupported_detection_without_lower_support_remains_unchanged(self):
        with self._mock_supported_architectures(["60", "70"]):
            selection = build_utils.select_cuda_architectures_for_nvcc(["50"])

        self.assertEqual(selection.architectures, ["50"])
        self.assertEqual(selection.ptx_architectures, [])

    def test_mixed_exact_and_unsupported_architectures_preserve_order(self):
        with self._mock_supported_architectures(["80", "90", "100", "120"]):
            selection = build_utils.select_cuda_architectures_for_nvcc(["90", "103"])

        self.assertEqual(selection.architectures, ["90"])
        self.assertEqual(selection.ptx_architectures, ["100"])

    def test_duplicate_ptx_fallbacks_are_deduplicated(self):
        with self._mock_supported_architectures(["100", "120"]):
            selection = build_utils.select_cuda_architectures_for_nvcc(["103", "103"])

        self.assertEqual(selection.architectures, [])
        self.assertEqual(selection.ptx_architectures, ["100"])

    def test_no_detected_nvcc_architectures_returns_input_unchanged(self):
        with self._mock_supported_architectures([]):
            selection = build_utils.select_cuda_architectures_for_nvcc(["103"])

        self.assertEqual(selection.architectures, ["103"])
        self.assertEqual(selection.ptx_architectures, [])

    def test_format_torch_cuda_arch_list_converts_compact_numbers(self):
        self.assertEqual(build_utils.format_torch_cuda_arch_list(["90"]), "9.0")
        self.assertEqual(build_utils.format_torch_cuda_arch_list(["103"]), "10.3")
        self.assertEqual(build_utils.format_torch_cuda_arch_list(["120a"]), "12.0a")
        self.assertEqual(
            build_utils.format_torch_cuda_arch_list(["90", "103"]),
            "9.0;10.3",
        )

    def test_resolve_torch_cuda_arch_list_uses_env_override(self):
        with mock.patch.dict(
            "os.environ",
            {"TORCH_CUDA_ARCH_LIST": "8.0"},
            clear=False,
        ):
            resolved = build_utils.resolve_torch_cuda_arch_list(
                {"cuda_available": True, "gpu_architectures": ["90"]}
            )
        self.assertEqual(resolved, "8.0")

    def test_resolve_torch_cuda_arch_list_uses_detected_real_arch(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self._mock_supported_architectures(["80", "90", "100"]):
                resolved = build_utils.resolve_torch_cuda_arch_list(
                    {"cuda_available": True, "gpu_architectures": ["90"]}
                )
        self.assertEqual(resolved, "9.0")

    def test_resolve_torch_cuda_arch_list_adds_ptx_for_unsupported_arch(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self._mock_supported_architectures(["100", "120"]):
                resolved = build_utils.resolve_torch_cuda_arch_list(
                    {"cuda_available": True, "gpu_architectures": ["103"]}
                )
        self.assertEqual(resolved, "10.0+PTX")

    def test_explicit_custom_architectures_are_not_rewritten(self):
        with mock.patch.object(
            build_utils,
            "select_cuda_architectures_for_nvcc",
            side_effect=AssertionError("unexpected selector call"),
        ):
            config = {
                "CPP_STANDARD": "c++17",
                "OPTIMIZE_LEVEL": 3,
                "USE_FAST_MATH": False,
                "DEBUG_BUILD": False,
                "ENABLE_PROFILING": False,
                "VERBOSE_BUILD": False,
                "CUSTOM_CUDA_ARCHS": ["103"],
            }
            cuda_info = {"cuda_available": True, "gpu_architectures": ["103"]}

            flags = build_utils.get_compile_flags(config, cuda_info)

        self.assertIn("-gencode=arch=compute_103,code=sm_103", flags["nvcc"])
        self.assertNotIn("-gencode=arch=compute_100,code=compute_100", flags["nvcc"])


if __name__ == "__main__":
    unittest.main()
