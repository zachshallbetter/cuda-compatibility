from __future__ import annotations

import unittest

from cuda_legacy_compat.profiles import ProfileError, audit_profile, validate_profile


def profile(minimum: list[int], host_toolkit_minimum: str | None = None) -> dict:
    return {
        "schema_version": 1,
        "id": "fixture",
        "name": "Fixture profile",
        "source": "fixture://profile",
        "build": {"cuda_version": "13.0", "host_toolkit_minimum": host_toolkit_minimum},
        "gpu": {"minimum_compute_capability": minimum},
    }


def inventory(capability: str = "5.2", nvcc_status: str = "not_found", nvcc_version: str | None = None) -> dict:
    major, minor = (int(part) for part in capability.split("."))
    return {
        "gpu_inventory": {
            "status": "available",
            "devices": [
                {"index": 0, "compute_capability": capability, "compute_capability_parts": [major, minor]}
            ],
        },
        "nvcc": {"status": nvcc_status, "version": nvcc_version},
    }


class ProfileAuditTests(unittest.TestCase):
    def test_reports_old_gpu_incompatible_even_when_host_toolkit_is_not_required(self) -> None:
        result = audit_profile(profile([7, 5]), inventory())
        self.assertEqual(result["status"], "incompatible")
        self.assertEqual(result["toolkit"]["status"], "not_required_for_runtime")
        self.assertEqual(result["issues"][0]["code"], "gpu_compute_capability_below_minimum")

    def test_accepts_supported_gpu_for_package_profile_without_host_toolkit(self) -> None:
        result = audit_profile(profile([7, 5]), inventory("8.6"))
        self.assertEqual(result["status"], "compatible")
        self.assertEqual(result["toolkit"]["status"], "not_required_for_runtime")

    def test_reports_missing_required_compiler_as_unknown(self) -> None:
        result = audit_profile(profile([5, 2], host_toolkit_minimum="13.0"), inventory())
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["issues"][0]["code"], "required_host_toolkit_unavailable")

    def test_reports_old_required_compiler_as_incompatible(self) -> None:
        result = audit_profile(
            profile([5, 2], host_toolkit_minimum="13.0"),
            inventory(nvcc_status="available", nvcc_version="12.4"),
        )
        self.assertEqual(result["status"], "incompatible")
        self.assertEqual(result["toolkit"]["status"], "incompatible")

    def test_compares_toolkit_patch_versions(self) -> None:
        result = audit_profile(
            profile([5, 2], host_toolkit_minimum="12.9.1"),
            inventory(nvcc_status="available", nvcc_version="12.9"),
        )
        self.assertEqual(result["status"], "incompatible")
        self.assertEqual(result["issues"][0]["code"], "host_toolkit_below_requirement")

    def test_rejects_boolean_compute_capability_parts(self) -> None:
        malformed = profile([5, 2])
        malformed["gpu"]["minimum_compute_capability"] = [True, 2]
        with self.assertRaises(ProfileError):
            validate_profile(malformed)

    def test_unknown_gpu_inventory_does_not_claim_compatibility(self) -> None:
        result = audit_profile(profile([5, 2]), {"gpu_inventory": {"status": "not_found", "devices": []}})
        self.assertEqual(result["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
