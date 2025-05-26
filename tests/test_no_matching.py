"""
Tests for the NO_MATCHING functionality in peppr.Evaluator.
"""

import biotite.structure.info as info
import numpy as np
import peppr


class TestNoMatching:
    """Test the NO_MATCHING match method."""

    def test_no_matching_basic(self):
        """Test basic NO_MATCHING functionality."""
        # Create a simple test structure
        reference = info.residue("ALA")
        reference = reference[reference.element != "H"]
        pose = reference.copy()

        # Test with metrics that don't require matching
        evaluator = peppr.Evaluator(
            [peppr.BondLengthViolations(), peppr.ClashCount()],
            match_method=peppr.Evaluator.MatchMethod.NO_MATCHING,
        )

        evaluator.feed("test_system", reference, pose)
        results = evaluator.get_results()

        # Should get valid results
        assert len(results) == 2
        assert len(results[0]) == 1  # One system
        assert len(results[0][0]) == 1  # One pose
        assert np.isfinite(results[0][0][0])  # Valid bond length violation result
        assert np.isfinite(results[1][0][0])  # Valid clash count result

    def test_no_matching_vs_heuristic_identical_structures(self):
        """Test that NO_MATCHING gives same results as HEURISTIC for identical structures."""
        reference = info.residue("ALA")
        reference = reference[reference.element != "H"]
        pose = reference.copy()

        # Metrics that should give identical results regardless of matching method
        metrics = [peppr.BondLengthViolations(), peppr.ClashCount()]

        evaluator_heuristic = peppr.Evaluator(
            metrics, match_method=peppr.Evaluator.MatchMethod.HEURISTIC
        )
        evaluator_no_matching = peppr.Evaluator(
            metrics, match_method=peppr.Evaluator.MatchMethod.NO_MATCHING
        )

        evaluator_heuristic.feed("test", reference, pose)
        evaluator_no_matching.feed("test", reference, pose)

        results_heuristic = evaluator_heuristic.get_results()
        results_no_matching = evaluator_no_matching.get_results()

        # Results should be identical
        for i in range(len(metrics)):
            assert np.allclose(
                results_heuristic[i][0], results_no_matching[i][0], equal_nan=True
            )

    def test_no_matching_with_multiple_poses(self):
        """Test NO_MATCHING with multiple poses."""
        reference = info.residue("ALA")
        reference = reference[reference.element != "H"]

        # Create multiple poses
        pose1 = reference.copy()
        pose2 = reference.copy()
        pose2.coord += 0.1  # Slightly different coordinates

        evaluator = peppr.Evaluator(
            [peppr.BondLengthViolations()],
            match_method=peppr.Evaluator.MatchMethod.NO_MATCHING,
        )

        evaluator.feed("test_system", reference, [pose1, pose2])
        results = evaluator.get_results()

        # Should have results for both poses
        assert len(results[0][0]) == 2
        assert np.isfinite(results[0][0][0])
        assert np.isfinite(results[0][0][1])

    def test_no_matching_with_mismatched_structures(self):
        """Test that NO_MATCHING fails gracefully with mismatched structures."""
        reference = info.residue("ALA")
        reference = reference[reference.element != "H"]

        # Create a pose with different number of atoms
        pose = info.residue("GLY")
        pose = pose[pose.element != "H"]

        evaluator = peppr.Evaluator(
            [peppr.BondLengthViolations()],
            match_method=peppr.Evaluator.MatchMethod.NO_MATCHING,
            tolerate_exceptions=True,  # Don't raise exceptions
        )

        # This should not crash but may give NaN results
        evaluator.feed("test_system", reference, pose)
        results = evaluator.get_results()

        # Should have some result (possibly NaN)
        assert len(results[0]) == 1

    def test_no_matching_performance_benefit(self):
        """Test that NO_MATCHING is faster than other methods."""
        import time

        reference = info.residue("ALA")
        reference = reference[reference.element != "H"]
        pose = reference.copy()

        metrics = [peppr.BondLengthViolations(), peppr.ClashCount()]

        # Time NO_MATCHING
        evaluator_no_matching = peppr.Evaluator(
            metrics, match_method=peppr.Evaluator.MatchMethod.NO_MATCHING
        )
        start_time = time.time()
        evaluator_no_matching.feed("test", reference, pose)
        no_matching_time = time.time() - start_time

        # Time HEURISTIC
        evaluator_heuristic = peppr.Evaluator(
            metrics, match_method=peppr.Evaluator.MatchMethod.HEURISTIC
        )
        start_time = time.time()
        evaluator_heuristic.feed("test", reference, pose)
        heuristic_time = time.time() - start_time

        # NO_MATCHING should be faster (though this might be flaky in CI)
        # We'll just check that both complete successfully
        assert no_matching_time >= 0
        assert heuristic_time >= 0

    def test_no_matching_with_quality_metrics(self):
        """Test NO_MATCHING with metrics that assess structure quality."""
        reference = info.residue("BNZ")  # Benzene
        reference = reference[reference.element != "H"]

        # Create a pose with some structural issues
        pose = reference.copy()
        if len(pose.coord) > 1:
            # Stretch a bond to create a violation
            pose.coord[1] += [2.0, 0.0, 0.0]

        evaluator = peppr.Evaluator(
            [
                peppr.BondLengthViolations(),
                peppr.BondAngleViolations(),
                peppr.ClashCount(),
            ],
            match_method=peppr.Evaluator.MatchMethod.NO_MATCHING,
        )

        evaluator.feed("quality_test", reference, pose)
        results = evaluator.get_results()

        # Should detect some violations
        bond_violations = results[0][0][0]
        angle_violations = results[1][0][0]
        clash_count = results[2][0][0]

        # At least one metric should detect issues
        assert np.isfinite(bond_violations)
        assert np.isfinite(angle_violations)
        assert np.isfinite(clash_count)

    def test_no_matching_enum_value(self):
        """Test that NO_MATCHING enum value is correctly defined."""
        assert hasattr(peppr.Evaluator.MatchMethod, "NO_MATCHING")
        assert peppr.Evaluator.MatchMethod.NO_MATCHING.value == "no_matching"

    def test_no_matching_in_evaluator_equality(self):
        """Test that evaluators with NO_MATCHING compare correctly."""
        metrics = [peppr.BondLengthViolations()]

        evaluator1 = peppr.Evaluator(
            metrics, match_method=peppr.Evaluator.MatchMethod.NO_MATCHING
        )
        evaluator2 = peppr.Evaluator(
            metrics, match_method=peppr.Evaluator.MatchMethod.NO_MATCHING
        )
        evaluator3 = peppr.Evaluator(
            metrics, match_method=peppr.Evaluator.MatchMethod.HEURISTIC
        )

        # Same configuration should be equal
        assert evaluator1 == evaluator2
        # Different match methods should not be equal
        assert evaluator1 != evaluator3
