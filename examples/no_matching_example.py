#!/usr/bin/env python3
"""
Example demonstrating the NO_MATCHING option in peppr.Evaluator.

This example shows:
1. When to use NO_MATCHING vs other matching methods
2. Performance benefits of skipping matching
3. Metrics that work well without matching
"""

import time
import biotite.structure.info as info
import peppr


def create_test_system():
    """Create a test system with multiple chains."""
    # Create a small protein-ligand system
    protein = info.residue("ALA")
    protein = protein[protein.element != "H"]  # Remove hydrogens
    protein.chain_id[:] = "A"
    protein.hetero[:] = False

    ligand = info.residue("BNZ")  # Benzene
    ligand = ligand[ligand.element != "H"]  # Remove hydrogens
    ligand.chain_id[:] = "L"
    ligand.hetero[:] = True

    # Combine into a system
    system = protein + ligand
    return system


def benchmark_matching_methods():
    """Compare performance of different matching methods."""
    print("=== Performance Comparison ===\n")

    # Create test system
    reference = create_test_system()
    pose = reference.copy()  # Perfect alignment case

    # Metrics that don't require matching
    metrics_no_matching_needed = [
        peppr.BondLengthViolations(),
        peppr.BondAngleViolations(),
        peppr.ClashCount(),
    ]

    # Metrics that benefit from matching
    metrics_matching_needed = [
        peppr.MonomerRMSD(threshold=2.0),
        peppr.GlobalLDDTScore(),
    ]

    methods_to_test = [
        (peppr.Evaluator.MatchMethod.HEURISTIC, "HEURISTIC"),
        (peppr.Evaluator.MatchMethod.NO_MATCHING, "NO_MATCHING"),
    ]

    print("Testing with metrics that DON'T require matching:")
    print("(BondLengthViolations, BondAngleViolations, ClashCount)\n")

    for method, method_name in methods_to_test:
        evaluator = peppr.Evaluator(metrics_no_matching_needed, match_method=method)

        # Time the evaluation
        start_time = time.time()
        evaluator.feed("test_system", reference, pose)
        end_time = time.time()

        results = evaluator.get_results()

        print(f"{method_name:12} - Time: {(end_time - start_time)*1000:.2f}ms")
        for i, metric in enumerate(evaluator.metrics):
            print(f"  {metric.name}: {results[i][0][0]:.4f}")
        print()

    print("\nTesting with metrics that DO require matching:")
    print("(MonomerRMSD, GlobalLDDTScore)\n")

    for method, method_name in methods_to_test:
        evaluator = peppr.Evaluator(metrics_matching_needed, match_method=method)

        # Time the evaluation
        start_time = time.time()
        evaluator.feed("test_system", reference, pose)
        end_time = time.time()

        results = evaluator.get_results()

        print(f"{method_name:12} - Time: {(end_time - start_time)*1000:.2f}ms")
        for i, metric in enumerate(evaluator.metrics):
            print(f"  {metric.name}: {results[i][0][0]:.4f}")
        print()


def demonstrate_use_cases():
    """Show different use cases for NO_MATCHING."""
    print("=== Use Cases for NO_MATCHING ===\n")

    reference = create_test_system()

    print("1. Quality assessment metrics (don't need reference comparison):")
    quality_evaluator = peppr.Evaluator(
        [peppr.BondLengthViolations(), peppr.BondAngleViolations(), peppr.ClashCount()],
        match_method=peppr.Evaluator.MatchMethod.NO_MATCHING,
    )

    # Create a pose with some bond length issues
    pose_with_issues = reference.copy()
    # Artificially stretch a bond (this would normally be detected)
    if len(pose_with_issues.coord) > 1:
        pose_with_issues.coord[1] += [2.0, 0.0, 0.0]  # Move an atom

    quality_evaluator.feed("quality_check", reference, pose_with_issues)
    results = quality_evaluator.get_results()

    print("  Quality metrics (higher values = more problems):")
    for i, metric in enumerate(quality_evaluator.metrics):
        print(f"    {metric.name}: {results[i][0][0]:.4f}")
    print()

    print("2. Pre-aligned structures (matching already done externally):")
    print("   When you've already aligned structures using external tools,")
    print("   NO_MATCHING avoids redundant computation.\n")

    print("3. Large-scale screening:")
    print("   For high-throughput analysis where speed is critical and")
    print("   structures are pre-processed to have consistent atom ordering.\n")


def main():
    """Run the example."""
    print("peppr NO_MATCHING Example")
    print("=" * 50)
    print()

    benchmark_matching_methods()
    demonstrate_use_cases()

    print("=== Summary ===")
    print("Use NO_MATCHING when:")
    print("• Evaluating quality metrics that don't need reference comparison")
    print("• Structures are already properly aligned")
    print("• Speed is critical and atom ordering is guaranteed to be correct")
    print("• Doing large-scale screening with pre-processed structures")
    print()
    print("⚠️  Warning: NO_MATCHING assumes reference and pose have identical")
    print("   atom ordering. Incorrect ordering will give meaningless results!")


if __name__ == "__main__":
    main()
