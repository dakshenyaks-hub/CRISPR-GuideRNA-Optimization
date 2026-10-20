import sys
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mock pytest if not installed
try:
    import pytest
except ImportError:
    class MockPytest:
        @staticmethod
        def raises(expected_exception):
            class ExceptionContext:
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    if exc_type is None:
                        raise AssertionError(f"Expected {expected_exception} but no exception was raised.")
                    return issubclass(exc_type, expected_exception)
            return ExceptionContext()
    sys.modules['pytest'] = MockPytest()

# Import test modules
import tests.test_cfd_score as t1
import tests.test_cfd_e2e as t2
import tests.test_pipeline_controller as t3
import tests.test_ncbi_sequence_provider as t4
import tests.test_optimization_lab as t5
import tests.test_validation_dashboard as t6
import tests.test_dynamic_pipeline as t7

def run_test_class(cls):
    obj = cls()
    if hasattr(obj, 'setUp'):
        obj.setUp()
    methods = [m for m in dir(obj) if m.startswith('test_') and callable(getattr(obj, m))]
    for m in methods:
        getattr(obj, m)()
    if hasattr(obj, 'tearDown'):
        obj.tearDown()

def main():
    print("=== Running Unit Tests in tests/test_cfd_score.py ===")
    test_classes = [
        t1.TestMismatchTable,
        t1.TestPAMTable,
        t1.TestCalcCFDSite,
        t1.TestPAMKeyExtraction,
        t1.TestIntendedLocusExclusion,
        t1.TestPerfectMatchRetention,
        t1.TestCensoredHandling,
        t1.TestZeroHitDistinction,
        t1.TestNoRawCountSubstitution,
        t1.TestUnsupportedPAM,
        t1.TestSchemaPreservation,
        t1.TestReproducibility,
        t1.TestStep10AVarianceAndExclusionRegression,
    ]
    for cls in test_classes:
        run_test_class(cls)
        print(f"  [PASS] {cls.__name__}")

    print("\n=== Running E2E Tests in tests/test_cfd_e2e.py ===")
    t2.test_actual_grch38_extraction()
    print("  [PASS] test_actual_grch38_extraction")
    t2.test_exclude_on_target_unresolved()
    print("  [PASS] test_exclude_on_target_unresolved")

    print("\n=== Running Pipeline Controller Baseline Tests in tests/test_pipeline_controller.py ===")
    t3.test_supported_genes()
    t3.test_frozen_baseline_integrity()
    t3.test_gene_overview_metrics()
    t3.test_pareto_front_records()
    t3.test_selected_guides_records()
    t3.test_sensitivity_summary()
    t3.test_full_payload_generation()
    print("  [PASS] PipelineController 100% frozen baseline fidelity")

    print("\n=== Running NCBI Sequence Provider Tests in tests/test_ncbi_sequence_provider.py ===")
    run_test_class(t4.TestGeneSymbolValidation)
    print("  [PASS] TestGeneSymbolValidation")
    t4.TestDataStructures().test_metadata_to_dict()
    print("  [PASS] TestDataStructures.test_metadata_to_dict")

    print("\n=== Running Mathematical Formulation & Optimization Lab Tests in tests/test_optimization_lab.py ===")
    run_test_class(t5.TestOptimizationLabStructure)
    print("  [PASS] TestOptimizationLabStructure (KaTeX, Sequence, DOM elements)")
    run_test_class(t5.TestOptimizationLabCalculations)
    print("  [PASS] TestOptimizationLabCalculations (Weight sweeps, Baseline fidelity, Pareto optimality)")

    print("\n=== Running Validation & Robustness Dashboard Tests in tests/test_validation_dashboard.py ===")
    run_test_class(t6.TestDashboardNavigationAndStructure)
    print("  [PASS] TestDashboardNavigationAndStructure (5 Tabs, Content Relocation, Section A & B)")
    run_test_class(t6.TestValidationDataIntegrity)
    print("  [PASS] TestValidationDataIntegrity (4-Gene Authoritative Baseline & Honest Statuses)")

    print("\n=== Running Dynamic Gene Pipeline Tests in tests/test_dynamic_pipeline.py ===")
    run_test_class(t7.TestDynamicPipelineModule)
    print("  [PASS] TestDynamicPipelineModule (PAM generation, DeepSpCas9, isolation, resource checks)")
    run_test_class(t7.TestDashboardDynamicIntegration)
    print("  [PASS] TestDashboardDynamicIntegration (API status, FASTA input, baseline preservation)")

    print("\nALL UNIT, E2E, PIPELINE CONTROLLER, NCBI PROVIDER, OPTIMIZATION LAB, VALIDATION DASHBOARD, AND DYNAMIC PIPELINE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
