from test_metrics import run_test_metrics
from test_results import run_validation_tests
from test_vis import run_vis_tests
from test_utils import run_utils_test

def main():
    """ Runs generation and analysis steps back-to-back. """
    print("Running basic metric tests...")
    run_test_metrics()
    print("Basic metric tests completed.\n")

    print("Running visualization tests...")
    run_vis_tests()
    print("All visualization tests completed.\n")

    print("Running utility function tests...")
    run_utils_test()
    print("All utility function tests completed.\n")
    
    print("Running validation tests against external packages...")
    run_validation_tests()
    print("Validation tests completed.")

if __name__ == "__main__":
    main()