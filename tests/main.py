from test_metrics import run_test_metrics
from test_results import run_validation_tests

def main():
    """ Runs generation and analysis steps back-to-back. """
    print("Running basic metric tests...")
    run_test_metrics()
    print("Basic metric tests completed.\n")
    
    print("Running validation tests against external packages...")
    run_validation_tests()
    print("Validation tests completed.")

if __name__ == "__main__":
    main()