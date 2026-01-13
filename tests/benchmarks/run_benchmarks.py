#!/usr/bin/env python
"""
ProximA Benchmark Runner

Command-line tool for running performance benchmarks.

Usage:
    python run_benchmarks.py --all
    python run_benchmarks.py --backend
    python run_benchmarks.py --circuit
    python run_benchmarks.py --comparison
    python run_benchmarks.py --output results.json
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ProximA Performance Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all benchmarks
    python run_benchmarks.py --all

    # Run only backend benchmarks
    python run_benchmarks.py --backend

    # Run with specific backends
    python run_benchmarks.py --backend --backends cirq qiskit_aer

    # Save results to file
    python run_benchmarks.py --all --output results.json

    # Generate markdown report
    python run_benchmarks.py --all --report benchmark_report.md
        """
    )
    
    # Benchmark selection
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmark suites"
    )
    parser.add_argument(
        "--backend",
        action="store_true",
        help="Run backend benchmarks"
    )
    parser.add_argument(
        "--circuit",
        action="store_true",
        help="Run circuit operation benchmarks"
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Run backend comparison benchmarks"
    )
    
    # Configuration
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["cirq", "qiskit_aer"],
        help="Backends to benchmark (default: cirq qiskit_aer)"
    )
    parser.add_argument(
        "--qubits",
        nargs="+",
        type=int,
        default=[2, 4, 8, 12, 16],
        help="Qubit counts to test (default: 2 4 8 12 16)"
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1000,
        help="Number of shots per run (default: 1000)"
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Repetitions for timing (default: 5)"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for JSON results"
    )
    parser.add_argument(
        "--report", "-r",
        type=str,
        help="Output file for markdown report"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Default to all if no specific benchmark selected
    if not any([args.all, args.backend, args.circuit, args.comparison]):
        args.all = True
    
    all_results = []
    verbose = not args.quiet
    
    try:
        # Backend benchmarks
        if args.all or args.backend:
            if verbose:
                print("\n" + "=" * 60)
                print("  BACKEND BENCHMARKS")
                print("=" * 60)
            
            from backend_benchmarks import (
                BenchmarkConfig,
                run_backend_benchmarks
            )
            
            config = BenchmarkConfig(
                backends=args.backends,
                qubit_counts=args.qubits,
                shots=args.shots,
                repetitions=args.repetitions,
            )
            
            results = run_backend_benchmarks(config, verbose=verbose)
            all_results.extend(results)
        
        # Circuit benchmarks
        if args.all or args.circuit:
            if verbose:
                print("\n" + "=" * 60)
                print("  CIRCUIT BENCHMARKS")
                print("=" * 60)
            
            from circuit_benchmarks import (
                CircuitBenchmarkConfig,
                run_circuit_benchmarks
            )
            
            config = CircuitBenchmarkConfig(
                qubit_counts=args.qubits,
                target_backends=args.backends,
            )
            
            results = run_circuit_benchmarks(config, verbose=verbose)
            all_results.extend(results)
        
        # Comparison benchmarks
        if args.all or args.comparison:
            if verbose:
                print("\n" + "=" * 60)
                print("  COMPARISON BENCHMARKS")
                print("=" * 60)
            
            from comparison_benchmarks import (
                ComparisonBenchmarkConfig,
                run_comparison_benchmarks
            )
            
            config = ComparisonBenchmarkConfig(
                backends=args.backends,
                qubit_counts=args.qubits,
                shots=args.shots,
                repetitions=args.repetitions,
            )
            
            results = run_comparison_benchmarks(config, verbose=verbose)
            all_results.extend(results)
        
        # Save results
        if args.output:
            from utils import save_benchmark_results
            save_benchmark_results(all_results, args.output)
            print(f"\nResults saved to: {args.output}")
        
        # Generate report
        if args.report:
            from utils import generate_benchmark_report
            report = generate_benchmark_report(
                all_results,
                output_format="markdown",
                title="ProximA Benchmark Report"
            )
            Path(args.report).write_text(report, encoding='utf-8')
            print(f"Report saved to: {args.report}")
        
        # Summary
        if verbose:
            print("\n" + "=" * 60)
            print("  SUMMARY")
            print("=" * 60)
            successful = sum(1 for r in all_results if getattr(r, 'successful', True))
            print(f"Total benchmarks: {len(all_results)}")
            print(f"Successful: {successful}")
            print(f"Failed: {len(all_results) - successful}")
            print("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
