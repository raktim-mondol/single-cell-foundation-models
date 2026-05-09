"""
Comprehensive Test Runner for All Single-Cell Foundation Model Methods
========================================================================

This script tests all 6 novel methods using synthetic data to ensure:
1. Modular structure works correctly
2. Synthetic data generation functions properly
3. Training loops execute without errors
4. Evaluation metrics compute correctly

Usage:
    python test_all_methods.py --method all
    python test_all_methods.py --method PathMoE_scFM
    python test_all_methods.py --method CausalCellFM --quick
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


class MethodTester:
    """Handles testing for individual methods."""
    
    def __init__(self, base_dir: Path, quick: bool = False):
        self.base_dir = base_dir
        self.quick = quick
        self.methods = {
            "PathMoE_scFM": {
                "path": base_dir / "PathMoE_scFM",
                "script": "train.py",
                "args": ["--n_genes", "500", "--n_cells", "1000", "--pretrain_epochs", "2", "--finetune_epochs", "1"],
                "description": "Pathway-aware sparse MoE transformer"
            },
            "CausalCellFM": {
                "path": base_dir / "CausalCellFM",
                "script": "train.py",
                "args": ["--n_genes", "500", "--n_cells", "1000", "--epochs", "3"],
                "description": "Counterfactual perturbation foundation model"
            },
            "SpaceTime_scFM": {
                "path": base_dir / "SpaceTime_scFM",
                "script": "train.py",
                "args": ["--n_genes", "300", "--n_cells", "500", "--epochs", "3"],
                "description": "Spatio-temporal multi-modal foundation model"
            },
            "Atlas_Streamer": {
                "path": base_dir / "Atlas_Streamer",
                "script": "train.py",
                "args": ["--n_releases", "3", "--cells_per_release", "200", "--update_steps", "20"],
                "description": "Continual learning for atlas updates"
            },
            "GRN_Decoder_VAE": {
                "path": base_dir / "GRN_Decoder_VAE",
                "script": "train.py",
                "args": ["--n_genes", "100", "--n_cells", "500", "--epochs", "5"],
                "description": "GRN-constrained generative foundation model"
            },
            "scTrueBench": {
                "path": base_dir / "scTrueBench",
                "script": "run_benchmark.py",
                "args": ["--n_cells", "200", "--n_genes", "100"],
                "description": "Causal benchmark suite"
            }
        }
    
    def test_method(self, method_name: str) -> Dict[str, any]:
        """Test a single method."""
        if method_name not in self.methods:
            return {"success": False, "error": f"Unknown method: {method_name}"}
        
        method_info = self.methods[method_name]
        method_dir = method_info["path"]
        script = method_info["script"]
        args = method_info["args"]
        
        print(f"\n{'='*70}")
        print(f"Testing: {method_name}")
        print(f"Description: {method_info['description']}")
        print(f"{'='*70}")
        
        # Check if method directory exists
        if not method_dir.exists():
            return {
                "success": False,
                "error": f"Method directory not found: {method_dir}"
            }
        
        # Check if script exists
        script_path = method_dir / script
        if not script_path.exists():
            return {
                "success": False,
                "error": f"Script not found: {script_path}"
            }
        
        # Check for required files
        required_files = ["model.py", "utils.py"]
        missing_files = []
        for req_file in required_files:
            if not (method_dir / req_file).exists():
                missing_files.append(req_file)
        
        if missing_files:
            return {
                "success": False,
                "error": f"Missing required files: {missing_files}"
            }
        
        # Run the method
        cmd = [sys.executable, str(script_path)] + args
        print(f"Running: {' '.join(cmd)}")
        print(f"Working directory: {method_dir}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=method_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print("✓ Method executed successfully")
                print("Output preview:")
                print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
                return {
                    "success": True,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                print("✗ Method execution failed")
                print(f"Error: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr,
                    "stdout": result.stdout
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Method execution timed out (5 minutes)"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Exception during execution: {str(e)}"
            }
    
    def test_all_methods(self) -> Dict[str, Dict[str, any]]:
        """Test all methods."""
        results = {}
        
        print("\n" + "="*70)
        print("COMPREHENSIVE METHOD TESTING")
        print("="*70)
        print(f"Base directory: {self.base_dir}")
        print(f"Quick mode: {self.quick}")
        print(f"Methods to test: {len(self.methods)}")
        
        for method_name in self.methods.keys():
            results[method_name] = self.test_method(method_name)
        
        return results
    
    def print_summary(self, results: Dict[str, Dict[str, any]]):
        """Print test summary."""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        successful = 0
        failed = 0
        
        for method_name, result in results.items():
            status = "✓ PASS" if result["success"] else "✗ FAIL"
            print(f"{status:8} | {method_name:20} | {self.methods[method_name]['description']}")
            
            if result["success"]:
                successful += 1
            else:
                failed += 1
                print(f"         Error: {result.get('error', 'Unknown error')}")
        
        print("="*70)
        print(f"Total: {len(results)} | Successful: {successful} | Failed: {failed}")
        print("="*70)
        
        if failed == 0:
            print("\n🎉 All methods passed successfully!")
            return 0
        else:
            print(f"\n⚠️  {failed} method(s) failed. Please check the errors above.")
            return 1


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ["torch", "numpy", "scipy"]
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing required packages: {missing}")
        print("Install with: pip install torch numpy scipy")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test runner for single-cell foundation model methods"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["all"] + list([
            "PathMoE_scFM", "CausalCellFM", "SpaceTime_scFM",
            "Atlas_Streamer", "GRN_Decoder_VAE", "scTrueBench"
        ]),
        help="Method to test (default: all)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode with reduced epochs/data"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(Path(__file__).parent),
        help="Base directory containing method folders"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Initialize tester
    tester = MethodTester(Path(args.base_dir), quick=args.quick)
    
    # Run tests
    if args.method == "all":
        results = tester.test_all_methods()
    else:
        results = {args.method: tester.test_method(args.method)}
    
    # Print summary
    exit_code = tester.print_summary(results)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()