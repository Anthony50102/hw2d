#!/usr/bin/env python3
"""
Test script to verify precision implementation in HW2D
"""

import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from hw2d.model import HW
from hw2d.utils.namespaces import Namespace


def test_precision():
    """Test that precision setting works correctly"""
    print("Testing precision implementation...")
    
    # Test parameters
    dx = 0.1
    N = 3
    c1 = 1.0
    nu = 5e-8
    k0 = 0.15
    grid_size = 32
    
    # Test double precision
    print("\n=== Testing Double Precision ===")
    hw_double = HW(
        dx=dx, N=N, c1=c1, nu=nu, k0=k0, 
        precision="double", debug=True
    )
    
    print(f"Double precision float type: {hw_double.float_type}")
    print(f"Double precision complex type: {hw_double.complex_type}")
    print(f"Expected: {np.float64}, {np.complex128}")
    
    # Test single precision
    print("\n=== Testing Single Precision ===")
    hw_single = HW(
        dx=dx, N=N, c1=c1, nu=nu, k0=k0, 
        precision="single", debug=True
    )
    
    print(f"Single precision float type: {hw_single.float_type}")
    print(f"Single precision complex type: {hw_single.complex_type}")
    print(f"Expected: {np.float32}, {np.complex64}")
    
    # Test with some data
    print("\n=== Testing Data Type Consistency ===")
    
    # Create test data
    omega = np.random.randn(grid_size, grid_size).astype(np.float64)
    
    # Test double precision phi calculation
    phi_double = hw_double.get_phi(omega, dx)
    print(f"Double precision phi dtype: {phi_double.dtype}")
    print(f"Expected: {np.float64}")
    
    # Test single precision phi calculation  
    phi_single = hw_single.get_phi(omega, dx)
    print(f"Single precision phi dtype: {phi_single.dtype}")
    print(f"Expected: {np.float32}")
    
    # Test precision conversion
    print("\n=== Testing Precision Conversion ===")
    test_array = np.random.randn(10, 10).astype(np.float64)
    
    converted_double = hw_double._ensure_precision(test_array)
    converted_single = hw_single._ensure_precision(test_array) 
    
    print(f"Original dtype: {test_array.dtype}")
    print(f"Double converted dtype: {converted_double.dtype}")
    print(f"Single converted dtype: {converted_single.dtype}")
    
    # Test invalid precision
    print("\n=== Testing Invalid Precision ===")
    try:
        hw_invalid = HW(dx=dx, N=N, c1=c1, nu=nu, k0=k0, precision="invalid")
        print("ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"Correctly caught error: {e}")
    
    print("\n=== All tests completed! ===")
    
    # Test solver selection
    print("\n=== Testing Solver Selection ===")
    print(f"Double precision solver: {hw_double.poisson_solver.__name__}")
    print(f"Single precision solver: {hw_single.poisson_solver.__name__}")
    print("Expected: fourier_poisson_double, fourier_poisson_single")


if __name__ == "__main__":
    test_precision()