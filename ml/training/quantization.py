"""Model quantization for mobile deployment (int8)."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from pathlib import Path


def quantize_model_int8(
    model: nn.Module,
    calibration_data: Optional[torch.utils.data.DataLoader] = None,
    num_calibration_batches: int = 10
) -> nn.Module:
    """
    Quantize model to int8 using PyTorch's quantization API.
    
    Args:
        model: Model to quantize
        calibration_data: DataLoader for calibration (optional, uses dummy data if None)
        num_calibration_batches: Number of batches to use for calibration
    
    Returns:
        Quantized model
    """
    model.eval()
    
    # Prepare model for quantization
    model_fp32 = model
    
    # Use post-training static quantization
    # This requires calibration data to determine quantization parameters
    if calibration_data is None:
        # Create dummy calibration data
        dummy_input = torch.randn(1, 3, 224, 224)
        calibration_data = [(dummy_input,)]
    
    # Set quantization config
    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # type: ignore
    
    # Prepare model
    model_prepared = torch.quantization.prepare(model_fp32)  # type: ignore
    
    # Calibrate with sample data
    print("Calibrating model for quantization...")
    batch_count = 0
    for batch in calibration_data:
        if isinstance(batch, (list, tuple)):
            inputs = batch[0]
        else:
            inputs = batch['images'] if isinstance(batch, dict) else batch
        
        model_prepared(inputs)
        batch_count += 1
        if batch_count >= num_calibration_batches:
            break
    
    # Convert to quantized model
    print("Converting to int8...")
    model_int8 = torch.quantization.convert(model_prepared)  # type: ignore
    
    return model_int8


def compare_model_sizes(
    model_fp32: nn.Module,
    model_int8: Optional[nn.Module] = None
) -> Dict[str, Any]:
    """
    Compare model sizes (FP32 vs INT8).
    
    Returns:
        Dictionary with size information
    """
    # Count parameters
    total_params = sum(p.numel() for p in model_fp32.parameters())
    
    # Estimate FP32 size (4 bytes per parameter)
    fp32_size_mb = total_params * 4 / (1024 * 1024)
    
    results = {
        'total_parameters': total_params,
        'fp32_size_mb': fp32_size_mb,
        'target_size_mb': 50.0,
        'compression_ratio': None,
        'int8_size_mb': None
    }
    
    if model_int8 is not None:
        # INT8 uses 1 byte per parameter
        int8_size_mb = total_params / (1024 * 1024)
        results['int8_size_mb'] = int8_size_mb
        results['compression_ratio'] = fp32_size_mb / int8_size_mb
        results['meets_target'] = int8_size_mb < 50.0
    
    return results


def validate_quantized_model(
    model_fp32: nn.Module,
    model_int8: nn.Module,
    test_input: torch.Tensor,
    tolerance: float = 0.01
) -> Dict[str, Any]:
    """
    Validate quantized model by comparing outputs with FP32 model.
    
    Args:
        model_fp32: Original FP32 model
        model_int8: Quantized INT8 model
        test_input: Test input tensor
        tolerance: Maximum allowed difference (relative)
    
    Returns:
        Dictionary with validation results
    """
    model_fp32.eval()
    model_int8.eval()
    
    with torch.no_grad():
        output_fp32 = model_fp32(test_input)
        output_int8 = model_int8(test_input)
    
    # Compare outputs
    if isinstance(output_fp32, dict) and isinstance(output_int8, dict):
        differences = {}
        max_diff = 0.0
        
        for key in output_fp32.keys():
            if key in output_int8:
                fp32_val = output_fp32[key]
                int8_val = output_int8[key]
                
                if isinstance(fp32_val, torch.Tensor) and isinstance(int8_val, torch.Tensor):
                    # Compute relative difference
                    diff = torch.abs(fp32_val.float() - int8_val.float())
                    rel_diff = diff / (torch.abs(fp32_val.float()) + 1e-8)
                    max_rel_diff = rel_diff.max().item()
                    
                    differences[key] = max_rel_diff
                    max_diff = max(max_diff, max_rel_diff)
        
        accuracy_loss = max_diff * 100  # Convert to percentage
        
        results = {
            'max_relative_difference': max_diff,
            'accuracy_loss_percent': accuracy_loss,
            'per_output_differences': differences,
            'meets_tolerance': accuracy_loss < (tolerance * 100),
            'target': '<1% accuracy loss'
        }
    else:
        # Fallback for non-dict outputs
        if isinstance(output_fp32, torch.Tensor) and isinstance(output_int8, torch.Tensor):
            diff = torch.abs(output_fp32.float() - output_int8.float())
            rel_diff = diff / (torch.abs(output_fp32.float()) + 1e-8)
            max_diff = rel_diff.max().item()
            
            results = {
                'max_relative_difference': max_diff,
                'accuracy_loss_percent': max_diff * 100,
                'meets_tolerance': max_diff < tolerance,
                'target': '<1% accuracy loss'
            }
        else:
            results = {
                'error': 'Cannot compare outputs - incompatible types'
            }
    
    return results


def print_quantization_results(size_info: Dict[str, Any], validation: Dict[str, Any]) -> None:
    """Print quantization results in readable format."""
    print("\n" + "=" * 70)
    print("Model Quantization Results")
    print("=" * 70)
    
    print("\nSize Comparison:")
    print(f"  FP32 Size:  {size_info['fp32_size_mb']:.1f} MB")
    if size_info['int8_size_mb'] is not None:
        print(f"  INT8 Size:  {size_info['int8_size_mb']:.1f} MB")
        print(f"  Compression: {size_info['compression_ratio']:.1f}x")
        print(f"  Target:     <50 MB")
        print(f"  Status:     {'✓ PASS' if size_info.get('meets_target', False) else '✗ FAIL'}")
    
    print("\nAccuracy Validation:")
    if 'accuracy_loss_percent' in validation:
        print(f"  Accuracy Loss: {validation['accuracy_loss_percent']:.2f}%")
        print(f"  Target:        <1%")
        print(f"  Status:        {'✓ PASS' if validation.get('meets_tolerance', False) else '✗ FAIL'}")
    else:
        print(f"  Error: {validation.get('error', 'Unknown error')}")
    
    print("=" * 70 + "\n")

