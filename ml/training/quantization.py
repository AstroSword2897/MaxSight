"""Enhanced model quantization for mobile deployment (int8) - Production Ready...."""

import torch
import torch.nn as nn
import torch.ao.quantization as quantization
from typing import Optional, Dict, Any, Union, List, Tuple
from pathlib import Path
import warnings
from copy import deepcopy


def _fuse_maxsight_modules(model: nn.Module) -> nn.Module:
    """Fuse Conv+BN+ReLU patterns in MaxSight CNN architecture...."""
    fuse_list = []
    
    # Helper to check if a Sequential has Conv+BN+ReLU pattern
    def is_fusable_conv_bn_relu(seq: nn.Sequential, start_idx: int = 0) -> bool:
        # nn.Sequential supports len() but type checker doesn't always recognize it
        seq_len = len(seq)  # type: ignore[arg-type]
        if seq_len < start_idx + 3:
            return False
        return (isinstance(seq[start_idx], nn.Conv2d) and
                isinstance(seq[start_idx + 1], nn.BatchNorm2d) and
                isinstance(seq[start_idx + 2], (nn.ReLU, nn.ReLU6)))
    
    # Traverse model and find fusable patterns
    for name, module in model.named_modules():
        if isinstance(module, nn.Sequential):
            # Check for Conv+BN+ReLU at start of Sequential
            if is_fusable_conv_bn_relu(module, 0):
                # Create fuse pattern: ['module_name.0', 'module_name.1', 'module_name.2']
                fuse_pattern = [f"{name}.0", f"{name}.1", f"{name}.2"]
                fuse_list.append(fuse_pattern)
    
    # Also handle ResNet backbone patterns if present
    # ResNet conv1, bn1, relu can be fused
    if hasattr(model, 'conv1') and hasattr(model, 'bn1') and hasattr(model, 'relu'):
        try:
            fuse_list.append(['conv1', 'bn1', 'relu'])
        except Exception:
            pass
    
    # Fuse all detected patterns
    fused_count = 0
    for fuse_pattern in fuse_list:
        try:
            quantization.fuse_modules(model, [fuse_pattern], inplace=True)
            fused_count += 1
        except Exception as e:
            # Some patterns might not be fusable (e.g., already fused, wrong structure)
            continue
    
    if fused_count > 0:
        print(f"Fused {fused_count} Conv+BN+ReLU patterns for better performance")
    else:
        warnings.warn("No modules were fused. Model may not have standard Conv+BN+ReLU patterns.")
    
    return model


def quantize_model_int8(
    model: nn.Module,
    calibration_data: Optional[torch.utils.data.DataLoader] = None,
    num_calibration_batches: int = 10,
    backend: str = 'qnnpack',  # Default to ARM for iOS deployment
    fuse_modules: bool = True
) -> nn.Module:
    """Quantize model to int8 using PyTorch's quantization API...."""
    # Create a copy to avoid modifying original model
    model = deepcopy(model)
    model.eval()
    
    # Set quantization backend
    torch.backends.quantized.engine = backend
    
    # Fuse modules if requested (MaxSight-specific fusion)
    if fuse_modules:
        try:
            model = _fuse_maxsight_modules(model)
        except Exception as e:
            warnings.warn(f"Module fusion failed: {e}. Continuing without fusion.")
    
    # Set quantization config with proper API usage
    if backend == 'qnnpack':
        # ARM/iOS backend - REQUIRES per-channel weight quantization for stability
        model.qconfig = quantization.get_default_qconfig('qnnpack')  # type: ignore
        # Critical: per-channel weight quantization for mobile CNN stability
        if model.qconfig is not None:
            model.qconfig.weight = quantization.default_per_channel_weight_observer  # type: ignore
    elif backend == 'fbgemm':
        # x86 backend
        model.qconfig = quantization.get_default_qconfig('fbgemm')  # type: ignore
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'qnnpack' (ARM) or 'fbgemm' (x86)")
    
    # Prepare model for quantization - USE MODERN API
    model_prepared = quantization.prepare(model, inplace=False)
    
    # Calibrate with sample data
    print(f"Calibrating model for quantization using {num_calibration_batches} batches...")
    
    if calibration_data is None:
        # Create dummy calibration data with realistic variations
        print("Warning: No calibration data provided. Using synthetic data.")
        dummy_inputs = [torch.randn(1, 3, 224, 224) for _ in range(num_calibration_batches)]
        calibration_data = [(inp,) for inp in dummy_inputs]  # type: ignore
    
    batch_count = 0
    with torch.no_grad():
        for batch in calibration_data:  # type: ignore
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            elif isinstance(batch, dict):
                inputs = batch.get('images') or batch.get('input') or batch.get('data')
            else:
                inputs = batch
            
            # Move to CPU if necessary (quantization typically done on CPU)
            if inputs is not None and hasattr(inputs, 'device') and inputs.device.type != 'cpu':
                inputs = inputs.cpu()
            
            try:
                model_prepared(inputs)
                batch_count += 1
                if batch_count % 5 == 0:
                    print(f"  Processed {batch_count}/{num_calibration_batches} batches")
                if batch_count >= num_calibration_batches:
                    break
            except Exception as e:
                warnings.warn(f"Error processing batch {batch_count}: {e}")
                continue
    
    if batch_count == 0:
        raise RuntimeError("No batches successfully processed during calibration")
    
    # Convert to quantized model - USE MODERN API
    print("Converting to int8...")
    model_int8 = quantization.convert(model_prepared, inplace=False)
    
    print(f"Quantization complete ({batch_count} calibration batches used)")
    print(f"  Backend: {backend} ({'ARM/iOS ready' if backend == 'qnnpack' else 'x86'})")
    return model_int8


def compare_model_sizes(
    model_fp32: nn.Module,
    model_int8: Optional[nn.Module] = None,
    save_models: bool = False,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Compare model sizes (FP32 vs INT8) with optional disk size measurement...."""
    # Count parameters
    total_params = sum(p.numel() for p in model_fp32.parameters())
    trainable_params = sum(p.numel() for p in model_fp32.parameters() if p.requires_grad)
    
    # Estimate FP32 size (4 bytes per parameter)
    fp32_size_mb = total_params * 4 / (1024 * 1024)
    
    results = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'fp32_size_mb': fp32_size_mb,
        'fp32_size_estimate': f"{fp32_size_mb:.2f} MB",
        'target_size_mb': 50.0,
        'compression_ratio': None,
        'int8_size_mb': None,
        'disk_sizes': {}
    }
    
    if model_int8 is not None:
        # INT8 uses 1 byte per parameter (theoretical)
        int8_size_mb = total_params / (1024 * 1024)
        results['int8_size_mb'] = int8_size_mb
        results['int8_size_estimate'] = f"{int8_size_mb:.2f} MB"
        results['compression_ratio'] = fp32_size_mb / int8_size_mb
        results['meets_target'] = int8_size_mb < 50.0
        results['size_reduction'] = f"{((fp32_size_mb - int8_size_mb) / fp32_size_mb * 100):.1f}%"
    
    # Measure actual disk sizes if requested
    if save_models and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FP32 model
        fp32_path = output_dir / "model_fp32.pth"
        torch.save(model_fp32.state_dict(), fp32_path)
        fp32_disk_size = fp32_path.stat().st_size / (1024 * 1024)
        results['disk_sizes']['fp32'] = f"{fp32_disk_size:.2f} MB"
        
        # Save INT8 model if available
        if model_int8 is not None:
            int8_path = output_dir / "model_int8.pth"
            torch.save(model_int8.state_dict(), int8_path)
            int8_disk_size = int8_path.stat().st_size / (1024 * 1024)
            results['disk_sizes']['int8'] = f"{int8_disk_size:.2f} MB"
            results['disk_compression_ratio'] = fp32_disk_size / int8_disk_size
    
    return results


def validate_quantized_model(
    model_fp32: nn.Module,
    model_int8: nn.Module,
    test_inputs: Union[torch.Tensor, List[torch.Tensor]],
    tolerance: float = 0.01,
    metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Validate quantized model by comparing outputs with FP32 model...."""
    model_fp32.eval()
    model_int8.eval()
    
    if not isinstance(test_inputs, list):
        test_inputs = [test_inputs]
    
    metrics = metrics or ['mse', 'mae']
    
    all_results = []
    
    with torch.no_grad():
        for i, test_input in enumerate(test_inputs):
            # Ensure CPU tensors
            if test_input.device.type != 'cpu':
                test_input = test_input.cpu()
            
            output_fp32 = model_fp32(test_input)
            output_int8 = model_int8(test_input)
            
            result = _compare_outputs(output_fp32, output_int8, tolerance, metrics)
            result['input_index'] = i
            all_results.append(result)
    
    # Aggregate results
    if len(all_results) == 1:
        return all_results[0]
    else:
        # Average metrics across inputs
        aggregated = {
            'num_test_inputs': len(all_results),
            'max_relative_difference': max(r['max_relative_difference'] for r in all_results),
            'avg_relative_difference': sum(r['max_relative_difference'] for r in all_results) / len(all_results),
            'accuracy_loss_percent': max(r['accuracy_loss_percent'] for r in all_results),
            'meets_tolerance': all(r['meets_tolerance'] for r in all_results),
            'target': '<1% accuracy loss',
            'per_input_results': all_results
        }
        
        return aggregated


def _compare_outputs(
    output_fp32: Any,
    output_int8: Any,
    tolerance: float,
    metrics: List[str]
) -> Dict[str, Any]:
    """Helper to compare two outputs (dict or tensor)."""
    
    if isinstance(output_fp32, dict) and isinstance(output_int8, dict):
        differences = {}
        max_diff = 0.0
        metric_results = {m: {} for m in metrics}
        
        for key in output_fp32.keys():
            if key in output_int8:
                fp32_val = output_fp32[key]
                int8_val = output_int8[key]
                
                if isinstance(fp32_val, torch.Tensor) and isinstance(int8_val, torch.Tensor):
                    diff_metrics = _compute_tensor_differences(
                        fp32_val, int8_val, metrics
                    )
                    differences[key] = diff_metrics['max_relative_diff']
                    max_diff = max(max_diff, diff_metrics['max_relative_diff'])
                    
                    for metric in metrics:
                        if metric in diff_metrics:
                            metric_results[metric][key] = diff_metrics[metric]
        
        accuracy_loss = max_diff * 100
        
        results = {
            'max_relative_difference': max_diff,
            'accuracy_loss_percent': accuracy_loss,
            'per_output_differences': differences,
            'meets_tolerance': accuracy_loss < (tolerance * 100),
            'target': '<1% accuracy loss',
            'additional_metrics': metric_results
        }
    
    elif isinstance(output_fp32, torch.Tensor) and isinstance(output_int8, torch.Tensor):
        diff_metrics = _compute_tensor_differences(output_fp32, output_int8, metrics)
        
        results = {
            'max_relative_difference': diff_metrics['max_relative_diff'],
            'accuracy_loss_percent': diff_metrics['max_relative_diff'] * 100,
            'meets_tolerance': diff_metrics['max_relative_diff'] < tolerance,
            'target': '<1% accuracy loss',
            'additional_metrics': {k: v for k, v in diff_metrics.items() if k != 'max_relative_diff'}
        }
    else:
        results = {
            'error': f'Cannot compare outputs - incompatible types: {type(output_fp32)} vs {type(output_int8)}'
        }
    
    return results


def _compute_tensor_differences(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    metrics: List[str]
) -> Dict[str, float]:
    """Compute various difference metrics between two tensors."""
    t1_float = tensor1.float()
    t2_float = tensor2.float()
    
    # Relative difference
    diff = torch.abs(t1_float - t2_float)
    rel_diff = diff / (torch.abs(t1_float) + 1e-8)
    max_rel_diff = rel_diff.max().item()
    
    result = {'max_relative_diff': max_rel_diff}
    
    if 'mse' in metrics:
        mse = torch.mean((t1_float - t2_float) ** 2).item()
        result['mse'] = mse
    
    if 'mae' in metrics:
        mae = torch.mean(torch.abs(t1_float - t2_float)).item()
        result['mae'] = mae
    
    if 'cosine' in metrics:
        # Cosine similarity
        t1_flat = t1_float.flatten()
        t2_flat = t2_float.flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(
            t1_flat.unsqueeze(0), t2_flat.unsqueeze(0)
        ).item()
        result['cosine_similarity'] = cosine_sim
    
    return result


def print_quantization_results(
    size_info: Dict[str, Any],
    validation: Dict[str, Any],
    verbose: bool = True
) -> None:
    """Print quantization results in readable format."""
    print("\nModel Quantization Results")
    
    print("\nModel Statistics:")
    print(f"  Total Parameters:     {size_info['total_parameters']:,}")
    print(f"  Trainable Parameters: {size_info['trainable_parameters']:,}")
    
    print("\nSize Comparison:")
    print(f"  FP32 Size (est):  {size_info['fp32_size_mb']:.1f} MB")
    if size_info['int8_size_mb'] is not None:
        print(f"  INT8 Size (est):  {size_info['int8_size_mb']:.1f} MB")
        print(f"  Compression:      {size_info['compression_ratio']:.1f}x")
        print(f"  Size Reduction:   {size_info.get('size_reduction', 'N/A')}")
        print(f"  Target:           <50 MB")
        print(f"  Status:           {'PASS' if size_info.get('meets_target', False) else 'FAIL'}")
    
    if size_info.get('disk_sizes'):
        print("\nActual Disk Sizes:")
        for model_type, size in size_info['disk_sizes'].items():
            print(f"  {model_type.upper()}: {size}")
    
    print("\nAccuracy Validation:")
    if 'accuracy_loss_percent' in validation:
        if 'avg_relative_difference' in validation:
            print(f"  Max Accuracy Loss:  {validation['accuracy_loss_percent']:.2f}%")
            print(f"  Avg Accuracy Loss:  {validation['avg_relative_difference'] * 100:.2f}%")
        else:
            print(f"  Accuracy Loss:      {validation['accuracy_loss_percent']:.2f}%")
        print(f"  Target:             <1%")
        print(f"  Status:             {'PASS' if validation.get('meets_tolerance', False) else 'FAIL'}")
        
        if verbose and 'additional_metrics' in validation:
            print("\nDetailed Metrics:")
            for metric, values in validation['additional_metrics'].items():
                if isinstance(values, dict):
                    print(f"  {metric.upper()}:")
                    for key, val in values.items():
                        print(f"    {key}: {val:.6f}")
                elif values is not None:
                    print(f"  {metric.upper()}: {values:.6f}")
    else:
        print(f"  Error: {validation.get('error', 'Unknown error')}")
    
    print()


def quantize_and_validate(
    model_fp32: nn.Module,
    calibration_data: Optional[torch.utils.data.DataLoader] = None,
    test_data: Optional[torch.utils.data.DataLoader] = None,
    num_calibration_batches: int = 20,
    backend: str = 'qnnpack',
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Complete Week 1 pipeline: Quantize → Validate → Size Check...."""
    print("\nWeek 1: Model Quantization Pipeline")
    
    # Step 1: Quantize
    print("\n[Step 1/3] Quantizing model to INT8...")
    model_int8 = quantize_model_int8(
        model=model_fp32,
        calibration_data=calibration_data,
        num_calibration_batches=num_calibration_batches,
        backend=backend,
        fuse_modules=True
    )
    
    # Step 2: Compare sizes
    print("\n[Step 2/3] Comparing model sizes...")
    size_info = compare_model_sizes(
        model_fp32=model_fp32,
        model_int8=model_int8,
        save_models=(output_dir is not None),
        output_dir=output_dir
    )
    
    # Step 3: Validate accuracy
    print("\n[Step 3/3] Validating quantized model accuracy...")
    
    # Get test inputs from test_data or calibration_data
    test_inputs = []
    data_source = test_data if test_data is not None else calibration_data
    
    if data_source is not None:
        batch_count = 0
        for batch in data_source:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            elif isinstance(batch, dict):
                inputs = batch.get('images') or batch.get('input') or batch.get('data')
            else:
                inputs = batch
            
            if inputs is not None:
                if inputs.device.type != 'cpu':
                    inputs = inputs.cpu()
                test_inputs.append(inputs)
                batch_count += 1
                if batch_count >= 5:  # Use 5 batches for validation
                    break
    
    if len(test_inputs) == 0:
        # Fallback to dummy data
        test_inputs = [torch.randn(1, 3, 224, 224) for _ in range(3)]
        warnings.warn("No test data provided. Using synthetic data for validation.")
    
    validation = validate_quantized_model(
        model_fp32=model_fp32,
        model_int8=model_int8,
        test_inputs=test_inputs,
        tolerance=0.01,
        metrics=['mse', 'mae', 'cosine']
    )
    
    # Print results
    print_quantization_results(size_info, validation, verbose=True)
    
    # Determine if ready for export
    ready_for_export = (
        size_info.get('meets_target', False) and
        validation.get('meets_tolerance', False)
    )
    
    # Save quantized model if output_dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save quantized model state dict
        int8_path = output_dir / "model_int8.pth"
        torch.save(model_int8.state_dict(), int8_path)
        print(f"Saved quantized model to {int8_path}")
        
        # Save results summary
        results_summary = {
            'size_info': size_info,
            'validation': validation,
            'ready_for_export': ready_for_export,
            'backend': backend
        }
        import json
        summary_path = output_dir / "quantization_summary.json"
        # Convert to JSON-serializable format
        json_summary = {}
        for k, v in results_summary.items():
            if isinstance(v, dict):
                json_summary[k] = {k2: float(v2) if isinstance(v2, (int, float)) else str(v2) 
                                 for k2, v2 in v.items()}
            else:
                json_summary[k] = str(v) if not isinstance(v, (int, float, bool)) else v
        with open(summary_path, 'w') as f:
            json.dump(json_summary, f, indent=2)
        print(f"Saved results summary to {summary_path}")
    
    if ready_for_export:
        print("\nQUANTIZATION COMPLETE - Model ready for ExecuTorch export (Week 2)")
    else:
        print("\nQUANTIZATION COMPLETE - Model may need tuning before export")
        if not size_info.get('meets_target', False):
            print("   - Size target not met")
        if not validation.get('meets_tolerance', False):
            print("   - Accuracy loss exceeds 1% threshold")
    print()
    
    return {
        'model_int8': model_int8,
        'size_info': size_info,
        'validation': validation,
        'ready_for_export': ready_for_export
    }
