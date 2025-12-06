"""Model export for iOS: JIT, ExecuTorch, CoreML, ONNX. Handles dict outputs gracefully."""

import json
import logging
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def export_to_jit(model: nn.Module, save_path: str = 'maxsight_traced.pt', input_size: tuple = (1, 3, 224, 224), device: Optional[str] = None, validate: bool = True) -> Path:
    """Export to PyTorch JIT format. Most reliable, always available. strict=False for dict outputs.
    
        Arguments:
        device: Device to export from ('cpu', 'cuda', 'mps'). If None, uses model's current device.
        validate: If True, test exported model with dummy input to verify it works.
    """
    logger.info(f"Exporting to JIT format: {save_path}")
    
    model.eval()
    export_device = device if device else next(model.parameters()).device
    if isinstance(export_device, torch.device):
        export_device = str(export_device)
    
    # Move to export device
    if export_device == 'cpu':
        model.cpu()
    elif export_device.startswith('cuda'):
        model.cuda()
    elif export_device == 'mps':
        model.to('mps')
    
    dummy_input = torch.randn(*input_size)
    if export_device.startswith('cuda'):
        dummy_input = dummy_input.cuda()
    elif export_device == 'mps':
        dummy_input = dummy_input.to('mps')
    
    try:
        traced_model = torch.jit.trace(model, dummy_input, strict=False)
        
        # Validate exported model if requested
        if validate:
            test_output = traced_model(dummy_input)  # type: ignore
            logger.debug("Validation: Exported model forward pass successful")
        
        # Move to CPU for saving (JIT models should be CPU)
        traced_model.cpu()
        save_path_obj = Path(save_path)
        traced_model.save(str(save_path_obj))
        
        size_mb = save_path_obj.stat().st_size / (1024 * 1024)
        logger.info(f"Saved: {save_path}, Size: {size_mb:.1f} MB")
        
        return save_path_obj
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        raise


def export_to_executorch(model: nn.Module, save_path: str = 'maxsight.pte', input_size: tuple = (1, 3, 224, 224)) -> Optional[Path]:
    """Export to ExecuTorch format. Falls back to JIT if not installed."""
    logger.info(f"Exporting to ExecuTorch format: {save_path}")
    
    try:
        from executorch.extension.pybind11.portable import to_edge  # type: ignore
        
        model.eval()
        model.cpu()
        dummy_input = torch.randn(*input_size)
        
        edge_program = to_edge(model, (dummy_input,))
        executorch_program = edge_program.to_executorch()
        
        save_path_obj = Path(save_path)
        with open(save_path_obj, 'wb') as f:
            f.write(executorch_program.buffer)
        
        size_mb = save_path_obj.stat().st_size / (1024 * 1024)
        logger.info(f"Saved: {save_path}, Size: {size_mb:.1f} MB")
        return save_path_obj
        
    except ImportError:
        logger.warning("ExecuTorch not installed, falling back to JIT...")
        return export_to_jit(model, save_path.replace('.pte', '_traced.pt'), input_size)
    except Exception as e:
        logger.error(f"ExecuTorch export failed: {e}, falling back to JIT...", exc_info=True)
        return export_to_jit(model, save_path.replace('.pte', '_traced.pt'), input_size)


def export_to_coreml(model: nn.Module, save_path: str = 'maxsight.mlpackage', input_size: tuple = (1, 3, 224, 224), device: Optional[str] = None, validate: bool = True) -> Optional[Path]:
    """Export to CoreML format (iOS native). Requires coremltools. Handles dict outputs."""
    logger.info(f"Exporting to CoreML format: {save_path}")
    
    try:
        import coremltools as ct  # type: ignore
        
        model.eval()
        export_device = device if device else 'cpu'
        if export_device == 'cpu':
            model.cpu()
        elif export_device.startswith('cuda'):
            model.cuda()
        elif export_device == 'mps':
            model.to('mps')
        
        dummy_input = torch.randn(*input_size)
        if export_device.startswith('cuda'):
            dummy_input = dummy_input.cuda()
        elif export_device == 'mps':
            dummy_input = dummy_input.to('mps')
        
        # Wrap model to handle dict outputs
        class FlattenedModel(nn.Module):
            """Wrapper to flatten dict outputs for CoreML compatibility."""
            def __init__(self, model: nn.Module):
                super().__init__()
                self.model = model
            
            def forward(self, x: torch.Tensor):
                outputs = self.model(x)
                if isinstance(outputs, dict):
                    # Flatten to tuple of tensors (CoreML can handle tuples)
                    return tuple(outputs.values())
                return outputs
        
        wrapped_model = FlattenedModel(model)
        traced_model = torch.jit.trace(wrapped_model, dummy_input, strict=False)
        
        # Validate traced model if requested
        test_output = None
        if validate:
            test_output = traced_model(dummy_input)  # type: ignore
            logger.debug("Validation: Traced model forward pass successful")
        
        # Determine output types
        if validate and isinstance(test_output, tuple):
            # Multiple outputs from flattened dict
            output_types = [ct.TensorType(name=f"output_{i}") for i in range(len(test_output))]
        elif validate:
            # Single tensor output
            output_types = [ct.TensorType(name="output")]
        else:
            # Default: single output (will fail if model returns dict, but user should use validate=True)
            output_types = [ct.TensorType(name="output")]
        
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="image", shape=input_size)],
            outputs=output_types,
            minimum_deployment_target=ct.target.iOS15
        )
        
        save_path_obj = Path(save_path)
        if coreml_model is not None:
            # Validate CoreML model if requested
            if validate:
                try:
                    test_input_np = dummy_input.cpu().numpy()
                    test_output_ml = coreml_model.predict({"image": test_input_np})
                    logger.debug("Validation: CoreML model forward pass successful")
                except Exception as e:
                    logger.warning(f"CoreML validation failed: {e}")
            
            coreml_model.save(str(save_path_obj))
        else:
            raise ValueError("CoreML conversion failed")
        
        size_mb = save_path_obj.stat().st_size / (1024 * 1024)
        logger.info(f"Saved: {save_path}, Size: {size_mb:.1f} MB, iOS 15+")
        return save_path_obj
        
    except ImportError:
        logger.warning("CoreML tools not installed (pip install coremltools)")
        return None
    except Exception as e:
        logger.error(f"CoreML export failed: {e}", exc_info=True)
        return None


def export_to_onnx(model: nn.Module, save_path: str = 'maxsight.onnx', input_size: tuple = (1, 3, 224, 224)) -> Optional[Path]:
    """Export to ONNX format. May fail with dict outputs - use JIT/CoreML for iOS."""
    logger.info(f"Exporting to ONNX format: {save_path}")
    
    try:
        import onnx  # type: ignore
        
        model.eval()
        model.cpu()
        dummy_input = torch.randn(*input_size)
        
        torch.onnx.export(
            model,
            (dummy_input,),
            save_path,
            input_names=['image'],
            output_names=['output'],  # Note: may not work with dict outputs
            dynamic_axes={'image': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=11
        )
        
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        
        save_path_obj = Path(save_path)
        size_mb = save_path_obj.stat().st_size / (1024 * 1024)
        logger.info(f"Saved: {save_path}, Size: {size_mb:.1f} MB")
        return save_path_obj
        
    except ImportError:
        logger.warning("ONNX not installed (pip install onnx)")
        return None
    except Exception as e:
        logger.error(f"ONNX export failed: {e}", exc_info=True)
        return None


def export_model(model: nn.Module, format: str = 'jit', save_dir: str = 'exports', 
                 input_size: tuple = (1, 3, 224, 224), device: Optional[str] = None, 
                 validate: bool = True) -> dict:
    """Export model to specified format(s). Formats: 'jit', 'executorch', 'coreml', 'onnx', 'all'."""
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(exist_ok=True, parents=True)  # Create export directory if needed
    
    results = {
        'format': format,  # Requested format
        'exports': {},  # Successful export paths
        'metadata': {
            'input_size': input_size,  # Input dimensions
            'model_params': sum(p.numel() for p in model.parameters()),  # Total parameter count
        }
    }
    
    if format == 'jit' or format == 'all':
        path = export_to_jit(model, str(save_dir_path / 'maxsight_traced.pt'), input_size, device, validate)
        results['exports']['jit'] = str(path)
    
    if format == 'executorch' or format == 'all':
        path = export_to_executorch(model, str(save_dir_path / 'maxsight.pte'), input_size)
        if path:
            results['exports']['executorch'] = str(path)
    
    if format == 'coreml' or format == 'all':
        path = export_to_coreml(model, str(save_dir_path / 'maxsight.mlpackage'), input_size)
        if path:
            results['exports']['coreml'] = str(path)
    
    if format == 'onnx' or format == 'all':
        path = export_to_onnx(model, str(save_dir_path / 'maxsight.onnx'), input_size)
        if path:
            results['exports']['onnx'] = str(path)
    
    metadata_path = save_dir_path / 'export_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Export metadata saved to: {metadata_path}")
    
    return results


if __name__ == "__main__":
    from ml.models.maxsight_cnn import create_model
    
    model = create_model()
    model.eval()
    
    print("Testing export functionality...")
    export_to_jit(model, 'test_maxsight_traced.pt')
    export_to_executorch(model, 'test_maxsight.pte')
    export_to_coreml(model, 'test_maxsight.mlpackage')
    export_to_onnx(model, 'test_maxsight.onnx')
    print("Export system ready!")

