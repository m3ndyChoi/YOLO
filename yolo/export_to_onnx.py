# export_to_onnx.py
"""
YOLOv9 Model Export to ONNX

Exports trained PyTorch Lightning checkpoint to ONNX format
for deployment and inference optimization.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from omegaconf import OmegaConf
import argparse


def load_model_from_checkpoint(checkpoint_path, config_path, class_num=1):
    """
    Load YOLO model from PyTorch Lightning checkpoint
    
    Args:
        checkpoint_path: Path to .ckpt file
        config_path: Path to config.yaml
        class_num: Number of classes
    
    Returns:
        model: Loaded PyTorch model
    """
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    
    # Load config
    cfg = OmegaConf.load(config_path)
    
    # Import YOLO model
    sys.path.append(str(Path(__file__).parent))
    from yolo.model.yolo import YOLO
    
    # Create model
    model = YOLO(cfg.model, class_num=class_num)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'model.' prefix if exists
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.model.'):
            new_key = k.replace('model.model.', '')
        elif k.startswith('model.'):
            new_key = k.replace('model.', '')
        else:
            new_key = k
        new_state_dict[new_key] = v
    
    # Load weights
    model.model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    print("✅ Model loaded successfully")
    return model


def export_to_onnx(
    model,
    output_path,
    input_size=(640, 640),
    batch_size=1,
    dynamic_batch=True,
    opset_version=12,
    simplify=False
):
    """
    Export PyTorch model to ONNX format
    
    Args:
        model: PyTorch model
        output_path: Output .onnx file path
        input_size: Input image size (height, width)
        batch_size: Batch size for export
        dynamic_batch: Enable dynamic batch size
        opset_version: ONNX opset version
        simplify: Simplify ONNX model (requires onnx-simplifier)
    """
    print(f"\n🔄 Exporting to ONNX...")
    print(f"  Input size: {input_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Dynamic batch: {dynamic_batch}")
    print(f"  Opset version: {opset_version}")
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, input_size[0], input_size[1])
    
    # Dynamic axes configuration
    if dynamic_batch:
        dynamic_axes = {
            'images': {0: 'batch'},
            'output': {0: 'batch'}
        }
    else:
        dynamic_axes = None
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    print(f"✅ ONNX export complete: {output_path}")
    
    # Simplify ONNX model
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            
            print("\n🔧 Simplifying ONNX model...")
            onnx_model = onnx.load(output_path)
            onnx_model_simplified, check = onnx_simplify(onnx_model)
            
            if check:
                onnx.save(onnx_model_simplified, output_path)
                print("✅ ONNX simplification complete")
            else:
                print("⚠️  ONNX simplification failed, using original model")
        except ImportError:
            print("⚠️  onnx-simplifier not installed. Skipping simplification.")
            print("   Install with: pip install onnx-simplifier")


def verify_onnx_model(onnx_path, input_size=(640, 640)):
    """
    Verify exported ONNX model
    
    Args:
        onnx_path: Path to .onnx file
        input_size: Input image size
    """
    try:
        import onnx
        import onnxruntime as ort
        
        print(f"\n🔍 Verifying ONNX model...")
        
        # Check ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model structure is valid")
        
        # Test inference with ONNX Runtime
        print("\n🧪 Testing ONNX Runtime inference...")
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # Get input/output info
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        print(f"  Input: {input_name}")
        print(f"  Output: {output_name}")
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).numpy()
        
        # Run inference
        outputs = session.run([output_name], {input_name: dummy_input})
        
        print(f"  Output shape: {outputs[0].shape}")
        print("✅ ONNX Runtime inference successful")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Verification skipped: {e}")
        print("   Install with: pip install onnx onnxruntime")
        return False
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def get_model_size(file_path):
    """Get file size in MB"""
    size_bytes = Path(file_path).stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


def main():
    parser = argparse.ArgumentParser(description='Export YOLOv9 model to ONNX')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to PyTorch Lightning checkpoint (.ckpt)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output ONNX file path (default: same as checkpoint with .onnx extension)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='yolo/config/config.yaml',
        help='Path to config.yaml'
    )
    parser.add_argument(
        '--class-num',
        type=int,
        default=1,
        help='Number of classes'
    )
    parser.add_argument(
        '--input-size',
        type=int,
        nargs=2,
        default=[640, 640],
        help='Input size (height width)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for export'
    )
    parser.add_argument(
        '--dynamic-batch',
        action='store_true',
        help='Enable dynamic batch size'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=12,
        help='ONNX opset version'
    )
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Simplify ONNX model'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        default=True,
        help='Verify exported ONNX model'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
        return
    
    # Set output path
    if args.output is None:
        output_path = checkpoint_path.with_suffix('.onnx')
    else:
        output_path = Path(args.output)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("YOLOv9 ONNX Export")
    print("="*60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output:     {output_path}")
    print(f"Classes:    {args.class_num}")
    print("="*60)
    
    try:
        # Load model
        model = load_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            config_path=args.config,
            class_num=args.class_num
        )
        
        # Export to ONNX
        export_to_onnx(
            model=model,
            output_path=output_path,
            input_size=tuple(args.input_size),
            batch_size=args.batch_size,
            dynamic_batch=args.dynamic_batch,
            opset_version=args.opset,
            simplify=args.simplify
        )
        
        # Verify
        if args.verify:
            verify_onnx_model(output_path, tuple(args.input_size))
        
        # Show file sizes
        print("\n📊 File Sizes:")
        ckpt_size = get_model_size(checkpoint_path)
        onnx_size = get_model_size(output_path)
        print(f"  Checkpoint: {ckpt_size:.2f} MB")
        print(f"  ONNX:       {onnx_size:.2f} MB")
        print(f"  Reduction:  {((ckpt_size - onnx_size) / ckpt_size * 100):.1f}%")
        
        print("\n✨ Export complete!")
        print(f"\n📦 ONNX model saved to: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
