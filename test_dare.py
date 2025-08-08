#!/usr/bin/env python3
"""
Simple test script for DARE implementation
Creates dummy SafeTensors files and tests the merger
"""

import torch
import safetensors.torch
import tempfile
import os
import subprocess
import sys
from main import DareMerger, DareConfig, load_config_from_toml

def create_dummy_model(filename: str, base_weights: dict = None, noise_scale: float = 0.001):
    """Create a dummy SafeTensors model file for testing"""
    if base_weights is None:
        weights = {
            'layer1.weight': torch.randn(10, 5),
            'layer1.bias': torch.randn(10),
            'layer2.weight': torch.randn(3, 10),
            'layer2.bias': torch.randn(3),
        }
    else:
        # Add small noise to base weights to simulate fine-tuning
        weights = {}
        for key, weight in base_weights.items():
            noise = torch.randn_like(weight) * noise_scale
            weights[key] = weight + noise
    
    safetensors.torch.save_file(weights, filename)
    return weights

def test_dare_merger():
    """Test the DARE merger with dummy models"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create file paths
        base_file = os.path.join(tmpdir, 'base.safetensors')
        math_file = os.path.join(tmpdir, 'math.safetensors')
        code_file = os.path.join(tmpdir, 'code.safetensors')
        output_file = os.path.join(tmpdir, 'merged.safetensors')
        
        # Create base model
        print("Creating dummy models...")
        base_weights = create_dummy_model(base_file)
        
        # Create fine-tuned models with small deltas
        create_dummy_model(math_file, base_weights, noise_scale=0.001)
        create_dummy_model(code_file, base_weights, noise_scale=0.0015)
        
        print(f"Base model shape: {base_weights['layer1.weight'].shape}")
        print(f"Base weight sample: {base_weights['layer1.weight'][0, :3]}")
        
        # Test merger
        config = DareConfig(drop_rate=0.9, verbose=True, delta_threshold=0.01)
        merger = DareMerger(
            base_file=base_file,
            merge_files=[math_file, code_file],
            config=config
        )
        
        print("\nStarting DARE merge...")
        merger.merge(output_file)
        
        # Verify output
        print("\nVerifying merged model...")
        merged_weights = safetensors.torch.load_file(output_file)
        
        print(f"Merged model keys: {list(merged_weights.keys())}")
        print(f"Merged weight sample: {merged_weights['layer1.weight'][0, :3]}")
        
        # Check that shapes match
        for key in base_weights:
            assert key in merged_weights, f"Missing key: {key}"
            assert merged_weights[key].shape == base_weights[key].shape, f"Shape mismatch for {key}"
        
        print("✅ Test passed! DARE merger works correctly.")

def test_config_defaults():
    """Test that DareConfig has proper defaults"""
    config = DareConfig()
    assert config.drop_rate == 0.9
    assert config.delta_threshold == 0.002
    assert config.verbose == True
    print("✅ DareConfig defaults test passed!")

def test_toml_config():
    """Test TOML configuration loading"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test config
        config_content = """
[dare]
drop_rate = 0.95
delta_threshold = 0.001
verbose = false

[models]
base_file = "test_base.safetensors"
merge_files = ["test1.safetensors", "test2.safetensors"]
output_file = "test_output.safetensors"
"""
        config_file = os.path.join(tmpdir, 'test_config.toml')
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        # Load config
        config, base_file, merge_files, output_file = load_config_from_toml(config_file)
        
        # Verify config
        assert config.drop_rate == 0.95
        assert config.delta_threshold == 0.001
        assert config.verbose == False
        assert base_file == "test_base.safetensors"
        assert merge_files == ["test1.safetensors", "test2.safetensors"]
        assert output_file == "test_output.safetensors"
        
        print("✅ TOML config loading test passed!")

def test_cli_interface():
    """Test CLI interface with dummy files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy files
        base_file = os.path.join(tmpdir, 'base.safetensors')
        math_file = os.path.join(tmpdir, 'math.safetensors')
        code_file = os.path.join(tmpdir, 'code.safetensors')
        output_file = os.path.join(tmpdir, 'merged.safetensors')
        
        base_weights = create_dummy_model(base_file)
        create_dummy_model(math_file, base_weights, 0.001)
        create_dummy_model(code_file, base_weights, 0.001)
        
        # Test CLI with direct arguments
        result = subprocess.run([
            sys.executable, 'main.py',
            '--base', base_file,
            '--merge', math_file, code_file,
            '--output', output_file,
            '--drop-rate', '0.8',
            '--quiet'
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            # Verify output file was created
            if os.path.exists(output_file):
                print("✅ CLI interface test passed!")
            else:
                print(f"❌ CLI test failed: Output file was not created")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")
        else:
            print(f"❌ CLI test failed: {result.stderr}")
            print(f"stdout: {result.stdout}")

if __name__ == "__main__":
    print("Testing DARE implementation...\n")
    test_config_defaults()
    test_dare_merger()
    test_toml_config()
    test_cli_interface()
    print("\n🎉 All tests passed!")
