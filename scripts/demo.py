#!/usr/bin/env python3
"""
Demo script for metrics-guided retrosynthesis.

This script demonstrates the basic usage of the retrosynthesis search
on a small subset of molecules from the USPTO-190 dataset.
"""

import os
import sys
import csv
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def create_demo_dataset():
    """Create a small demo dataset from the first 5 molecules."""
    input_file = project_root / "data" / "uspto_190_targets.csv"
    output_file = project_root / "data" / "demo_targets.csv"
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found")
        return False
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        count = 0
        for row in reader:
            if row and row[0].strip():  # Skip empty lines
                writer.writerow([row[0].strip()])
                count += 1
                if count >= 5:
                    break
    
    print(f"✓ Created demo dataset with {count} molecules: {output_file}")
    return True

def run_demo():
    """Run the retrosynthesis demo."""
    print("=" * 60)
    print("Metrics-Guided Retrosynthesis Demo")
    print("=" * 60)
    
    # Create demo dataset
    if not create_demo_dataset():
        return
    
    # Import after path setup
    try:
        from tree_search.ged_mcts_paral import evaluate_search_success
        from tree_search.ged_options import ExpandOneOptions, BuildTreeOptions, RetroBackendOption
    except ImportError as e:
        print(f"\nError importing modules: {e}")
        print("Please ensure all dependencies are installed and ASKCOS is set up.")
        print("\nTo install dependencies:")
        print("  1. Create conda environment: conda env create -f environment.yml")
        print("  2. Activate environment: conda activate metrics_guidance_retrosynthesis")
        print("  3. Install ASKCOS framework (see README.md)")
        return
    
    # Demo parameters
    dataset_name = "demo"
    model_name_list = ["uspto_original_consol_Roh"]
    ged_weight = 0.5
    pathway_output_folder = "demo_output"
    
    # Create output directory
    os.makedirs(pathway_output_folder, exist_ok=True)
    
    print(f"\nDataset: {dataset_name}")
    print(f"Model: {model_name_list}")
    print(f"GED weight: {ged_weight}")
    print(f"Output folder: {pathway_output_folder}")
    print("\nStarting retrosynthesis search...")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        # Read demo targets
        demo_file = project_root / "data" / "demo_targets.csv"
        targets = []
        with open(demo_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0].strip():
                    targets.append(row[0].strip())
        
        print(f"Processing {len(targets)} target molecules...")
        
        # Note: This is a simplified demo. In practice, you would call
        # evaluate_search_success for each target or use the full parallel processing
        print("\nNote: This demo script shows the setup. For full execution,")
        print("please use the main ged_mcts_paral.py module with proper ASKCOS setup.")
        print("\nExample command:")
        print(f"  python -m tree_search.ged_mcts_paral \\")
        print(f"    --dataset_name {dataset_name} \\")
        print(f"    --model_name_list {' '.join(model_name_list)} \\")
        print(f"    --ged_weight {ged_weight} \\")
        print(f"    --max_iterations 100 \\")
        print(f"    --num_workers 4 \\")
        print(f"    --pathway_output_folder {pathway_output_folder}")
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ Demo setup completed in {elapsed_time:.2f} seconds")
        print(f"✓ Demo dataset created: {demo_file}")
        print(f"✓ Output directory created: {pathway_output_folder}")
        
    except Exception as e:
        print(f"\nError during demo execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_demo()
