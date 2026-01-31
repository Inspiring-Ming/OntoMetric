#!/usr/bin/env python3
"""
Master Script: Run All Analyses
Executes all visualization and analysis scripts for the Ontometric pipeline results
"""

import subprocess
import sys
from pathlib import Path


def run_script(script_path, description):
    """Run a Python script and report status"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Script: {script_path.name}")
    print(f"{'='*80}\n")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent.parent,
            capture_output=False,
            text=True
        )

        if result.returncode == 0:
            print(f"\n✓ {description} completed successfully")
            return True
        else:
            print(f"\n✗ {description} failed with exit code {result.returncode}")
            return False

    except Exception as e:
        print(f"\n✗ {description} failed with error: {str(e)}")
        return False


def main():
    """Run all analysis scripts"""
    base_dir = Path(__file__).parent
    scripts_dir = base_dir / "scripts"

    print("\n" + "="*80)
    print("ONTOMETRIC PIPELINE - COMPREHENSIVE ANALYSIS SUITE")
    print("="*80)
    print(f"\nBase directory: {base_dir}")
    print(f"Scripts directory: {scripts_dir}")

    # Define all scripts to run
    scripts = [
        {
            'path': scripts_dir / "analyze_cost.py",
            'description': "Cost Efficiency Analysis"
        },
        {
            'path': scripts_dir / "analyze_stage2_ontology.py",
            'description': "Stage 2 Ontology Distribution Analysis"
        },
        {
            'path': scripts_dir / "analyze_stage2_comparison.py",
            'description': "Stage 2 Method Comparison (Baseline vs Ontology-Guided)"
        },
        {
            'path': scripts_dir / "analyze_stage3_ontology.py",
            'description': "Stage 3 Ontology Validation Analysis"
        },
        {
            'path': scripts_dir / "analyze_stage3_comparison.py",
            'description': "Stage 3 Validation Comparison (Quality & Filtering)"
        }
    ]

    # Check if scripts exist
    missing_scripts = [s for s in scripts if not s['path'].exists()]
    if missing_scripts:
        print("\n✗ ERROR: Missing scripts:")
        for s in missing_scripts:
            print(f"  - {s['path']}")
        sys.exit(1)

    # Run all scripts
    results = []
    for script in scripts:
        success = run_script(script['path'], script['description'])
        results.append({
            'script': script['path'].name,
            'description': script['description'],
            'success': success
        })

    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS SUITE SUMMARY")
    print("="*80)

    for result in results:
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"{status}: {result['description']}")

    total_success = sum(1 for r in results if r['success'])
    total = len(results)

    print(f"\nTotal: {total_success}/{total} scripts completed successfully")

    if total_success == total:
        print("\n✓ ALL ANALYSES COMPLETED SUCCESSFULLY!")
        print("\nGenerated outputs:")
        print("  • cost_analysis/ - Cost efficiency analysis (wasted vs useful spending)")
        print("  • stage2_ontology/ - Ontology-guided entity/relationship distributions")
        print("  • stage2_comparison/ - Baseline vs Ontology-Guided extraction comparison")
        print("  • stage3_ontology/ - Ontology-guided validation detailed analysis")
        print("  • stage3_comparison/ - Validation quality and filtering impact analysis")
        sys.exit(0)
    else:
        print(f"\n✗ {total - total_success} script(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
