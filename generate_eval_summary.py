"""
Generate a summary report from GSM8K evaluation results.

Usage:
    python generate_eval_summary.py results.json
    python generate_eval_summary.py results.json --output summary.txt
"""
import json
import argparse
import os
from datetime import datetime

def generate_summary(results_file, output_file=None):
    """Generate a text summary from evaluation results."""
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if output_file is None:
        output_file = os.path.splitext(results_file)[0] + '_summary.txt'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("GSM8K Evaluation Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Results file: {results_file}\n")
        f.write("\n")
        
        # Evaluation parameters
        f.write("Evaluation Parameters:\n")
        f.write("-" * 80 + "\n")
        eval_params = data.get('eval_params', {})
        for key, value in eval_params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Results summary
        f.write("Results Summary:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<15} {'Repository':<35} {'Step':<10} {'Accuracy':<12} {'Correct/Total':<15}\n")
        f.write("-" * 80 + "\n")
        
        summary = data.get('summary', [])
        for result in summary:
            f.write(f"{result['model_name']:<15} {result['hf_repo_id']:<35} {result['step']:<10} "
                   f"{result['accuracy_percent']:>6.2f}% {result.get('num_correct', 0)}/{result.get('num_total', 0)}\n")
        f.write("\n")
        
        # Best model
        if summary:
            best = max(summary, key=lambda x: x['accuracy'])
            f.write(f"Best Model: {best['model_name']}\n")
            f.write(f"  Accuracy: {best['accuracy_percent']:.2f}%\n")
            f.write(f"  Repository: {best['hf_repo_id']}\n")
            f.write(f"  Step: {best['step']}\n")
            f.write(f"  Correct: {best.get('num_correct', 0)}/{best.get('num_total', 0)}\n")
            f.write("\n")
        
        # Detailed statistics per model
        f.write("Detailed Statistics:\n")
        f.write("-" * 80 + "\n")
        detailed = data.get('detailed_results', {})
        for model_name, model_data in detailed.items():
            problems = model_data.get('problems', [])
            if not problems:
                continue
            
            correct = sum(1 for p in problems if p.get('correct', False))
            total = len(problems)
            accuracy = (correct / total * 100) if total > 0 else 0
            
            f.write(f"\n{model_name}:\n")
            f.write(f"  Total Problems: {total}\n")
            f.write(f"  Correct: {correct}\n")
            f.write(f"  Incorrect: {total - correct}\n")
            f.write(f"  Accuracy: {accuracy:.2f}%\n")
            
            # Show some example problems
            incorrect_problems = [p for p in problems if not p.get('correct', False)][:5]
            if incorrect_problems:
                f.write(f"  Sample Incorrect Problems (showing first 5):\n")
                for prob in incorrect_problems:
                    f.write(f"    Problem {prob.get('problem_id', '?')}: "
                           f"Expected {prob.get('ground_truth_answer', 'N/A')}, "
                           f"Got {prob.get('predicted_answer', 'N/A')}\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write(f"Full results: {results_file}\n")
        f.write("=" * 80 + "\n")
    
    print(f"Summary saved to: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Generate summary report from GSM8K evaluation results')
    parser.add_argument('results_file', type=str, help='Path to JSON results file')
    parser.add_argument('--output', type=str, default=None, help='Output summary file (default: results_file_summary.txt)')
    args = parser.parse_args()
    
    generate_summary(args.results_file, args.output)

if __name__ == "__main__":
    main()

