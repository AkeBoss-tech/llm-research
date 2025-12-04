"""
Generate an HTML visualization of GSM8K evaluation results.

Usage:
    python visualize_gsm8k_results.py results.json
    python visualize_gsm8k_results.py results.json --output report.html
"""
import json
import argparse
import os
from datetime import datetime

def generate_html(data, output_file):
    """Generate HTML visualization from evaluation results."""
    
    models = list(data['detailed_results'].keys())
    summary = data['summary']
    
    # Calculate statistics
    total_problems = len(data['detailed_results'][models[0]]['problems']) if models else 0
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GSM8K Evaluation Results</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        
        h1 {{
            color: #2c3e50;
            margin-bottom: 10px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        
        .header-info {{
            color: #7f8c8d;
            margin-bottom: 30px;
            font-size: 14px;
        }}
        
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .summary-card h3 {{
            margin-bottom: 10px;
            font-size: 14px;
            opacity: 0.9;
        }}
        
        .summary-card .value {{
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .summary-card .subtitle {{
            font-size: 12px;
            opacity: 0.8;
        }}
        
        .controls {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        
        .controls-row {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }}
        
        .controls-row label {{
            font-weight: 600;
            color: #2c3e50;
        }}
        
        .controls-row input, .controls-row select {{
            padding: 8px 12px;
            border: 2px solid #bdc3c7;
            border-radius: 4px;
            font-size: 14px;
        }}
        
        .controls-row input:focus, .controls-row select:focus {{
            outline: none;
            border-color: #3498db;
        }}
        
        .problem-grid {{
            display: grid;
            gap: 20px;
        }}
        
        .problem-card {{
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            background: white;
            transition: all 0.3s ease;
        }}
        
        .problem-card:hover {{
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }}
        
        .problem-card.correct {{
            border-left: 5px solid #27ae60;
            background: #f8fff9;
        }}
        
        .problem-card.incorrect {{
            border-left: 5px solid #e74c3c;
            background: #fff8f8;
        }}
        
        .problem-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .problem-id {{
            font-weight: 600;
            color: #7f8c8d;
        }}
        
        .status-badge {{
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        
        .status-badge.correct {{
            background: #27ae60;
            color: white;
        }}
        
        .status-badge.incorrect {{
            background: #e74c3c;
            color: white;
        }}
        
        .problem-content {{
            margin-bottom: 15px;
        }}
        
        .problem-section {{
            margin-bottom: 15px;
        }}
        
        .problem-section h4 {{
            color: #2c3e50;
            margin-bottom: 8px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .problem-section p {{
            background: #f8f9fa;
            padding: 12px;
            border-radius: 4px;
            border-left: 3px solid #3498db;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        
        .answer-comparison {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        
        .answer-box {{
            padding: 12px;
            border-radius: 4px;
        }}
        
        .answer-box.ground-truth {{
            background: #e8f5e9;
            border-left: 3px solid #27ae60;
        }}
        
        .answer-box.predicted {{
            background: #fff3e0;
            border-left: 3px solid #f39c12;
        }}
        
        .answer-box h5 {{
            font-size: 12px;
            text-transform: uppercase;
            margin-bottom: 5px;
            color: #7f8c8d;
        }}
        
        .answer-box .value {{
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .model-comparison {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin-top: 10px;
        }}
        
        .model-result {{
            padding: 10px;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
        }}
        
        .model-result.correct {{
            background: #e8f5e9;
        }}
        
        .model-result.incorrect {{
            background: #ffebee;
        }}
        
        .model-name {{
            font-weight: 600;
            margin-bottom: 5px;
        }}
        
        .hidden {{
            display: none;
        }}
        
        .stats-bar {{
            display: flex;
            gap: 10px;
            margin-top: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 4px;
        }}
        
        .stat-item {{
            flex: 1;
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .stat-label {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>GSM8K Evaluation Results</h1>
        <div class="header-info">
            Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | 
            Total Problems: {total_problems} |
            Models Evaluated: {len(models)}
        </div>
        
        <div class="summary">
"""
    
    # Add summary cards for each model
    for model_summary in summary:
        html += f"""
            <div class="summary-card">
                <h3>{model_summary['model_name']}</h3>
                <div class="value">{model_summary['accuracy_percent']:.2f}%</div>
                <div class="subtitle">{model_summary['num_correct']} / {model_summary['num_total']} correct</div>
            </div>
"""
    
    html += """
        </div>
        
        <div class="controls">
            <div class="controls-row">
                <label for="filter-status">Filter by Status:</label>
                <select id="filter-status">
                    <option value="all">All</option>
                    <option value="correct">Correct Only</option>
                    <option value="incorrect">Incorrect Only</option>
                </select>
                
                <label for="filter-model">Filter by Model:</label>
                <select id="filter-model">
                    <option value="all">All Models</option>
"""
    
    for model in models:
        html += f'<option value="{model}">{model}</option>\n'
    
    html += """
                </select>
                
                <label for="search">Search:</label>
                <input type="text" id="search" placeholder="Search questions...">
            </div>
        </div>
        
        <div class="stats-bar" id="stats-bar">
            <div class="stat-item">
                <div class="stat-value" id="stat-total">0</div>
                <div class="stat-label">Total Shown</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="stat-correct">0</div>
                <div class="stat-label">Correct</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="stat-incorrect">0</div>
                <div class="stat-label">Incorrect</div>
            </div>
            <div class="stat-item">
                <div class="stat-value" id="stat-accuracy">0%</div>
                <div class="stat-label">Accuracy</div>
            </div>
        </div>
        
        <div class="problem-grid" id="problem-grid">
"""
    
    # Add problem cards
    # Get problems from the first model (all models should have same problems)
    if models:
        first_model = models[0]
        problems = data['detailed_results'][first_model]['problems']
        
        for problem in problems:
            problem_id = problem['problem_id']
            question = problem['question']
            ground_truth = problem.get('ground_truth_answer', 'N/A')
            predicted = problem.get('predicted_answer', 'N/A')
            correct = problem.get('correct', False)
            model_response = problem.get('model_response', '')
            
            status_class = 'correct' if correct else 'incorrect'
            status_text = 'Correct' if correct else 'Incorrect'
            
            html += f"""
            <div class="problem-card {status_class}" data-problem-id="{problem_id}" data-status="{status_class}" data-models='{json.dumps({m: next((p.get("correct", False) for p in data["detailed_results"][m]["problems"] if p.get("problem_id") == problem_id), False) for m in models})}'>
                <div class="problem-header">
                    <span class="problem-id">Problem #{problem_id}</span>
                    <span class="status-badge {status_class}">{status_text}</span>
                </div>
                
                <div class="problem-content">
                    <div class="problem-section">
                        <h4>Question</h4>
                        <p>{question}</p>
                    </div>
                    
                    <div class="answer-comparison">
                        <div class="answer-box ground-truth">
                            <h5>Ground Truth</h5>
                            <div class="value">{ground_truth}</div>
                        </div>
                        <div class="answer-box predicted">
                            <h5>Predicted</h5>
                            <div class="value">{predicted}</div>
                        </div>
                    </div>
                    
                    <div class="problem-section">
                        <h4>Model Response</h4>
                        <p>{model_response[:500]}{'...' if len(model_response) > 500 else ''}</p>
                    </div>
"""
            
            # Add model comparison if multiple models
            if len(models) > 1:
                html += '<div class="problem-section"><h4>Model Comparison</h4><div class="model-comparison">'
                for model in models:
                    # Find the problem with matching problem_id in this model's results
                    model_problem = None
                    for p in data['detailed_results'][model]['problems']:
                        if p.get('problem_id') == problem_id:
                            model_problem = p
                            break
                    
                    if model_problem:
                        model_correct = model_problem.get('correct', False)
                        model_predicted = model_problem.get('predicted_answer', 'N/A')
                        model_class = 'correct' if model_correct else 'incorrect'
                        html += f"""
                        <div class="model-result {model_class}">
                            <div class="model-name">{model}</div>
                            <div>Predicted: {model_predicted}</div>
                            <div>Status: {'✓ Correct' if model_correct else '✗ Incorrect'}</div>
                        </div>
"""
                html += '</div></div>'
            
            html += """
                </div>
            </div>
"""
    
    html += """
        </div>
    </div>
    
    <script>
        const problems = document.querySelectorAll('.problem-card');
        const filterStatus = document.getElementById('filter-status');
        const filterModel = document.getElementById('filter-model');
        const searchInput = document.getElementById('search');
        const statTotal = document.getElementById('stat-total');
        const statCorrect = document.getElementById('stat-correct');
        const statIncorrect = document.getElementById('stat-incorrect');
        const statAccuracy = document.getElementById('stat-accuracy');
        
        function updateDisplay() {
            const statusFilter = filterStatus.value;
            const modelFilter = filterModel.value;
            const searchTerm = searchInput.value.toLowerCase();
            
            let visible = 0;
            let correct = 0;
            let incorrect = 0;
            
            problems.forEach(problem => {
                const status = problem.dataset.status;
                const question = problem.querySelector('.problem-section p').textContent.toLowerCase();
                const models = JSON.parse(problem.dataset.models || '{}');
                
                let show = true;
                
                // Status filter
                if (statusFilter !== 'all' && status !== statusFilter) {
                    show = false;
                }
                
                // Model filter
                if (modelFilter !== 'all' && !models[modelFilter]) {
                    show = false;
                }
                
                // Search filter
                if (searchTerm && !question.includes(searchTerm)) {
                    show = false;
                }
                
                if (show) {
                    problem.classList.remove('hidden');
                    visible++;
                    if (status === 'correct') correct++;
                    else incorrect++;
                } else {
                    problem.classList.add('hidden');
                }
            });
            
            statTotal.textContent = visible;
            statCorrect.textContent = correct;
            statIncorrect.textContent = incorrect;
            statAccuracy.textContent = visible > 0 ? ((correct / visible) * 100).toFixed(1) + '%' : '0%';
        }
        
        filterStatus.addEventListener('change', updateDisplay);
        filterModel.addEventListener('change', updateDisplay);
        searchInput.addEventListener('input', updateDisplay);
        
        // Initial update
        updateDisplay();
    </script>
</body>
</html>
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"HTML visualization saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate HTML visualization of GSM8K evaluation results')
    parser.add_argument('input_file', type=str, help='Path to JSON results file')
    parser.add_argument('--output', type=str, default=None, help='Output HTML file (default: same as input with .html extension)')
    args = parser.parse_args()
    
    # Load JSON data
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        output_file = os.path.splitext(args.input_file)[0] + '.html'
    
    # Generate HTML
    generate_html(data, output_file)
    print(f"Open {output_file} in your browser to view the results")

if __name__ == "__main__":
    main()

