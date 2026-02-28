"""
Flask web server to visualize evaluation results.
Allows navigation between examples and viewing all details.
"""
import json
import os
from flask import Flask, render_template_string, jsonify, request
from typing import List, Dict

app = Flask(__name__)

# HTML template for the visualization page
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Results Visualization</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .navigation {
            background: #f8f9fa;
            padding: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #e9ecef;
        }
        
        .nav-buttons {
            display: flex;
            gap: 10px;
        }
        
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .btn-primary {
            background: #667eea;
            color: white;
        }
        
        .btn-primary:hover {
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        
        .btn-secondary {
            background: #6c757d;
            color: white;
        }
        
        .btn-secondary:hover {
            background: #5a6268;
        }
        
        .example-counter {
            font-size: 18px;
            font-weight: 600;
            color: #495057;
        }
        
        .content {
            padding: 30px;
        }
        
        .section {
            margin-bottom: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        
        .section h2 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        
        .section h3 {
            color: #495057;
            margin-top: 15px;
            margin-bottom: 10px;
            font-size: 1.2em;
        }
        
        .score-display {
            display: inline-block;
            padding: 10px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 8px;
            font-size: 1.5em;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .question-box {
            background: white;
            padding: 20px;
            border-radius: 8px;
            border: 2px solid #e9ecef;
            margin: 15px 0;
            font-size: 1.1em;
            line-height: 1.6;
        }
        
        .variables-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }
        
        .variable-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border: 2px solid #e9ecef;
            text-align: center;
        }
        
        .variable-name {
            font-weight: bold;
            color: #667eea;
            font-size: 1.1em;
            margin-bottom: 5px;
        }
        
        .variable-value {
            font-size: 1.2em;
            color: #495057;
        }
        
        .variable-unit {
            font-size: 0.9em;
            color: #6c757d;
            margin-top: 5px;
        }
        
        .explanation-box {
            background: #fff3cd;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
            margin: 15px 0;
            line-height: 1.6;
        }
        
        .formula-list {
            list-style: none;
            padding: 0;
        }
        
        .formula-item {
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #28a745;
        }
        
        .formula-step {
            font-weight: bold;
            color: #28a745;
            margin-bottom: 5px;
        }
        
        .target-badge {
            display: inline-block;
            padding: 8px 16px;
            background: #28a745;
            color: white;
            border-radius: 20px;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .length-badge {
            display: inline-block;
            padding: 8px 16px;
            background: #17a2b8;
            color: white;
            border-radius: 20px;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .loading {
            text-align: center;
            padding: 50px;
            font-size: 1.2em;
            color: #6c757d;
        }
        
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 20px;
            border-radius: 8px;
            margin: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Evaluation Results Visualization</h1>
            <p>Navigate through examples and view detailed evaluation metrics</p>
        </div>
        
        <div class="navigation">
            <div class="nav-buttons">
                <button class="btn-primary" onclick="loadExample(0)">⏮ First</button>
                <button class="btn-primary" onclick="loadExample(currentIndex - 1)">◀ Previous</button>
                <button class="btn-primary" onclick="loadExample(currentIndex + 1)">Next ▶</button>
                <button class="btn-primary" onclick="loadExample(examples.length - 1)">Last ⏭</button>
            </div>
            <div class="example-counter">
                Example <span id="current-num">1</span> of <span id="total-num">{{ total }}</span>
            </div>
            <div class="nav-buttons">
                <button class="btn-secondary" onclick="goToExample()">Go to #</button>
                <input type="number" id="goto-input" min="1" max="{{ total }}" value="1" style="width: 80px; padding: 8px; border-radius: 5px; border: 1px solid #ccc;">
            </div>
        </div>
        
        <div class="content" id="content">
            <div class="loading">Loading example...</div>
        </div>
    </div>
    
    <script>
        const examples = {{ examples|safe }};
        let currentIndex = 0;
        
        function loadExample(index) {
            if (index < 0) index = 0;
            if (index >= examples.length) index = examples.length - 1;
            
            currentIndex = index;
            const example = examples[index];
            
            document.getElementById('current-num').textContent = index + 1;
            document.getElementById('goto-input').value = index + 1;
            
            const content = document.getElementById('content');
            
            // Build HTML content
            let html = `
                <div class="section">
                    <h2>📈 Score & Metrics</h2>
                    <div class="score-display">Faithfulness Score: ${example.score !== null && example.score !== undefined ? example.score.toFixed(2) : 'N/A'}/10</div>
                    <div class="length-badge">Traversal Length: ${example.length || 'N/A'}</div>
                    ${example.target_variable ? `<div class="target-badge">Target Variable: ${escapeHtml(example.target_variable)}</div>` : ''}
                    ${example.trace_depth !== undefined ? `<div style="margin-top: 10px; color: #6c757d;">Trace Depth: ${example.trace_depth}</div>` : ''}
                </div>
                
                <div class="section">
                    <h2>❓ Generated Question</h2>
                    <div class="question-box">${escapeHtml(example.question || example.generated_question || 'No question generated')}</div>
                </div>
                
                ${example.given_variables && Object.keys(example.given_variables).length > 0 ? `
                <div class="section">
                    <h2>📋 Given Variables</h2>
                    <div class="variables-grid">
                        ${Object.entries(example.given_variables).map(([name, data]) => {
                            const value = (data && typeof data === 'object' && 'value' in data) ? data.value : data;
                            const unit = (data && typeof data === 'object' && 'unit' in data) ? data.unit : '';
                            return `
                                <div class="variable-card">
                                    <div class="variable-name">${escapeHtml(name)}</div>
                                    <div class="variable-value">${value !== null && value !== undefined ? value : 'N/A'}</div>
                                    ${unit ? `<div class="variable-unit">${escapeHtml(unit)}</div>` : ''}
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
                ` : ''}
                
                ${example.faithfulness_explanation ? `
                <div class="section">
                    <h2>💡 Faithfulness Explanation</h2>
                    <div class="explanation-box">${escapeHtml(example.faithfulness_explanation)}</div>
                </div>
                ` : ''}
                
                ${example.calculation_steps && Array.isArray(example.calculation_steps) && example.calculation_steps.length > 0 ? `
                <div class="section">
                    <h2>🔢 Calculation Steps</h2>
                    <ul class="formula-list">
                        ${example.calculation_steps.map((step, idx) => `
                            <li class="formula-item">
                                <div class="formula-step">Step ${idx + 1}</div>
                                <div>${escapeHtml(String(step))}</div>
                            </li>
                        `).join('')}
                    </ul>
                </div>
                ` : ''}
                
                ${example.formulas && Array.isArray(example.formulas) && example.formulas.length > 0 ? `
                <div class="section">
                    <h2>🧮 Formulas Used</h2>
                    <ul class="formula-list">
                        ${example.formulas.map((formula, idx) => {
                            const output = (formula && typeof formula === 'object' && 'output' in formula) ? formula.output : 'N/A';
                            const formulaText = (formula && typeof formula === 'object' && 'formula' in formula) ? formula.formula : (formula && typeof formula === 'object' && 'label' in formula) ? formula.label : 'N/A';
                            const inputs = (formula && typeof formula === 'object' && 'inputs' in formula) ? formula.inputs : [];
                            return `
                                <li class="formula-item">
                                    <div class="formula-step">Formula ${idx + 1}</div>
                                    <div><strong>Output:</strong> ${escapeHtml(String(output))}</div>
                                    <div><strong>Formula:</strong> ${escapeHtml(String(formulaText))}</div>
                                    ${inputs.length > 0 ? `<div><strong>Inputs:</strong> ${escapeHtml(inputs.join(', '))}</div>` : ''}
                                </li>
                            `;
                        }).join('')}
                    </ul>
                </div>
                ` : ''}
            `;
            
            content.innerHTML = html;
        }
        
        function goToExample() {
            const input = document.getElementById('goto-input');
            const num = parseInt(input.value);
            if (num >= 1 && num <= examples.length) {
                loadExample(num - 1);
            }
        }
        
        function escapeHtml(text) {
            if (text === null || text === undefined) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') {
                loadExample(currentIndex - 1);
            } else if (e.key === 'ArrowRight') {
                loadExample(currentIndex + 1);
            }
        });
        
        // Load first example on page load
        loadExample(0);
    </script>
</body>
</html>
"""


def load_jsonl_data(file_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    if not os.path.exists(file_path):
        return data
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    data.append(entry)
                except json.JSONDecodeError:
                    continue
    return data


@app.route('/')
def index():
    """Main page showing the first example."""
    default_file = os.environ.get('DEFAULT_JSONL_FILE', 'checkpoints/logs/instruction_model_evaluation_20260205_192630.jsonl')
    jsonl_file = request.args.get('file', default_file)
    
    # Load data
    examples = load_jsonl_data(jsonl_file)
    
    if not examples:
        return render_template_string("""
        <div class="error">
            <h2>Error: No data found</h2>
            <p>Could not load examples from: {{ file }}</p>
            <p>Make sure the JSONL file exists and contains valid data.</p>
        </div>
        """, file=jsonl_file)
    
    # Convert examples to JSON for JavaScript
    examples_json = json.dumps(examples)
    
    return render_template_string(HTML_TEMPLATE, examples=examples_json, total=len(examples))


@app.route('/api/examples')
def get_examples():
    """API endpoint to get all examples."""
    default_file = os.environ.get('DEFAULT_JSONL_FILE', 'checkpoints/logs/instruction_model_evaluation_20260205_192630.jsonl')
    jsonl_file = request.args.get('file', default_file)
    examples = load_jsonl_data(jsonl_file)
    return jsonify(examples)


@app.route('/api/example/<int:index>')
def get_example(index):
    """API endpoint to get a specific example by index."""
    default_file = os.environ.get('DEFAULT_JSONL_FILE', 'checkpoints/logs/instruction_model_evaluation_20260205_192630.jsonl')
    jsonl_file = request.args.get('file', default_file)
    examples = load_jsonl_data(jsonl_file)
    
    if 0 <= index < len(examples):
        return jsonify(examples[index])
    else:
        return jsonify({"error": "Index out of range"}), 404


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Flask web server for visualizing evaluation results")
    parser.add_argument(
        "--file",
        type=str,
        default="checkpoints/logs/instruction_model_evaluation_20260205_192630.jsonl",
        help="Path to JSONL file with evaluation results"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to (default: 5000)"
    )
    
    args = parser.parse_args()
    
    # Set default file as environment variable for Flask
    os.environ['DEFAULT_JSONL_FILE'] = args.file
    
    print(f"Starting Flask server...")
    print(f"Visualization available at: http://{args.host}:{args.port}")
    print(f"Loading data from: {args.file}")
    
    app.run(host=args.host, port=args.port, debug=True)
