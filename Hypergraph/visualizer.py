"""
Visualization tool for hypergraph solution traces.

Provides a web-based interface to visualize all solution methods for a target variable.
"""
import json
import os
from typing import Dict, List
from hypergraph_traverser import HypergraphTraverser
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Global traverser instance
traverser = None


def init_traverser(hypergraph_file: str = "formula_hypergraph.json"):
    """Initialize the traverser."""
    global traverser
    traverser = HypergraphTraverser(hypergraph_file)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Hypergraph Solution Traces Visualizer</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
        }
        .input-section {
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .input-group {
            display: flex;
            gap: 15px;
            align-items: center;
        }
        input[type="text"] {
            flex: 1;
            padding: 12px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 5px;
            transition: border-color 0.3s;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            padding: 12px 30px;
            font-size: 16px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.3s;
        }
        button:hover {
            background: #5568d3;
        }
        .results {
            display: grid;
            gap: 20px;
        }
        .trace-card {
            background: white;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .trace-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        .trace-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #f0f0f0;
        }
        .trace-title {
            font-size: 1.5em;
            color: #333;
            font-weight: bold;
        }
        .trace-meta {
            color: #666;
            font-size: 0.9em;
        }
        .nodes-section {
            margin: 15px 0;
        }
        .nodes-section h4 {
            margin: 10px 0 5px 0;
            color: #555;
        }
        .node-tag {
            display: inline-block;
            padding: 5px 12px;
            margin: 5px 5px 5px 0;
            border-radius: 20px;
            font-size: 0.9em;
        }
        .leaf-node {
            background-color: #e8f5e9;
            color: #2e7d32;
        }
        .cycle-node {
            background-color: #fff3e0;
            color: #e65100;
        }
        .formula-step {
            background: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #667eea;
            border-radius: 5px;
        }
        .formula-step-header {
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
        }
        .formula-expression {
            font-family: 'Courier New', monospace;
            background: white;
            padding: 10px;
            border-radius: 3px;
            margin: 8px 0;
            color: #d63384;
        }
        .formula-inputs {
            margin-top: 8px;
            font-size: 0.9em;
            color: #666;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .stats {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            flex: 1;
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Hypergraph Solution Traces Visualizer</h1>
        <p>Explore all possible methods to calculate a target variable using physics formulas</p>
    </div>
    
    <div class="input-section">
        <div class="input-group">
            <input type="text" id="targetInput" placeholder="Enter target variable (e.g., acceleration, kinetic_energy)" 
                   value="{{ default_target }}" onkeypress="if(event.key==='Enter') findTraces()">
            <button onclick="findTraces()">Find Solution Traces</button>
        </div>
    </div>
    
    <div id="results"></div>
    
    <script>
        function findTraces() {
            const target = document.getElementById('targetInput').value.trim();
            if (!target) {
                alert('Please enter a target variable');
                return;
            }
            
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<div class="loading">🔍 Finding solution traces...</div>';
            
            fetch(`/api/traces?target=${encodeURIComponent(target)}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        resultsDiv.innerHTML = `<div class="error">❌ Error: ${data.error}</div>`;
                        return;
                    }
                    
                    displayResults(data);
                })
                .catch(error => {
                    resultsDiv.innerHTML = `<div class="error">❌ Error: ${error.message}</div>`;
                });
        }
        
        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            
            if (data.traces.length === 0) {
                resultsDiv.innerHTML = '<div class="error">No solution traces found for this target variable.</div>';
                return;
            }
            
            let html = `
                <div class="stats">
                    <div class="stat-box">
                        <div class="stat-value">${data.traces.length}</div>
                        <div class="stat-label">Solution Methods</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">${data.target}</div>
                        <div class="stat-label">Target Variable</div>
                    </div>
                </div>
            `;
            
            data.traces.forEach((trace, index) => {
                html += `
                    <div class="trace-card">
                        <div class="trace-header">
                            <div class="trace-title">Method ${index + 1}</div>
                            <div class="trace-meta">Depth: ${trace.depth} | Formulas: ${trace.num_formulas}</div>
                        </div>
                        
                        ${trace.leaf_nodes.length > 0 ? `
                            <div class="nodes-section">
                                <h4>📥 Given Values (Leaf Nodes):</h4>
                                ${trace.leaf_nodes.map(node => `<span class="node-tag leaf-node">${node}</span>`).join('')}
                            </div>
                        ` : ''}
                        
                        ${trace.cycle_nodes.length > 0 ? `
                            <div class="nodes-section">
                                <h4>⚠️ Cycle Nodes (Assumed Given):</h4>
                                ${trace.cycle_nodes.map(node => `<span class="node-tag cycle-node">${node}</span>`).join('')}
                            </div>
                        ` : ''}
                        
                        <div class="nodes-section">
                            <h4>📐 Calculation Steps:</h4>
                            ${trace.formulas.map(formula => `
                                <div class="formula-step">
                                    <div class="formula-step-header">
                                        Step ${formula.step}: Calculate <strong>${formula.output}</strong> 
                                        ${formula.output_si_unit ? `(${formula.output_si_unit})` : ''}
                                    </div>
                                    <div class="formula-expression">${formula.formula}</div>
                                    <div class="formula-inputs">
                                        <strong>Inputs:</strong> ${formula.inputs.join(', ')}
                                        ${Object.keys(formula.input_si_units).length > 0 ? 
                                            '<br><strong>Units:</strong> ' + 
                                            formula.inputs.map(inp => 
                                                `${inp} (${formula.input_si_units[inp] || 'N/A'})`
                                            ).join(', ') 
                                            : ''}
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `;
            });
            
            resultsDiv.innerHTML = html;
        }
        
        // Load traces on page load if default target is set
        {% if default_target %}
        window.onload = function() {
            findTraces();
        };
        {% endif %}
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Main page."""
    default_target = request.args.get('target', 'acceleration')
    return render_template_string(HTML_TEMPLATE, default_target=default_target)


@app.route('/api/traces')
def api_traces():
    """API endpoint to get traces for a target variable."""
    target = request.args.get('target', '').strip()
    
    if not target:
        return jsonify({'error': 'Target variable is required'}), 400
    
    try:
        traces = traverser.get_all_traces_formatted(target, max_depth=10, max_traces=50)
        
        return jsonify({
            'target': target,
            'traces': traces,
            'count': len(traces)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Initialize traverser
    hypergraph_file = os.path.join(os.path.dirname(__file__), "formula_hypergraph.json")
    init_traverser(hypergraph_file)
    
    print("Starting Hypergraph Visualizer...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

