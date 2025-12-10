from pyvis.network import Network
import networkx as nx
import json

file_path = "variable_concept_graph.json"
with open(file_path, 'r') as f:
    data = json.load(f)

G = nx.DiGraph()
defined_variables = {v['variable'] for v in data['variables']}

# --- build the graph ---
for variable_data in data['variables']:
    variable = variable_data['variable']
    formulas_list = variable_data['formulas']

    tooltip = (
        f"Variable: {variable}\nFormulas:\n" +
        "\n".join([f"• {f}" for f in formulas_list])
    )

    G.add_node(
        variable,
        title=tooltip,
        color='skyblue',
        type='calculated'
    )

    for dependency in variable_data['dependencies']:
        if dependency not in defined_variables and dependency not in G:
            G.add_node(
                dependency,
                title=f"Base Input: {dependency}",
                color='lightcoral',
                type='input'
            )

        G.add_edge(
            dependency,
            variable,
            title=f"Uses {dependency} to calculate {variable}"
        )

# --- compute layout ONCE in Python ---
pos = nx.spring_layout(G, k=0.5, iterations=80)

net = Network(height="750px", width="100%", directed=True, notebook=False)

# add nodes with fixed positions
for node, data_dict in G.nodes(data=True):
    x, y = pos[node]
    net.add_node(
        node,
        x=float(x * 1000),
        y=float(y * 1000),
        physics=False,
        **data_dict
    )

# add edges
for u, v, data_dict in G.edges(data=True):
    net.add_edge(u, v, physics=False, **data_dict)

# --- disable physics & tune interactions (VALID JSON) ---
net.set_options("""
{
  "physics": {
    "enabled": false
  },
  "interaction": {
    "hover": true,
    "zoomView": true,
    "dragView": true,
    "dragNodes": true
  }
}
""")

output_file = "variable_dependency_graph_interactive.html"
net.write_html(output_file, notebook=False)

# optional: hide the loading bar by patching the HTML
with open(output_file, "r", encoding="utf-8") as f:
    html = f.read()

html = html.replace('id="loadingBar"', 'id="loadingBar" style="display:none"')
html = html.replace('id="loadingBarText"', 'id="loadingBarText" style="display:none"')

with open(output_file, "w", encoding="utf-8") as f:
    f.write(html)

print("Saved:", output_file)
