import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from pathlib import Path  # Cross-platform path handling

# Define file paths using Path for cross-platform compatibility
nodes_file = Path('network_nodes.csv')
edges_file = Path('network_edges.csv')

# Check if the files exist before loading
if not nodes_file.exists() or not edges_file.exists():
    st.error("Data files for nodes and edges are missing. Please ensure 'network_nodes.csv' and 'network_edges.csv' are in the same directory.")
else:
    # Load precomputed nodes and edges
    nodes = pd.read_csv(nodes_file)
    edges = pd.read_csv(edges_file)

    # Build network from saved data
    G = nx.from_pandas_edgelist(edges, source='source', target='target')
    community = nodes.set_index('user')['community'].to_dict()
    nx.set_node_attributes(G, community, 'community')

    # Visualize network
    pos = nx.spring_layout(G)
    fig = go.Figure()

    # Plot edges
    for edge in G.edges():
        fig.add_trace(go.Scatter(
            x=[pos[edge[0]][0], pos[edge[1]][0]],
            y=[pos[edge[0]][1], pos[edge[1]][1]],
            mode='lines',
            line=dict(width=0.5, color='blue')
        ))

    # Plot nodes
    for node in G.nodes():
        fig.add_trace(go.Scatter(
            x=[pos[node][0]], y=[pos[node][1]],
            mode='markers+text',
            text=[node],
            marker=dict(size=5, color='orange')
        ))

    # Update layout for better display
    fig.update_layout(
        title="User Interaction Network",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False)
    )

    st.plotly_chart(fig)
