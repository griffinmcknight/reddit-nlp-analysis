import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

# Load precomputed nodes and edges
nodes = pd.read_csv('network_nodes.csv')
edges = pd.read_csv('network_edges.csv')

# Build network from saved data
G = nx.from_pandas_edgelist(edges, source='source', target='target')
community = nodes.set_index('user')['community'].to_dict()
nx.set_node_attributes(G, community, 'community')

# Visualize network
pos = nx.spring_layout(G)
fig = go.Figure()
for edge in G.edges():
    fig.add_trace(go.Scatter(x=[pos[edge[0]][0], pos[edge[1]][0]], y=[pos[edge[0]][1], pos[edge[1]][1]],
                             mode='lines', line=dict(width=0.5, color='blue')))
for node in G.nodes():
    fig.add_trace(go.Scatter(x=[pos[node][0]], y=[pos[node][1]], mode='markers+text',
                             text=[node], marker=dict(size=5, color='orange')))
fig.update_layout(title="User Interaction Network", showlegend=False)
st.plotly_chart(fig)
