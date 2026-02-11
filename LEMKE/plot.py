import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import math


def visualize_stage_flow(eq_list, relationship_log, filename="stage_flow_map.png"):

    G = nx.DiGraph()
    
    # 1. Group nodes by stage to calculate local vertical spacing
    nodes_by_stage = defaultdict(list)
    for eq in eq_list:
        G.add_node(eq.ID, stage=eq.stage)
        nodes_by_stage[eq.stage].append(eq.ID)

    # 2. Setup Layout: X = Stage, Y = Rank within that stage (centered)
    pos = {}
    vertical_gap = 1.5  # Increase this for "ample space" between rows
    
    for stage, nodes in nodes_by_stage.items():
        # Sort by ID to keep the order consistent
        sorted_nodes = sorted(nodes)
        num_nodes = len(sorted_nodes)
        
        # Centering logic: calculate offset so the column is centered at Y=0
        offset = (num_nodes - 1) * vertical_gap / 2.0
        
        for i, node_id in enumerate(sorted_nodes):
            # X = stage, Y = rank * gap - offset
            pos[node_id] = (stage, -(i * vertical_gap) + offset)

    # 3. Categorize Edges (Discovery vs. Return)
    discovery_edges = []
    return_edges = []
    
    for base_id, target_id, prior_id, child_id, flag in relationship_log:
        if child_id != "None":
            G.add_edge(base_id, child_id)
            if G.nodes[child_id]['stage'] > G.nodes[base_id]['stage']:
                discovery_edges.append((base_id, child_id))
            else:
                return_edges.append((base_id, child_id))

    # 4. Setup Figure
    plt.figure(figsize=(24, 18)) # Wide and tall for readability
    
    # Node Styling
    stage_colors = {1:'#1f77b4', 2:'#2ca02c', 3:'#bcbd22', 4:'#ff7f0e', 5:'#d62728', 6:'#9467bd', 7:'#8c564b'}
    colors = [stage_colors.get(G.nodes[n]['stage'], 'gray') for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=1200, alpha=1.0, edgecolors='white', linewidths=2)

    # 5. Lighter Arrows (Subtle gray for high readability)
    nx.draw_networkx_edges(
        G, pos, edgelist=discovery_edges, 
        edge_color='silver', arrows=True, alpha=0.5, width=1.0,
        arrowsize=15, connectionstyle="arc3,rad=0.05" # Flatter curves
    )
    
    nx.draw_networkx_edges(
        G, pos, edgelist=return_edges, 
        edge_color='cornflowerblue', style='--', arrows=True, alpha=0.5, width=1.0,
        arrowsize=12, connectionstyle="arc3,rad=0.2"
    )

    # Labeling
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='white', font_weight='bold')
    
    # Final Touches
    plt.title("Equilibrium Discovery Stage Flow Map\n", fontsize=22, pad=20)
    plt.xlabel("Discovery Stage", fontsize=16)
    plt.xticks(range(1, max(nodes_by_stage.keys()) + 1))
    plt.grid(axis='x', linestyle=':', color='gray', alpha=0.2)
    
    plt.axis('on') # Keep axis to show stage numbers
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.yticks([]) # Hide Y-axis numbers as they don't represent values
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"LOG: Improved flow map saved to {filename}")
    
    
def visualize_stage_flow_with_attraction(eq_list, relationship_log, filename="stage_flow.png"):

    G = nx.DiGraph()
    node_sizes_dict = {}
    nodes_by_stage = defaultdict(list)
    
    # 1. Scaling and Grouping
    for eq in eq_list:
        G.add_node(eq.ID, stage=eq.stage)
        nodes_by_stage[eq.stage].append(eq.ID)
        attr = len(eq.priors) if eq.stage == 1 else len(eq.parent)
        node_sizes_dict[eq.ID] = 600 + (math.log1p(attr) * 800)

    # 2. Dynamic Layout (Large Vertical Spacing)
    pos = {}
    H_GAP, V_BUFFER = 3.5, 6.0
    for stage, nodes in nodes_by_stage.items():
        sorted_nodes = sorted(nodes)
        radii = [math.sqrt(node_sizes_dict[n]) / 35 for n in sorted_nodes]
        total_height = sum(radii) * 2 + (len(sorted_nodes) - 1) * V_BUFFER
        y_cursor = total_height / 2.0
        for i, n_id in enumerate(sorted_nodes):
            y_cursor -= radii[i]
            pos[n_id] = (stage * H_GAP, y_cursor)
            y_cursor -= (radii[i] + V_BUFFER)

    # 3. Edge Categorization
    discovery_edges = []
    return_edges = []
    for b_id, t_id, p_id, c_id, flag in relationship_log:
        if c_id != "None":
            G.add_edge(b_id, c_id)
            if G.nodes[c_id]['stage'] > G.nodes[b_id]['stage']:
                discovery_edges.append((b_id, c_id))
            else:
                return_edges.append((b_id, c_id))

    # 4. Rendering
    plt.figure(figsize=(28, 20))
    
    # Draw Nodes first so they are at the bottom layer
    sizes_list = [node_sizes_dict[n] for n in G.nodes()]
    stage_colors = {1:'#1f77b4', 2:'#2ca02c', 3:'#8c564b', 4:'#ff7f0e', 5:'#d62728', 6: '#7D27F5', 7: '#e377c2', 8:'#F2C305'}
    colors = [stage_colors.get(G.nodes[n]['stage'], 'gray') for n in G.nodes()]
    
    nx.draw_networkx_nodes(
        G, pos, 
        node_color=colors, 
        node_size=sizes_list, 
        edgecolors='darkgrey',   # THE BORDER COLOR
        linewidths=2.5,       # THE BORDER THICKNESS
        alpha=1.0
    )

    # DRAW EDGES WITH MARGINS
    # min_target_margin stops the arrowhead short of the node
    # passing node_size=sizes_list tells the function where the node edge is
    nx.draw_networkx_edges(
        G, pos, edgelist=discovery_edges, 
        node_size=sizes_list,       # Crucial: Tells edges to respect node radii
        min_target_margin=25,       # Forces arrow to stop 25 units early
        edge_color='silver', arrows=True, alpha=0.4, width=1.2, arrowsize=22,
        connectionstyle="arc3,rad=0.08"
    )
    
    nx.draw_networkx_edges(
        G, pos, edgelist=return_edges, 
        node_size=sizes_list,
        min_target_margin=30,       # More margin for return paths to handle curves
        edge_color='cornflowerblue', style='--', arrows=True, alpha=0.4, 
        width=1.2, arrowsize=18, connectionstyle="arc3,rad=0.25"
    )

    nx.draw_networkx_labels(G, pos, font_size=9, font_color='white', font_weight='bold')
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()