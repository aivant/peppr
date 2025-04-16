from typing import Any
from collections import defaultdict

import networkx as nx
from networkx.algorithms import isomorphism


def find_node_induced_subgraphs(
    G: nx.Graph,
    H: nx.Graph,
    GraphMatcher: isomorphism.GraphMatcher = isomorphism.GraphMatcher,
    one_isomorphism: bool = True,
) -> list[tuple[nx.Graph, dict]]:
    """
    Returns views of all NODE-induced subgraph of G that are
    isomorphic ("match") the graph H. Matches are defined by the
    `GraphMatcher` class.

    Returns
        subgraphs: Node-induced subgraphs (views) of G that is isomorphic to H.
    """
    # Instantiate the graph matcher object
    gm = GraphMatcher(G, H)

    # Collect all node-induced isomorphic subgraphs
    matches = defaultdict(list)
    for G_to_H_nodes in gm.subgraph_isomorphisms_iter():
        G_nodes = G_to_H_nodes.keys()
        H_to_G_nodes = {v: k for k, v in G_to_H_nodes.items()}
        matches[frozenset(G_nodes)].append((G.subgraph(G_nodes), H_to_G_nodes))

    if one_isomorphism:
        out = [v[0] for v in matches.values()]
    else:
        out = [item for sublist in matches.values() for item in sublist]

    return out


def find_edge_induced_subgraphs(
    G: nx.Graph,
    H: nx.Graph,
    GraphMatcher: isomorphism.GraphMatcher = isomorphism.GraphMatcher,
    one_isomorphism: bool = True,
) -> list[tuple[nx.Graph, dict]]:
    """
    Returns views of all EDGE-induced subgraph of G that are
    isomorphic ("match") the graph H. Matches are defined by the
    `GraphMatcher` class.

    Returns
        subgraphs: Edge-induced subgraphs (views) of G that is isomorphic to H.
    """
    # Can't search directly in edge space first, because extra edges between
    # nodes may prevent a match
    # Search in "edge" (line graph) space first.
    matches = find_node_induced_subgraphs(
        G=nx.line_graph(G),
        H=nx.line_graph(H),
        GraphMatcher=GraphMatcher,
        one_isomorphism=one_isomorphism,
    )

    # Subgraph G based on matched edges
    subgraphs = list(map(lambda x: G.edge_subgraph(x[0]), matches))

    # With extraneous edges removed, now find the node correspondence
    matches = [
        x
        for sg in subgraphs
        for x in find_node_induced_subgraphs(sg, H, GraphMatcher, one_isomorphism)
    ]

    return matches


def graph_to_connected_triples(G: nx.Graph) -> list[list[Any]]:
    """
    Given a graph, return a list of the node triples that are in
    all linear subgraphs with two edges (three nodes).
    """
    # Make the subgraph to find
    H = nx.path_graph(3)

    # Find all unique subgraph isomorphisms
    matches = find_edge_induced_subgraphs(G, H)

    # Extract the nodes
    nodes = [[m[1][0], m[1][1], m[1][2]] for m in matches]

    return nodes