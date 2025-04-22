__all__ = [
    "find_node_induced_subgraphs",
    "find_edge_induced_subgraphs",
    "graph_to_connected_triples",
]

from collections import defaultdict
from typing import Any
import networkx as nx
from networkx.algorithms import isomorphism


def find_node_induced_subgraphs(
    source_graph: nx.Graph,
    target_graph: nx.Graph,
    graph_matcher_class: isomorphism.GraphMatcher = isomorphism.GraphMatcher,
    one_isomorphism: bool = True,
) -> list[tuple[nx.Graph, dict]]:
    """
    Returns views of all NODE-induced subgraph of source_graph that are
    isomorphic ("match") the graph target_graph. Matches are defined by the
    `GraphMatcher` class.

    Parameters
    ----------
    source_graph : nx.Graph
        The graph to search for subgraphs in.
    target_graph : nx.Graph
        The graph pattern to match against.
    graph_matcher_class : isomorphism.GraphMatcher, optional
        The graph matcher class to use for isomorphism checking.
    one_isomorphism : bool, optional
        If True, return only one isomorphism per unique subgraph.

    Returns
    -------
    list[tuple[nx.Graph, dict]]
        List of tuples containing (subgraph, node mapping) where subgraph is a node-induced
        subgraph of source_graph that is isomorphic to target_graph.
    """
    # Instantiate the graph matcher object
    gm = graph_matcher_class(source_graph, target_graph)

    # Collect all node-induced isomorphic subgraphs
    matches = defaultdict(list)
    for source_to_target_nodes in gm.subgraph_isomorphisms_iter():
        source_nodes = source_to_target_nodes.keys()
        target_to_source_nodes = {v: k for k, v in source_to_target_nodes.items()}
        matches[frozenset(source_nodes)].append(
            (source_graph.subgraph(source_nodes), target_to_source_nodes)
        )

    if one_isomorphism:
        out = [v[0] for v in matches.values()]
    else:
        out = [item for sublist in matches.values() for item in sublist]

    return out


def find_edge_induced_subgraphs(
    source_graph: nx.Graph,
    target_graph: nx.Graph,
    graph_matcher_class: isomorphism.GraphMatcher = isomorphism.GraphMatcher,
    one_isomorphism: bool = True,
) -> list[tuple[nx.Graph, dict]]:
    """
    Returns views of all EDGE-induced subgraph of source_graph that are
    isomorphic ("match") the graph target_graph. Matches are defined by the
    `GraphMatcher` class.

    Parameters
    ----------
    source_graph : nx.Graph
        The graph to search for subgraphs in.
    target_graph : nx.Graph
        The graph pattern to match against.
    graph_matcher_class : isomorphism.GraphMatcher, optional
        The graph matcher class to use for isomorphism checking.
    one_isomorphism : bool, optional
        If True, return only one isomorphism per unique subgraph.

    Returns
    -------
    list[tuple[nx.Graph, dict]]
        List of tuples containing (subgraph, node mapping) where subgraph is an edge-induced
        subgraph of source_graph that is isomorphic to target_graph.
    """
    # Can't search directly in edge space first, because extra edges between
    # nodes may prevent a match
    # Search in "edge" (line graph) space first.
    matches = find_node_induced_subgraphs(
        source_graph=nx.line_graph(source_graph),
        target_graph=nx.line_graph(target_graph),
        graph_matcher_class=graph_matcher_class,
        one_isomorphism=one_isomorphism,
    )

    # Subgraph source_graph based on matched edges
    subgraphs = list(map(lambda x: source_graph.edge_subgraph(x[0]), matches))

    # With extraneous edges removed, now find the node correspondence
    matches = [
        x
        for subgraph in subgraphs
        for x in find_node_induced_subgraphs(
            subgraph, target_graph, graph_matcher_class, one_isomorphism
        )
    ]

    return matches


def graph_to_connected_triples(graph: nx.Graph) -> list[list[Any, Any, Any]]:
    """
    Given a graph, return a list of the node triples that are in
    all linear subgraphs with two edges (three nodes).

    Parameters
    ----------
    graph : nx.Graph
        The input graph to find triples in.

    Returns
    -------
    list[list[Any]]
        List of node triples that form linear subgraphs with two edges.
    """
    # Make the subgraph to find
    triple_pattern = nx.path_graph(3)

    # Find all unique subgraph isomorphisms
    matches = find_edge_induced_subgraphs(graph, triple_pattern)

    # Extract the nodes
    triples = [[m[1][0], m[1][1], m[1][2]] for m in matches]

    return triples
