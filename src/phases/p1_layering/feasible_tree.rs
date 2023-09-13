use std::collections::HashSet;

use petgraph::{stable_graph::{StableDiGraph, NodeIndex, EdgeIndex}, visit::{EdgeRef, IntoEdgeReferences}, Direction::{Incoming, Outgoing}};

use super::{Vertex, Edge, slack, cut_values::init_cutvalues, low_lim::init_low_lim};


pub(super) fn feasible_tree(graph: &mut StableDiGraph<Vertex, Edge>, minimum_length: i32) {
    init_rank(graph, minimum_length);

    let mut nodes = graph.node_indices().collect::<Vec<_>>().into_iter();

    while tight_tree(graph, nodes.next().unwrap(), &mut HashSet::new(), minimum_length) < graph.node_count() {
        let edge = find_non_tight_edge(graph, minimum_length);
        let (_, head) = graph.edge_endpoints(edge).unwrap();
        let mut delta = slack(graph, edge, minimum_length);

        if graph[head].is_tree_vertex {
            delta = -delta;
        }

        tighten_edge(graph, delta);
    }

    init_cutvalues(graph);
    init_low_lim(graph);
}

fn tight_tree(graph: &mut StableDiGraph<Vertex, Edge>, vertex: NodeIndex, visited: &mut HashSet<EdgeIndex>, minimum_length: i32) -> usize {
    // start from topmost nodes.
    // then for each topmost node add nodes to tree until done. Then continue with next node until no more nodes are found.
    let mut node_count = 1;
    if !graph[vertex].is_tree_vertex {
        graph[vertex].is_tree_vertex = true;
    }

    let mut neighbors = graph.neighbors_undirected(vertex).detach();
    while let Some(edge) = neighbors.next_edge(&graph) {
        let (tail, head) = graph.edge_endpoints(edge).unwrap();
        let other = if tail == vertex { head } else { tail };

        if !visited.contains(&edge) {
            visited.insert(edge);
            if graph[edge].is_tree_edge {
                node_count += tight_tree(graph, other, visited, minimum_length);
            } else if slack(graph, edge, minimum_length) == 0 && !graph[other].is_tree_vertex {
                graph[edge].is_tree_edge = true;
                node_count += tight_tree(graph, other, visited, minimum_length);
            }
        }
    }

    node_count
}

fn init_rank(graph: &mut StableDiGraph<Vertex, Edge>, minimum_length: i32) {
    // Sort nodes topologically so we don't need to verify that we've assigned
    // a rank to all incoming neighbors
    // assume graphs contain no circles for now
    for v in petgraph::algo::toposort(&*graph, None).unwrap() {
        let rank = graph.neighbors_directed(v, Incoming)
                                .map(|n| graph[n].rank + minimum_length)
                                .max();

        if let Some(rank) = rank {
            graph[v].rank = rank;
        }
    }
}

pub(super) fn is_incident_edge(graph: &StableDiGraph<Vertex, Edge>, edge: &EdgeIndex, ) -> bool {
    let (tail, head)  = graph.edge_endpoints(*edge).unwrap();
    graph[tail].is_tree_vertex ^ graph[head].is_tree_vertex
}

fn find_non_tight_edge(graph: &StableDiGraph<Vertex, Edge>, minimum_length: i32) -> EdgeIndex {
    graph.edge_indices()
        .filter(|e| !graph[*e].is_tree_edge &&  is_incident_edge(&graph, e))
        .min_by(|e1, e2| slack(graph, *e1, minimum_length).cmp(&slack(graph, *e2, minimum_length))).unwrap()
}

fn tighten_edge(graph: &mut StableDiGraph<Vertex, Edge>, delta: i32) {
    for v in graph.node_indices().collect::<Vec<_>>() {
        if graph[v].is_tree_vertex {
            graph[v].rank += delta;
        }
    }
}

#[cfg(test)]
mod tests {

    use std::collections::HashSet;

    use petgraph::Direction::{Incoming, Outgoing};

    use crate::phases::p1_layering::{feasible_tree::{feasible_tree, tight_tree}, tests::EXAMPLE_GRAPH_NON_TIGHT_INITIAL_RANKING, slack};

    use super::{super::tests::{ GraphBuilder, EXAMPLE_GRAPH }, init_rank};

    #[test]
    fn test_initial_ranking_correct_order() {
        let (mut graph, minimum_length, ..) = GraphBuilder::new(&EXAMPLE_GRAPH)
                             .build();

        init_rank(&mut graph, minimum_length);

        for v in graph.node_indices() {
            // all incoming neighbors need to have lower ranks,
            for inc in graph.neighbors_directed(v, Incoming) {
                assert!(graph[v].rank > graph[inc].rank);
            }
            // all outgoing higher
            for out in graph.neighbors_directed(v, Outgoing) {
                assert!(graph[v].rank < graph[out].rank);
            }
        }
    }

    #[test]
    fn test_initial_ranking_at_least_minimum_length_2() {
        let (mut graph, minimum_length, ..) = GraphBuilder::new(&EXAMPLE_GRAPH)
                             .with_minimum_length(2)
                             .build();

        init_rank(&mut graph, minimum_length);

        for v in graph.node_indices() {
            for n in graph.neighbors_undirected(v) {
                assert!(graph[v].rank.abs_diff(graph[n].rank) as i32 >= minimum_length )
            }
        }
    }


    #[test]
    fn test_dfs_start_from_root() {
        let (mut graph, ..)  = GraphBuilder::new(&EXAMPLE_GRAPH).build();
        init_rank(&mut graph, 1);
        let number_of_nodes = graph.node_count();
        tight_tree(&mut graph, 0.into(), &mut HashSet::new(), 1);

        assert_eq!(graph.edge_indices().filter(|e| graph[*e].is_tree_edge).count(), number_of_nodes - 1);
        assert_eq!(graph.node_indices().filter(|v| graph[*v].is_tree_vertex).count(), number_of_nodes);
    }

    #[test]
    fn test_dfs_start_not_from_root() {
        let (mut graph, ..)  = GraphBuilder::new(&EXAMPLE_GRAPH).build();
        let number_of_nodes = graph.node_count();
        init_rank(&mut graph, 1);
        tight_tree(&mut graph, 4.into(), &mut HashSet::new(), 1);

        assert_eq!(graph.edge_indices().filter(|e| graph[*e].is_tree_edge).count(), number_of_nodes - 1);
        assert_eq!(graph.node_indices().filter(|v| graph[*v].is_tree_vertex).count(), number_of_nodes);
    }

    #[test]
    fn test_feasible_tree_is_spanning_tree() {
        // needs to have exactly n - 1 tree edges 
        let (mut graph, minimum_length, ..) = GraphBuilder::new(&EXAMPLE_GRAPH).build();
        
        feasible_tree(&mut graph, minimum_length);

        assert_eq!(graph.edge_weights().filter(|e| e.is_tree_edge).count(), graph.node_indices().count() - 1);
    }

    #[test]
    fn test_make_tight_is_actually_tight() {
        // all tree edges need to have minimum slack
        let (mut graph, minimum_length, ..) = GraphBuilder::new(&EXAMPLE_GRAPH).build();
        
        feasible_tree(&mut graph, minimum_length);
        
        for edge in graph.edge_indices() {
            if graph[edge].is_tree_edge {
                assert_eq!(slack(&graph, edge, minimum_length), 0);
            }
        }
    }

    #[test]
    fn test_make_tight_is_spanning_tree_non_tight_initial_ranking() {
        let (mut graph, minimum_length, ..) = GraphBuilder::new(&EXAMPLE_GRAPH_NON_TIGHT_INITIAL_RANKING).build();

        feasible_tree(&mut graph, minimum_length);

        assert_eq!(graph.edge_weights().filter(|e| e.is_tree_edge).count(), graph.node_indices().count() - 1);
    }

    #[test]
    fn test_make_tight_is_actually_tight_non_tight_inital_ranking() {
        let (mut graph, minimum_length, ..) = GraphBuilder::new(&EXAMPLE_GRAPH_NON_TIGHT_INITIAL_RANKING).build();
        
        feasible_tree(&mut graph, minimum_length);

        for edge in graph.edge_indices() {
            if graph[edge].is_tree_edge {
                assert_eq!(slack(&graph, edge, minimum_length), 0);
            }
        }
    }

    #[test]
    fn test_make_tight_is_actually_tight_non_tight_initial_ranking_2() {
        let (mut graph, minimum_length, ..) = GraphBuilder::new(&[(0, 1), (1, 4), (2, 4), (3, 4)]).build();

        feasible_tree(&mut graph, minimum_length);

        for edge in graph.edge_indices() {
            if graph[edge].is_tree_edge {
                assert_eq!(slack(&graph, edge, minimum_length), 0);
            }
        }
    }
}
