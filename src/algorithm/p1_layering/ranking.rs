use std::collections::{HashSet, VecDeque};

use petgraph::{stable_graph::{StableDiGraph, NodeIndex, EdgeIndex}, Direction::{Incoming, Outgoing, self}};

use super::{Vertex, Edge, slack, cut_values::init_cutvalues, low_lim::init_low_lim};

pub(crate) fn print_ranks(graph: &StableDiGraph<Vertex, Edge>) {
    for (i, v) in graph.node_indices().enumerate() {
        if i != 0 && i % 5 == 0 {
            println!("");
        }
        print!("{}: {},\t ", v.index(), graph[v].rank);
    }
    println!("\n");
}

pub(super) fn feasible_tree(graph: &mut StableDiGraph<Vertex, Edge>, minimum_length: i32) {
    init_rank(graph, minimum_length);
    print_ranks(graph);

    let tree_root = graph.node_indices().next().unwrap();

    while tight_tree(graph, tree_root, &mut HashSet::new(), minimum_length) < graph.node_count() {
        let edge = find_non_tight_edge(graph, minimum_length);
        let (_, head) = graph.edge_endpoints(edge).unwrap();
        let mut delta = slack(graph, edge, minimum_length);

        if graph[head].is_tree_vertex {
            delta = -delta;
        }

        tighten_edge(graph, delta);
        print_ranks(graph);
    }

    init_cutvalues(graph);
    init_low_lim(graph);
}

pub(super) fn update_ranks(graph: &mut StableDiGraph<Vertex, Edge>, minimum_length: i32) {
    let node = graph.node_indices().next().unwrap();
    let mut visited = HashSet::from([node]);
    graph[node].rank = 0;
    let mut queue = VecDeque::from([node]);

    while let Some(parent) = queue.pop_front() {
        update_neighbor_ranks(graph, parent, Outgoing, 1, &mut queue, &mut visited, minimum_length);
        update_neighbor_ranks(graph, parent, Incoming, -1, &mut queue, &mut visited, minimum_length);
    }
}

// TODO: verify this is correct
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

fn is_incident_edge(graph: &StableDiGraph<Vertex, Edge>, edge: &EdgeIndex, ) -> bool {
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


fn update_neighbor_ranks(graph: &mut StableDiGraph<Vertex, Edge>, parent: NodeIndex, direction: Direction, coefficient: i32, queue: &mut VecDeque<NodeIndex>,  visited: &mut HashSet<NodeIndex>, minimum_length: i32) {
    let mut walker = graph.neighbors_directed(parent, direction).detach();
    while let Some((edge, other)) = walker.next(graph) {
        if !graph[edge].is_tree_edge || visited.contains(&other) {
            continue;
        }
        graph[other].rank = graph[parent].rank + minimum_length * coefficient;
        queue.push_back(other);
        visited.insert(other);
    }
}

#[cfg(test)]
mod tests {

    use std::collections::HashSet;

    use petgraph::Direction::{Incoming, Outgoing};

    use crate::algorithm::p1_layering::{ranking::{feasible_tree, tight_tree}, tests::{EXAMPLE_GRAPH_NON_TIGHT_INITIAL_RANKING, EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE}, slack};

    use super::{super::tests::{ GraphBuilder, EXAMPLE_GRAPH }, init_rank, update_ranks};

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

    #[test]
    fn update_ranks_example_graph() {
        let (mut graph, minimum_length, ..) = GraphBuilder::new(&EXAMPLE_GRAPH)
            .with_tree_edges(&EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE)
            .with_ranks(&[
                (0, 0),
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 2),
                (5, 2),
                (6, 3),
                (7, 4),
            ])
            .build();

        let expected = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 1), (5, 1), (6, 2), (7, 4)];
        update_ranks(&mut graph, minimum_length);
            
        for id in graph.node_indices() {
            let rank = graph[id].rank;
            let id = id.index();
            assert_eq!(expected[id], (id, rank));
        }
    }
}
