use std::collections::HashSet;

use petgraph::stable_graph::{StableDiGraph, NodeIndex};

use super::{Vertex, Edge};

pub(super) fn init_low_lim(graph: &mut StableDiGraph<Vertex, Edge>) {
    // start at arbitrary root node
    let root = graph.node_indices().next().unwrap();
    let mut max_lim = graph.node_count() as u32;
    dfs_low_lim(graph, root, None, &mut max_lim, &mut HashSet::new());
}

pub(super) fn update_low_lim(graph: &mut StableDiGraph<Vertex, Edge>, least_common_ancestor: NodeIndex) {
    let parent = graph[least_common_ancestor].parent;
    let mut visited = match &parent {
        Some(parent) => HashSet::from([*parent]),
        None => HashSet::new()
    };
    let mut max_lim = graph[least_common_ancestor].lim;
    dfs_low_lim(graph, least_common_ancestor, parent, &mut max_lim, &mut visited);
}

fn dfs_low_lim(graph: &mut StableDiGraph<Vertex, Edge>, next: NodeIndex, parent: Option<NodeIndex>, max_lim: &mut u32, visited: &mut HashSet<NodeIndex>) {
    visited.insert(next);
    graph[next].lim = *max_lim;
    graph[next].parent = parent;
    let mut walker = graph.neighbors_undirected(next).detach();
    while let Some((edge, n)) = walker.next(graph) {
        if !visited.contains(&n) && graph[edge].is_tree_edge {
            *max_lim -= 1;
            dfs_low_lim(graph, n, Some(next), max_lim, visited);
        }
    }
    graph[next].low = *max_lim;
}

#[cfg(test)]
mod tests {
    use petgraph::stable_graph::NodeIndex;

    use crate::algorithm::p1_layering::{Vertex, low_lim::{init_low_lim, update_low_lim}, tests::{LOW_LIM_GRAPH_AFTER_UPDATE, LOW_LIM_GRAPH_LOW_LIM_VALUES, EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE, EXAMPLE_GRAPH_LOW_LIM_VALUES_NEG_CUT_VALUE}};

    use super::super::tests::{LOW_LIM_GRAPH, GraphBuilder, EXAMPLE_GRAPH, EXAMPLE_GRAPH_FEASIBLE_TREE_NEG_CUT_VALUE};

    #[test]
    fn init_low_lim_low_lim_graph() {
        let (mut graph, ..) = GraphBuilder::new(&LOW_LIM_GRAPH)
                    .with_tree_edges(&LOW_LIM_GRAPH)
                    .build();
        
        init_low_lim(&mut graph);

        let v0 = graph[NodeIndex::from(0)];
        let v1 = graph[NodeIndex::from(1)];
        let v2 = graph[NodeIndex::from(2)];
        let v3 = graph[NodeIndex::from(3)];
        let v4 = graph[NodeIndex::from(4)];
        let v5 = graph[NodeIndex::from(5)];
        let v6 = graph[NodeIndex::from(6)];
        let v7 = graph[NodeIndex::from(7)];
        let v8 = graph[NodeIndex::from(8)];

        assert_eq!(v0, Vertex::new_test_p1(1, 9, None, true));
        assert_eq!(v1, Vertex::new_test_p1(1, 3, Some(0.into()), true));
        assert_eq!(v2, Vertex::new_test_p1(1, 1, Some(1.into()), true));
        assert_eq!(v3, Vertex::new_test_p1(2, 2, Some(1.into()), true));
        assert_eq!(v4, Vertex::new_test_p1(4, 8, Some(0.into()), true));
        assert_eq!(v5, Vertex::new_test_p1(4, 5, Some(4.into()), true));
        assert_eq!(v6, Vertex::new_test_p1(4, 4, Some(5.into()), true));
        assert_eq!(v7, Vertex::new_test_p1(6, 6, Some(4.into()), true));
        assert_eq!(v8, Vertex::new_test_p1(7, 7, Some(4.into()), true));
    }

    #[test]
    fn test_init_low_lim_neg_cut_value() {
        let (mut graph, ..) = GraphBuilder::new(&EXAMPLE_GRAPH)
            .with_tree_edges(&EXAMPLE_GRAPH_FEASIBLE_TREE_NEG_CUT_VALUE)
            .build();

        init_low_lim(&mut graph);

        assert_eq!(graph[NodeIndex::from(0)].low, 1);
        assert_eq!(graph[NodeIndex::from(0)].lim, 8);
        assert_eq!(graph[NodeIndex::from(0)].parent, None);
        assert_eq!(graph[NodeIndex::from(1)].low, 1);
        assert_eq!(graph[NodeIndex::from(1)].lim, 7);
        assert_eq!(graph[NodeIndex::from(1)].parent, Some(0.into()));
        assert_eq!(graph[NodeIndex::from(2)].low, 1);
        assert_eq!(graph[NodeIndex::from(2)].lim, 6);
        assert_eq!(graph[NodeIndex::from(2)].parent, Some(1.into()));
        assert_eq!(graph[NodeIndex::from(3)].low, 1);
        assert_eq!(graph[NodeIndex::from(3)].lim, 5);
        assert_eq!(graph[NodeIndex::from(3)].parent, Some(2.into()));
        assert!(graph[NodeIndex::from(4)].low == 1 || graph[NodeIndex::from(4)].low == 2);
        assert!(graph[NodeIndex::from(4)].lim == 1 || graph[NodeIndex::from(4)].lim == 2);
        assert_eq!(graph[NodeIndex::from(4)].parent, Some(6.into()));
        assert!(graph[NodeIndex::from(5)].low == 1 || graph[NodeIndex::from(5)].low == 2);
        assert!(graph[NodeIndex::from(5)].lim == 1 || graph[NodeIndex::from(5)].lim == 2);
        assert_eq!(graph[NodeIndex::from(5)].parent, Some(6.into()));
        assert_eq!(graph[NodeIndex::from(6)].low, 1);
        assert_eq!(graph[NodeIndex::from(6)].lim, 3);
        assert_eq!(graph[NodeIndex::from(6)].parent, Some(7.into()));
        assert_eq!(graph[NodeIndex::from(7)].low, 1);
        assert_eq!(graph[NodeIndex::from(7)].lim, 4);
        assert_eq!(graph[NodeIndex::from(7)].parent, Some(3.into()));
    }
    
    #[test]
    fn update_low_lim_low_lim_graph() {
        let (mut graph, .., least_common_ancestor) = GraphBuilder::new(&LOW_LIM_GRAPH_AFTER_UPDATE)
            .with_tree_edges(&LOW_LIM_GRAPH_AFTER_UPDATE)
            .with_low_lim_values(&LOW_LIM_GRAPH_LOW_LIM_VALUES)
            .with_least_common_ancestor(4)
            .build();

        update_low_lim(&mut graph, least_common_ancestor); 
        let v4 = graph[NodeIndex::from(4)];
        let v5 = graph[NodeIndex::from(5)];
        let v6 = graph[NodeIndex::from(6)];
        let v7 = graph[NodeIndex::from(7)];
        let v8 = graph[NodeIndex::from(8)];
        assert_eq!(v4, Vertex::new_test_p1(4, 8, Some(0.into()), true));
        assert_eq!(v5, Vertex::new_test_p1(4, 4, Some(6.into()), true));
        assert_eq!(v6, Vertex::new_test_p1(4, 5, Some(7.into()), true));
        assert_eq!(v7, Vertex::new_test_p1(4, 6, Some(4.into()), true));
        assert_eq!(v8, Vertex::new_test_p1(7, 7, Some(4.into()), true));
    }

    #[test]
    fn update_low_lim_example_graph() {
        let (mut graph, .., least_common_ancestor) = GraphBuilder::new(&EXAMPLE_GRAPH)
            .with_tree_edges(&EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE)
            .with_low_lim_values(&EXAMPLE_GRAPH_LOW_LIM_VALUES_NEG_CUT_VALUE)
            .with_least_common_ancestor(0)
            .build();

        update_low_lim(&mut graph, least_common_ancestor);
        let v0 = graph[NodeIndex::from(0)];
        let v1 = graph[NodeIndex::from(1)];
        let v2 = graph[NodeIndex::from(2)];
        let v3 = graph[NodeIndex::from(3)];
        let v4 = graph[NodeIndex::from(4)];
        let v5 = graph[NodeIndex::from(5)];
        let v6 = graph[NodeIndex::from(6)];
        let v7 = graph[NodeIndex::from(7)];
        assert_eq!(v0, Vertex::new_test_p1(1, 8, None, true));
        assert_eq!(v1, Vertex::new_test_p1(1, 4, Some(0.into()), true));
        assert_eq!(v2, Vertex::new_test_p1(1, 3, Some(1.into()), true));
        assert_eq!(v3, Vertex::new_test_p1(1, 2, Some(2.into()), true));
        assert_eq!(v4, Vertex::new_test_p1(5, 7, Some(0.into()), true));
        assert_eq!(v5, Vertex::new_test_p1(5, 5, Some(6.into()), true));
        assert_eq!(v6, Vertex::new_test_p1(5, 6, Some(4.into()), true));
        assert_eq!(v7, Vertex::new_test_p1(1, 1, Some(3.into()), true));
    }
}