use petgraph::stable_graph::StableDiGraph;

use super::TightTreeBuilder;

pub(crate) fn create_test_graph<T: Default>() -> StableDiGraph<Option<T>, usize> {
    petgraph::stable_graph::StableDiGraph::from_edges(&[(0, 1), (1, 2), (2, 3), (0, 4), (0, 5), (4, 6), (5, 6), (3, 7), (6, 7)])
}

pub(crate) fn create_tight_tree_builder_non_tight_ranking<T: Default>() -> TightTreeBuilder<T> {
    let mut graph = create_test_graph();
    graph.add_node(None);
    graph.add_edge(7.into(), 8.into(), usize::default());
    let ranks = super::rank::tests::create_test_ranking_not_tight(); 
    TightTreeBuilder::new(graph, ranks)
}

mod tight_tree_builder {
    use std::collections::HashSet;

    use crate::graphs::p1_layering::{tree::Tree, start_layering, tests::{create_test_graph, create_tight_tree_builder_non_tight_ranking}};

    #[test]
    fn test_dfs_start_from_root() {
        let mut tree = Tree::new();
        let tight_tree_builder = start_layering(create_test_graph::<i32>()).initial_ranking(1);
        let number_of_nodes = tight_tree_builder.graph.node_count();
        tight_tree_builder.tight_tree_dfs(&mut tree, 0.into(), &mut HashSet::new());
        assert_eq!(tree.edge_count(), number_of_nodes - 1);
        assert_eq!(tree.vertice_count(), number_of_nodes);
    }

    #[test]
    fn test_dfs_start_not_from_root() {
        let mut tree = Tree::new();
        let tight_tree_builder = start_layering(create_test_graph::<i32>()).initial_ranking(1);
        let number_of_nodes = tight_tree_builder.graph.node_count();
        tight_tree_builder.tight_tree_dfs(&mut tree, 4.into(), &mut HashSet::new());
        assert_eq!(tree.edge_count(), number_of_nodes - 1);
        assert_eq!(tree.vertice_count(), number_of_nodes);
    }

    #[test]
    fn test_dfs_non_tight_ranking() {
        let mut tree = Tree::new();
        let mut tight_tree_builder = create_tight_tree_builder_non_tight_ranking::<i32>();
        let number_of_nodes = tight_tree_builder.graph.node_count();
        tight_tree_builder.tight_tree_dfs(&mut tree, 0.into(), &mut HashSet::new());
        assert_eq!(tree.edge_count(), number_of_nodes - 2);
        assert_eq!(tree.vertice_count(), number_of_nodes - 1);

        let (tail, head) = tight_tree_builder.find_non_tight_edge(&tree);
        assert_eq!(head.index(), 8);
        assert_eq!(tail.index(), 7);
        let delta = tight_tree_builder.ranks.slack(tail, head);

        tight_tree_builder.ranks.tighten_edge(&tree, delta);
        
        tight_tree_builder.tight_tree_dfs(&mut tree, 0.into(), &mut HashSet::new());
        assert_eq!(tree.edge_count(), number_of_nodes - 1);
        assert_eq!(tree.vertice_count(), number_of_nodes);
    }
}