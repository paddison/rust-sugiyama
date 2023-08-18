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

mod feasible_tree_builder {
    use crate::graphs::p1_layering::{start_layering, FeasibleTreeBuilder, tree::{TreeSubgraph, TighTreeDFS}, rank::tests::create_test_ranking_not_tight};

    use super::create_test_graph;

    fn create_test_builder() -> FeasibleTreeBuilder<isize> {
        let dfs = TighTreeDFS::from_edges(&[(0, 1), (1, 2), (2, 3), (3, 7), (4, 6), (5, 6), (6, 7)]);
        let mut graph = create_test_graph();
        dfs.make_edges_disjoint(&mut graph);

        let ranks = create_test_ranking_not_tight();
        FeasibleTreeBuilder { graph, ranks, tree: dfs.into_tree_subgraph() }
    }

    #[test]
    fn print_cutvalues() {
        start_layering::<usize>(create_test_graph()).initial_ranking(1).make_tight().init_cutvalues();
    }
    
    #[test]
    fn test_init_cutvalues() {
        let cut_values = create_test_builder().init_cutvalues().cut_values;
        assert_eq!(cut_values.get(&(0.into(), 1.into())), Some(&3));
        assert_eq!(cut_values.get(&(1.into(), 2.into())), Some(&3));
        assert_eq!(cut_values.get(&(2.into(), 3.into())), Some(&3));
        assert_eq!(cut_values.get(&(3.into(), 7.into())), Some(&3));
        assert_eq!(cut_values.get(&(4.into(), 6.into())), Some(&0));
        assert_eq!(cut_values.get(&(5.into(), 6.into())), Some(&0));
        assert_eq!(cut_values.get(&(6.into(), 7.into())), Some(&-1));
    }
}