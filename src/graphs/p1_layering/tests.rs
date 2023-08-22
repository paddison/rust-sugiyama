use petgraph::stable_graph::StableDiGraph;

use super::{TightTreeBuilder, FeasibleTreeBuilder, rank::tests::create_test_ranking_not_tight, tree::TighTreeDFS};

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

static EXAMPLE_GRAPH_1: [(usize, usize); 7] = [(0, 1), (1, 2), (2, 3), (3, 7), (4, 6), (5, 6), (6, 7)];
static LOW_LIM_GRAPH: [(u32, u32); 8] = [(0, 1), (1, 2), (1, 3), (0, 4), (4, 5), (5, 6), (4, 7), (4, 8)];
static FEASIBLE_TREE_NEG_CUT_VALUE: [(u32, u32); 7] = [(0, 1), (1, 2), (2, 3), (3, 7), (4, 6), (5, 6), (6, 7)];
static FEASIBLE_TREE_POS_CUT_VALUE: [(u32, u32); 7] = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 6), (5, 6)];


fn create_test_builder(edges: &[(usize, usize)]) -> FeasibleTreeBuilder<isize> {
    let dfs = TighTreeDFS::from_edges(edges);
    let mut graph = create_test_graph();
    dfs.make_edges_disjoint(&mut graph);

    let ranks = create_test_ranking_not_tight();
    FeasibleTreeBuilder { graph, ranks, tree: dfs.into_tree_subgraph() }
}
mod feasible_tree_builder {
    use crate::graphs::p1_layering::{start_layering, FeasibleTreeBuilder, tree::{TighTreeDFS}, rank::tests::create_test_ranking_not_tight, tests::{create_test_builder, EXAMPLE_GRAPH_1}};

    use super::create_test_graph;


    #[test]
    fn print_cutvalues() {
        start_layering::<usize>(create_test_graph()).initial_ranking(1).make_tight().init_cutvalues();
    }
    
    #[test]
    fn test_init_cutvalues() {
        let cut_values = create_test_builder(&EXAMPLE_GRAPH_1).init_cutvalues().cut_values;
        assert_eq!(cut_values.get(&(0.into(), 1.into())), Some(&3));
        assert_eq!(cut_values.get(&(1.into(), 2.into())), Some(&3));
        assert_eq!(cut_values.get(&(2.into(), 3.into())), Some(&3));
        assert_eq!(cut_values.get(&(3.into(), 7.into())), Some(&3));
        assert_eq!(cut_values.get(&(4.into(), 6.into())), Some(&0));
        assert_eq!(cut_values.get(&(5.into(), 6.into())), Some(&0));
        assert_eq!(cut_values.get(&(6.into(), 7.into())), Some(&-1));
    }
}

mod feasible_tree {
    use std::collections::HashMap;

    use petgraph::stable_graph::{NodeIndex, StableDiGraph};

    use crate::graphs::p1_layering::{FeasibleTree, TreeData, tests::{LOW_LIM_GRAPH, FEASIBLE_TREE_POS_CUT_VALUE}, rank::{tests::create_test_ranking_not_tight, Ranks}};

    use super::{create_test_builder, EXAMPLE_GRAPH_1, FEASIBLE_TREE_NEG_CUT_VALUE};

    fn create_feasible_tree() -> FeasibleTree<isize>{
        create_test_builder(&EXAMPLE_GRAPH_1).init_cutvalues()
    }

    fn get_low_lim() -> HashMap<NodeIndex, TreeData> {
        HashMap::<NodeIndex, TreeData>::from([
            (0.into(), TreeData::new(9, 1, None)),
            (1.into(), TreeData::new(3, 1, Some(0.into()))),
            (2.into(), TreeData::new(1, 1, Some(1.into()))),
            (3.into(), TreeData::new(2, 2, Some(1.into()))),
            (4.into(), TreeData::new(8, 4, Some(0.into()))),
            (5.into(), TreeData::new(5, 4, Some(4.into()))),
            (6.into(), TreeData::new(4, 4, Some(5.into()))),
            (7.into(), TreeData::new(6, 6, Some(4.into()))),
            (8.into(), TreeData::new(7, 7, Some(4.into()))),
        ])
    }

    #[test]
    fn test_dfs_low_lim() {
        let ft = create_feasible_tree();
        let mut low_lim = HashMap::new();
        let root = ft.graph.node_indices().next().unwrap();
        ft.dfs_low_lim(&mut low_lim, root, 1, None);
        println!("{low_lim:?}");
    }

    #[test]
    fn test_dfs_update_low_lim() {
        let tree = StableDiGraph::from_edges(&LOW_LIM_GRAPH);
        let graph = StableDiGraph::from_edges(&[(6, 8)]);
        let ranks = create_test_ranking_not_tight();
        let ft: FeasibleTree<isize> = FeasibleTree{ graph, tree, ranks, cut_values: HashMap::new() };
        let least_common_ancestor = 4;
        let mut low_lim = HashMap::from([
            (4.into(), TreeData::new(8, 4, Some(1.into()))),
            // initialize with wrong values
            (5.into(), TreeData::new(0, 0, Some(0.into()))),
            (6.into(), TreeData::new(0, 0, Some(0.into()))),
            (7.into(), TreeData::new(0, 0, Some(0.into()))),
            (8.into(), TreeData::new(0, 0, Some(0.into()))),
        
        ]);
        ft.update_low_lim(&mut low_lim, least_common_ancestor.into());
        // these tests may fail since i don't know the exact order that petgraph returns neighbors
        assert_eq!(low_lim.get(&4.into()), Some(&TreeData::new(8, 4, Some(1.into()))));
        assert_eq!(low_lim.get(&5.into()), Some(&TreeData::new(5, 4, Some(4.into()))));
        assert_eq!(low_lim.get(&6.into()), Some(&TreeData::new(4, 4, Some(5.into()))));
        assert_eq!(low_lim.get(&7.into()), Some(&TreeData::new(6, 6, Some(4.into()))));
        assert_eq!(low_lim.get(&8.into()), Some(&TreeData::new(7, 7, Some(4.into()))));
    }

    #[test]
    fn test_is_head_to_tail_true_root_in_tail() {
        // u is always considered to be the tail of the edge to be swapped
        let low_lim = get_low_lim();
        let u = 5;
        let tail = 6;
        let head = 7;
        assert!(FeasibleTree::<isize>::is_head_to_tail(&low_lim, tail.into(), head.into(), *low_lim.get(&u.into()).unwrap(), false));
    }

    #[test]
    fn test_is_head_to_tail_true_root_in_head() {
        let low_lim = get_low_lim();
        let u = 4;
        let tail = 3;
        let head = 5;
        assert!(FeasibleTree::<isize>::is_head_to_tail(&low_lim, tail.into(), head.into(), *low_lim.get(&u.into()).unwrap(), true));
    }

    #[test]
    fn test_is_head_to_tail_false_root_in_tail() {
        let low_lim = get_low_lim();
        let u = 3;
        let tail = 4;
        let head = 3;
        assert!(!FeasibleTree::<isize>::is_head_to_tail(&low_lim, tail.into(), head.into(), *low_lim.get(&u.into()).unwrap(), false));
    }

    #[test]
    fn test_is_head_to_tail_false_root_in_head() {
        let low_lim = get_low_lim();
        let u = 2;
        let tail = 2;
        let head = 0;
        assert!(!FeasibleTree::<isize>::is_head_to_tail(&low_lim, tail.into(), head.into(), *low_lim.get(&u.into()).unwrap(), true));
    }

    #[test]
    fn test_update_ranks_neg_cutvalue_tree() {
        let tree = StableDiGraph::from_edges(FEASIBLE_TREE_NEG_CUT_VALUE);
        let graph = StableDiGraph::new();
        let ranks = Ranks::new_unchecked(HashMap::new(), 1);
        let mut ft: FeasibleTree<isize> = FeasibleTree{ graph, tree, ranks, cut_values: HashMap::new() };
        ft.update_ranks();
        ft.ranks.normalize();
        assert_eq!(ft.ranks[0.into()], 0);
        assert_eq!(ft.ranks[1.into()], 1);
        assert_eq!(ft.ranks[2.into()], 2);
        assert_eq!(ft.ranks[3.into()], 3);
        assert_eq!(ft.ranks[4.into()], 2);
        assert_eq!(ft.ranks[5.into()], 2);
        assert_eq!(ft.ranks[6.into()], 3);
        assert_eq!(ft.ranks[7.into()], 4);
    }

    #[test]
    fn test_update_ranks_pos_cutvalue_tree() {
        let tree = StableDiGraph::from_edges(FEASIBLE_TREE_POS_CUT_VALUE);
        let graph = StableDiGraph::new();
        let ranks = Ranks::new_unchecked(HashMap::new(), 1);
        let mut ft: FeasibleTree<isize> = FeasibleTree{ graph, tree, ranks, cut_values: HashMap::new() };
        ft.update_ranks();
        ft.ranks.normalize();
        assert_eq!(ft.ranks[0.into()], 0);
        assert_eq!(ft.ranks[1.into()], 1);
        assert_eq!(ft.ranks[2.into()], 2);
        assert_eq!(ft.ranks[3.into()], 3);
        assert_eq!(ft.ranks[4.into()], 1);
        assert_eq!(ft.ranks[5.into()], 1);
        assert_eq!(ft.ranks[6.into()], 2);
        assert_eq!(ft.ranks[7.into()], 4);
    }
    
}