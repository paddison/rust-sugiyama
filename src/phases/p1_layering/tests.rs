use petgraph::stable_graph::{StableDiGraph, NodeIndex, EdgeIndex};

use super::{Vertex, Edge};

pub(crate) static EXAMPLE_GRAPH: [(u32, u32); 9] = [(0, 1), (1, 2), (2, 3), (3, 7), (4, 6), (5, 6), (6, 7), (0, 4), (0, 5)];
pub(crate) static EXAMPLE_GRAPH_NON_TIGHT_INITIAL_RANKING: [(u32, u32); 7] = [(0, 1), (1, 2), (2, 3), (3, 7), (4, 6), (5, 6), (6, 7)];
pub(crate) static EXAMPLE_GRAPH_FEASIBLE_TREE_NEG_CUT_VALUE: [(u32, u32); 7] = [(0, 1), (1, 2), (2, 3), (3, 7), (4, 6), (5, 6), (6, 7)];
pub(crate) static EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE: [(u32, u32); 7] = [(0, 1), (0, 4), (1, 2), (2, 3), (3, 7), (4, 6), (5, 6)];
pub(crate) static LOW_LIM_GRAPH: [(u32, u32); 8] = [(0, 1), (1, 2), (1, 3), (0, 4), (4, 5), (5, 6), (4, 7), (4, 8)];
static LOW_LIM_GRAPH_AFTER_UPDATE: [(u32, u32); 8] = [(0, 1), (1, 2), (1, 3), (0, 4), (5, 6), (4, 7), (4, 8), (6, 7)];
static LOW_LIM_GRAPH_LOW_LIM_VALUES: [(u32, u32, u32, Option<u32>); 9] = [
    (0, 1, 9, None),
    (1, 1, 3, Some(0)),
    (2, 1, 1, Some(1)),
    (3, 2, 2, Some(1)),
    (4, 4, 8, Some(0)),
    (5, 4, 5, Some(4)),
    (6, 4, 4, Some(5)),
    (7, 6, 6, Some(4)),
    (8, 7, 7, Some(4))
];
static EXAMPLE_GRAPH_LOW_LIM_VALUES_NEG_CUT_VALUE: [(u32, u32, u32, Option<u32>); 8] = [
    (0, 1, 8, None),
    (1, 1, 7, Some(0)),
    (2, 1, 6, Some(1)),
    (3, 1, 5, Some(2)),
    (4, 1, 1, Some(6)),
    (5, 2, 2, Some(6)),
    (6, 1, 3, Some(7)),
    (7, 1, 4, Some(3)),
];

static CUT_VALUES_EXAMPLE_GRAPH_NEG_CUT_VALUE: [(u32, u32, i32); 7] = [
    (0, 1, 3),
    (1, 2, 3),
    (2, 3, 3),
    (3, 7, 3),
    (4, 6, 0),
    (5, 6, 0),
    (6, 7, -1),
];

pub(super) struct GraphBuilder {
    graph: StableDiGraph<Vertex, Edge>,
    minimum_length: i32,
    connecting_path: Vec<EdgeIndex>,
    removed_edge: EdgeIndex,
    least_common_ancestor: NodeIndex,
}

impl GraphBuilder {
    pub(super) fn new(edges: &[(u32, u32)]) -> Self {
        let graph = StableDiGraph::<Vertex, Edge>::from_edges(edges);
        Self { 
            graph, 
            minimum_length: 1,
            connecting_path: Vec::new(),
            removed_edge: EdgeIndex::from(0),
            least_common_ancestor: NodeIndex::from(0),
        }
    }

    pub(super) fn minimum_length(&mut self) -> &mut i32 {
        &mut self.minimum_length
    }

    pub(super) fn build(self) -> (StableDiGraph<Vertex, Edge>, i32, Vec<EdgeIndex>, EdgeIndex, NodeIndex) {
        let Self { 
            graph, 
            minimum_length, 
            connecting_path, 
            removed_edge, 
            least_common_ancestor 
        } = self;
        (graph, minimum_length, connecting_path, removed_edge, least_common_ancestor)
    }

    // Blanket implementations
    pub(super) fn with_minimum_length(mut self, minimum_length: u32) -> Self 
    where Self: Sized 
    {
        self.minimum_length = minimum_length as i32;
        self
    }

    pub(super) fn with_cut_values(mut self, cut_values: &[(u32, u32, i32)]) -> Self 
    where Self: Sized
    {
        for (tail, head, cut_value) in cut_values {
            // ignore any edges that do not exist
            if let Some(edge) = self.graph.find_edge((*tail).into(), (*head).into()) {
                self.graph[edge].cut_value = Some(*cut_value);
            }
        }

        self
    }

    pub(super) fn with_tree_edges(mut self, tree_edges: &[(u32, u32)]) -> Self 
    where Self: Sized
    {
        for (tail, head) in tree_edges {
            self.graph[NodeIndex::from(*tail)].is_tree_vertex = true;
            self.graph[NodeIndex::from(*head)].is_tree_vertex = true;
            let edge = self.graph.find_edge((*tail).into(), (*head).into()).expect("Edge not found in tree.");
            self.graph[edge].is_tree_edge = true;
        }
        self
    }

    /// Add low_lim_values to the graph.
    /// 
    /// Input is given by a slice of (NodeIndex, low, lim, parent).
    /// 
    /// Panics if vertex is not contained in tree.
    pub(super) fn with_low_lim_values(mut self, low_lim: &[(u32, u32, u32, Option<u32>)]) -> Self 
    where Self: Sized
    {
        for (id, low, lim, parent) in low_lim {
            let mut weight = self.graph.node_weight_mut((*id).into()).unwrap();
            weight.low = *low;
            weight.lim = *lim;
            weight.parent = parent.map(|p| p.into());
        }

        self
    }

    pub(super) fn with_connecting_path(mut self, connecting_path: &[(u32, u32)]) -> Self {
        self.connecting_path = connecting_path.into_iter()
                                              .map(|(tail, head)| self.graph.find_edge_undirected((*tail).into(), (*head).into()).unwrap().0)
                                              .collect::<Vec<_>>();
        self
    }

    /// Set the edge that was removed during rank procedure.
    /// Panics if edge is not contained in graph.
    pub(super) fn with_removed_edge(mut self, tail: u32, head: u32) -> Self {
        self.removed_edge = self.graph.find_edge(tail.into(), head.into()).unwrap();
        self.graph[self.removed_edge].is_tree_edge = false;
        self
    }

    /// Set the ranks, of the vertices.
    /// Used to check if outtdated ranks are updated
    /// correctly after update_ranks is run.
    /// 
    /// Input is  [(NodeIndex, rank)].
    /// 
    /// Panics if node is not part of graph.
    pub(super) fn with_ranks(mut self, ranks: &[(u32, i32)]) -> Self {
        for (n, r) in ranks {
            self.graph[NodeIndex::from(*n)].rank = *r;
        }
        self
    }

    /// Set the least common ancestor, which is the root
    /// of the path connecting the edge that will be added 
    /// to the graph.
    /// 
    /// Panics if vertex is not part of graph.
    pub(super) fn with_least_common_ancestor(mut self, lca: u32) -> Self {
        assert!(self.graph.contains_node(NodeIndex::from(lca)));
        self.least_common_ancestor = lca.into();
        self
    }
}

mod feasible_tree {
    use petgraph::stable_graph::NodeIndex;

    use crate::phases::p1_layering::{tests::{LOW_LIM_GRAPH, LOW_LIM_GRAPH_LOW_LIM_VALUES}, Edge, cut_values::init_cutvalues, low_lim::init_low_lim, leave_edge, is_head_to_tail, enter_edge, remove_outdated_cut_values};

    use super::{EXAMPLE_GRAPH, GraphBuilder, EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE, EXAMPLE_GRAPH_FEASIBLE_TREE_NEG_CUT_VALUE};

    #[test]
    fn leave_edge_has_negative_cut_value() {
        let (mut graph, ..) = GraphBuilder::new(&EXAMPLE_GRAPH)
                                                 .with_tree_edges(&EXAMPLE_GRAPH_FEASIBLE_TREE_NEG_CUT_VALUE)
                                                 .build();
        
        init_cutvalues(&mut graph);
        init_low_lim(&mut graph);
        
        let leave_edge = leave_edge(&graph);
        assert!(leave_edge.is_some());
        let (tail, head) = graph.edge_endpoints(leave_edge.unwrap()).unwrap();
        assert_eq!(tail, NodeIndex::from(6));
        assert_eq!(head, NodeIndex::from(7));
    }

    #[test]
    fn leave_edge_has_no_negative_cut_value() {
        let (mut graph, ..) = GraphBuilder::new(&EXAMPLE_GRAPH)
                                                 .with_tree_edges(&EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE)
                                                 .build();
        
        init_cutvalues(&mut graph);
        init_low_lim(&mut graph);
        
        let leave_edge = leave_edge(&graph);
        assert!(leave_edge.is_none());
    }

    #[test]
    fn test_is_head_to_tail_true_root_in_tail() {
        // u is always considered to be the tail of the edge to be swapped
        let (mut graph, ..) = GraphBuilder::new(&LOW_LIM_GRAPH).with_tree_edges(&LOW_LIM_GRAPH).with_low_lim_values(&LOW_LIM_GRAPH_LOW_LIM_VALUES).build();
        let u = 5;
        let u = *graph.node_weight(u.into()).unwrap();
        let tail = 6;
        let head = 7;
        let edge = graph.add_edge(tail.into(), head.into(), Edge::default());
        assert!(is_head_to_tail(&graph, edge, u, false));
    }

    #[test]
    fn test_is_head_to_tail_true_root_in_head() {
        let (mut graph, ..) = GraphBuilder::new(&LOW_LIM_GRAPH).with_tree_edges(&LOW_LIM_GRAPH).with_low_lim_values(&LOW_LIM_GRAPH_LOW_LIM_VALUES).build();
        let u = 4;
        let u = *graph.node_weight(u.into()).unwrap();
        let tail = 3;
        let head = 5;
        let edge = graph.add_edge(tail.into(), head.into(), Edge::default());
        assert!(is_head_to_tail(&graph, edge, u, true));
    }

    #[test]
    fn test_is_head_to_tail_false_root_in_tail() {
        let (mut graph, ..) = GraphBuilder::new(&LOW_LIM_GRAPH).with_tree_edges(&LOW_LIM_GRAPH).with_low_lim_values(&LOW_LIM_GRAPH_LOW_LIM_VALUES).build();
        let u = 3;
        let u = *graph.node_weight(u.into()).unwrap();
        let tail = 4;
        let head = 3;
        let edge = graph.add_edge(tail.into(), head.into(), Edge::default());
        assert!(!is_head_to_tail(&graph, edge, u, false));
    }

    #[test]
    fn test_is_head_to_tail_false_root_in_head() {
        let (mut graph, ..) = GraphBuilder::new(&LOW_LIM_GRAPH).with_tree_edges(&LOW_LIM_GRAPH).with_low_lim_values(&LOW_LIM_GRAPH_LOW_LIM_VALUES).build();
        let u = 2;
        let u = *graph.node_weight(u.into()).unwrap();
        let tail = 2;
        let head = 0;
        let edge = graph.add_edge(tail.into(), head.into(), Edge::default());
        assert!(!is_head_to_tail(&graph, edge, u, true));
    }

    #[test]
    fn enter_edge_find_edge() {
        let (mut graph, ..) = GraphBuilder::new(&EXAMPLE_GRAPH)
            .with_tree_edges(&EXAMPLE_GRAPH_FEASIBLE_TREE_NEG_CUT_VALUE)
            .build();

        init_cutvalues(&mut graph);
        init_low_lim(&mut graph);
        
        let leave_edge = leave_edge(&graph).unwrap();
        let enter_edge = enter_edge(&mut graph, leave_edge, 1);
        let (tail, head) = graph.edge_endpoints(enter_edge).unwrap();
        assert!(tail == NodeIndex::from(0));
        assert!(head == NodeIndex::from(4) || head == NodeIndex::from(5));
    }

    #[test]
    fn get_path_in_tree_low_lim_graph() {
        let (mut graph, ..) = GraphBuilder::new(&LOW_LIM_GRAPH)
            .with_tree_edges(&LOW_LIM_GRAPH)
            .with_low_lim_values(&LOW_LIM_GRAPH_LOW_LIM_VALUES)
            .build();

        init_cutvalues(&mut graph);

        let tail = 6.into();
        let head = 8.into();
        let edge = graph.add_edge(tail, head, Edge::default());
        let actual_lca = remove_outdated_cut_values(&mut graph, edge, edge);
        let expected_path = [(5, 6), (4, 5), (4, 8)].into_iter().map(|(t, h)| graph.find_edge(t.into(), h.into()).unwrap()).collect::<Vec<_>>();
        for edge in graph.edge_indices() {
            if expected_path.contains(&edge) {
                assert!(graph[edge].cut_value.is_none());
            } else if graph[edge].is_tree_edge {
                assert!(graph[edge].cut_value.is_some())
            }
        }
        let expected_lca = NodeIndex::from(4_u32);
        assert_eq!(actual_lca, expected_lca);
    }

    #[test]
    fn get_path_in_tree_low_lim_graph_lca_is_root() {
        let (mut graph, ..) = GraphBuilder::new(&LOW_LIM_GRAPH)
            .with_tree_edges(&LOW_LIM_GRAPH)
            .with_low_lim_values(&LOW_LIM_GRAPH_LOW_LIM_VALUES)
            .build();
    
        init_cutvalues(&mut graph);

        let tail = 3.into();
        let head = 8.into();
        let edge = graph.add_edge(tail, head, Edge::default());
        let actual_lca = remove_outdated_cut_values(&mut graph, edge, edge);
        let expected_path = [(1, 3), (0, 1), (4, 8), (0, 4)].into_iter().map(|(t, h)| graph.find_edge(t.into(), h.into()).unwrap()).collect::<Vec<_>>();
        for edge in graph.edge_indices() {
            if expected_path.contains(&edge) {
                assert!(graph[edge].cut_value.is_none());
            } else if graph[edge].is_tree_edge {
                assert!(graph[edge].cut_value.is_some())
            }
        }
        let expected_lca = NodeIndex::from(0_u32);
        assert_eq!(actual_lca, expected_lca);
    }
}

mod update_tree {
    use petgraph::stable_graph::NodeIndex;

    use crate::phases::p1_layering::{Vertex, cut_values::update_cutvalues, low_lim::{update_low_lim, init_low_lim}, update_ranks, tests::EXAMPLE_GRAPH_FEASIBLE_TREE_NEG_CUT_VALUE};

    use super::{GraphBuilder, EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE, EXAMPLE_GRAPH, CUT_VALUES_EXAMPLE_GRAPH_NEG_CUT_VALUE, LOW_LIM_GRAPH_LOW_LIM_VALUES, LOW_LIM_GRAPH_AFTER_UPDATE, EXAMPLE_GRAPH_LOW_LIM_VALUES_NEG_CUT_VALUE};

    #[test]
    fn update_cutvalues_updated_correctly() {
        let (mut graph, _, _, removed_edge, ..) = GraphBuilder::new(&EXAMPLE_GRAPH)
            .with_tree_edges(&EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE)
            .with_cut_values(&CUT_VALUES_EXAMPLE_GRAPH_NEG_CUT_VALUE)
            .with_least_common_ancestor(0)
            .with_connecting_path(&[(4, 6), (6, 7), (3, 7), (2, 3), (1, 2), (0, 1)])
            .with_removed_edge(6, 7)
            .build();

        init_low_lim(&mut graph);

        let swap_edge = graph.find_edge(0.into(), 4.into()).unwrap();
        graph[swap_edge].is_tree_edge = true;
        update_cutvalues(&mut graph, removed_edge, swap_edge);
        let edges = EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE;

        assert_eq!(graph[graph.find_edge(edges[0].0.into(), edges[0].1.into()).unwrap()].cut_value, Some(2));
        assert_eq!(graph[graph.find_edge(edges[1].0.into(), edges[1].1.into()).unwrap()].cut_value, Some(1));
        assert_eq!(graph[graph.find_edge(edges[2].0.into(), edges[2].1.into()).unwrap()].cut_value, Some(2));
        assert_eq!(graph[graph.find_edge(edges[3].0.into(), edges[3].1.into()).unwrap()].cut_value, Some(2));
        assert_eq!(graph[graph.find_edge(edges[4].0.into(), edges[4].1.into()).unwrap()].cut_value, Some(2));
        assert_eq!(graph[graph.find_edge(edges[5].0.into(), edges[5].1.into()).unwrap()].cut_value, Some(1));
        assert_eq!(graph[graph.find_edge(edges[6].0.into(), edges[6].1.into()).unwrap()].cut_value, Some(0));
    }

    #[test]
    fn update_cutvalues_only_tree_edges_have_cut_values() {
        let (mut graph, _, _, removed_edge, ..)= GraphBuilder::new(&EXAMPLE_GRAPH)
            .with_tree_edges(&EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE)
            .with_cut_values(&CUT_VALUES_EXAMPLE_GRAPH_NEG_CUT_VALUE)
            .with_least_common_ancestor(0)
            .with_connecting_path(&[(4, 6), (6, 7), (3, 7), (2, 3), (1, 2), (0, 1)])
            .with_removed_edge(6, 7)
            .build();

        let swap_edge = graph.find_edge(0.into(), 4.into()).unwrap();
        graph[swap_edge].is_tree_edge = true;
        update_cutvalues(&mut graph, removed_edge, swap_edge);

        for e in graph.edge_weights() {
            if e.is_tree_edge {
                assert!(e.cut_value.is_some());
            } else {
                assert!(e.cut_value.is_none());
            }
        }
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
        assert_eq!(v4, Vertex::new(4, 8, Some(0.into()), true));
        assert_eq!(v5, Vertex::new(4, 4, Some(6.into()), true));
        assert_eq!(v6, Vertex::new(4, 5, Some(7.into()), true));
        assert_eq!(v7, Vertex::new(4, 6, Some(4.into()), true));
        assert_eq!(v8, Vertex::new(7, 7, Some(4.into()), true));
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
        assert_eq!(v0, Vertex::new(1, 8, None, true));
        assert_eq!(v1, Vertex::new(1, 4, Some(0.into()), true));
        assert_eq!(v2, Vertex::new(1, 3, Some(1.into()), true));
        assert_eq!(v3, Vertex::new(1, 2, Some(2.into()), true));
        assert_eq!(v4, Vertex::new(5, 7, Some(0.into()), true));
        assert_eq!(v5, Vertex::new(5, 5, Some(6.into()), true));
        assert_eq!(v6, Vertex::new(5, 6, Some(4.into()), true));
        assert_eq!(v7, Vertex::new(1, 1, Some(3.into()), true));
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

// mod integration {

//     use petgraph::stable_graph::StableDiGraph;

//     use crate::phases::p1_layering::{Vertex, Edge};

//     use super::{EXAMPLE_GRAPH, GraphBuilder};

//     fn is_correct(actual: StableDiGraph<Vertex, Edge>) -> bool {
//         // all cut values must be positive,
//         0 <= actual.graph().edge_indices()
//             .filter(|e| actual.graph[*e].is_tree_edge)
//             .filter_map(|e| actual.graph[e].cut_value)
//             .min()
//             .unwrap_or(0)
//         &&
//         // tree must be tight
//         0 == actual.graph.edge_indices()
//             .filter(|e| actual.graph[*e].is_tree_edge)
//             .map(|e| actual.slack(e))
//             .max()
//             .unwrap()
//         &&
//         // minimum rank must be 0
//         0 == actual.graph.node_weights()
//             .map(|w| w.rank)
//             .min()
//             .unwrap()
//     }

//     #[test]
//     fn run_algorithm_example_graph() {
//         let graph = GraphBuilder::new(&EXAMPLE_GRAPH).build();
//         let actual = graph.init_rank().make_tight().init_cutvalues().init_low_lim().rank();
//         assert!(is_correct(actual));
//     }

//     #[test]
//     fn run_algorithm_tree_500_nodes_three_edges_per_node() {
//         use graph_generator::GraphLayout;
//         let edges = GraphLayout::new_from_num_nodes(500, 3).build_edges().into_iter().map(|(t, h)| (t as u32, h as u32)).collect::<Vec<_>>();
//         let actual = GraphBuilder::new(&edges).build()
//             .init_rank()
//             .make_tight()
//             .init_cutvalues()
//             .init_low_lim()
//             .rank();
//         assert!(is_correct(actual));
//     }

//     #[test]
//     fn run_algorithm_random_graph_300_nodes() {
//         use graph_generator::RandomLayout;
//         let edges = RandomLayout::new(1000).build_edges().into_iter().map(|(t, h)| (t as u32, h as u32)).collect::<Vec<_>>();
//         println!("built random layout");
//         let actual = GraphBuilder::new(&edges).build()
//             .init_rank()
//             .make_tight()
//             .init_cutvalues()
//             .init_low_lim()
//             .rank();
//         assert!(is_correct(actual));
//     }
// }