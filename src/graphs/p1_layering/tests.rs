use std::marker::PhantomData;

use petgraph::stable_graph::{StableDiGraph, NodeIndex, EdgeIndex};

use super::{InitialRanks, UnlayeredGraph, Vertex, Edge, TightTree, InitLowLim, UpdateTree, FeasibleTree};

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

pub(crate) trait GraphBuilder {
    type Target;

    fn new(edges: &[(u32, u32)]) -> Self; 
    fn inner(&mut self) -> &mut StableDiGraph<Vertex, Edge>; 
    fn minimum_length(&mut self) -> &mut i32;
    fn build(self) -> Self::Target;

    // Blanket implementations
    fn with_minimum_length(mut self, minimum_length: u32) -> Self 
    where Self: Sized 
    {
        *self.minimum_length() = minimum_length as i32;
        self
    }

    fn with_cut_values(mut self, cut_values: &[(u32, u32, i32)]) -> Self 
    where Self: Sized
    {
        for (tail, head, cut_value) in cut_values {
            // ignore any edges that do not exist
            if let Some(edge) = self.inner().find_edge((*tail).into(), (*head).into()) {
                self.inner()[edge].cut_value = Some(*cut_value);
            }
        }

        self
    }

    fn with_tree_edges(mut self, tree_edges: &[(u32, u32)]) -> Self 
    where Self: Sized
    {
        for (tail, head) in tree_edges {
            let edge = self.inner().find_edge((*tail).into(), (*head).into()).expect("Edge not found in tree.");
            self.inner()[edge].is_tree_edge = true;
        }
        self
    }

    /// Add low_lim_values to the graph.
    /// 
    /// Input is given by a slice of (NodeIndex, low, lim, parent).
    /// 
    /// Panics if vertex is not contained in tree.
    fn with_low_lim_values(mut self, low_lim: &[(u32, u32, u32, Option<u32>)]) -> Self 
    where Self: Sized
    {
        for (id, low, lim, parent) in low_lim {
            let mut weight = self.inner().node_weight_mut((*id).into()).unwrap();
            weight.low = *low;
            weight.lim = *lim;
            weight.parent = parent.map(|p| p.into());
        }

        self
    }
}

macro_rules! impl_graph_builder {
    ($t:ty, $target:ty) => {
        impl GraphBuilder for $t {
            type Target = $target;

            fn new(edges: &[(u32, u32)]) -> Self {
                let graph = StableDiGraph::<Vertex, Edge>::from_edges(edges);
                Self { _inner: graph, minimum_length: 1 }
            }

            fn inner(&mut self) -> &mut StableDiGraph<Vertex, Edge> {
                &mut self._inner
            } 

            fn minimum_length(&mut self) -> &mut i32 {
                &mut self.minimum_length
            }

            fn build(self) -> Self::Target {
                let Self { _inner, minimum_length } = self;
                Self::Target { graph: _inner, minimum_length }
            }
        }
    };
}


pub(crate) struct Builder<T> {
    phantom: PhantomData<T>,
}

impl<T: GraphBuilder> Builder<T> {
    pub(crate) fn from_edges(edges: &[(u32, u32)]) -> T {
        T::new(edges)
    }
}

pub(crate) struct UnlayeredGraphBuilder {
    _inner: StableDiGraph<Vertex, Edge>,
    minimum_length: i32,
}

impl_graph_builder!(UnlayeredGraphBuilder, UnlayeredGraph);

pub(crate) struct InitialRanksBuilder {
    _inner: StableDiGraph<Vertex, Edge>,
    minimum_length: i32,
}

impl_graph_builder!(InitialRanksBuilder, InitialRanks);

pub(crate) struct TightTreeBuilder {
    _inner: StableDiGraph<Vertex, Edge>,
    minimum_length: i32,
}

impl_graph_builder!(TightTreeBuilder, TightTree);

pub(crate) struct InitLowLimBuilder {
    _inner: StableDiGraph<Vertex, Edge>,
    minimum_length: i32,
}

impl_graph_builder!(InitLowLimBuilder, InitLowLim);

pub(crate) struct FeasibleTreeBuilder {
    _inner: StableDiGraph<Vertex, Edge>,
    minimum_length: i32,
}

impl_graph_builder!(FeasibleTreeBuilder, FeasibleTree);

pub(crate) struct UpdateTreeBuilder {
    _inner: StableDiGraph<Vertex, Edge>,
    minimum_length: i32,
    connecting_path: Vec<EdgeIndex>,
    removed_edge: EdgeIndex,
    least_common_ancestor: NodeIndex,
}

impl GraphBuilder for UpdateTreeBuilder {
    type Target = UpdateTree;

    fn new(edges: &[(u32, u32)]) -> Self {
        let _inner = StableDiGraph::from_edges(edges);
        Self {
            _inner,
            minimum_length: 1,
            connecting_path: Vec::new(),
            removed_edge: EdgeIndex::from(0),
            least_common_ancestor: NodeIndex::from(0),
        }
    }

    fn build(self) -> Self::Target {
        Self::Target {
            graph: self._inner,
            minimum_length: self.minimum_length,
            connecting_path: self.connecting_path,
            removed_edge: self.removed_edge,
            least_common_ancestor: self.least_common_ancestor
        }
    }

    fn inner(&mut self) -> &mut StableDiGraph<Vertex, Edge> {
        &mut self._inner
    }

    fn minimum_length(&mut self) -> &mut i32 {
        &mut self.minimum_length
    }
}

impl UpdateTreeBuilder {
    /// Set the connecting path, which connects the vertices of the edge
    /// which will be swapped in the tree.
    /// Panics if edge is not contained in graph.
    fn with_connecting_path(mut self, connecting_path: &[(u32, u32)]) -> Self {
        self.connecting_path = connecting_path.into_iter()
                                              .map(|(tail, head)| self._inner.find_edge_undirected((*tail).into(), (*head).into()).unwrap().0)
                                              .collect::<Vec<_>>();
        self
    }

    /// Set the edge that was removed during rank procedure.
    /// Panics if edge is not contained in graph.
    fn with_removed_edge(mut self, tail: u32, head: u32) -> Self {
        self.removed_edge = self._inner.find_edge(tail.into(), head.into()).unwrap();
        self._inner[self.removed_edge].is_tree_edge = false;
        self
    }

    /// Set the ranks, of the vertices.
    /// Used to check if outtdated ranks are updated
    /// correctly after update_ranks is run.
    /// 
    /// Input is  [(NodeIndex, rank)].
    /// 
    /// Panics if node is not part of graph.
    fn with_ranks(mut self, ranks: &[(u32, i32)]) -> Self {
        for (n, r) in ranks {
            self._inner[NodeIndex::from(*n)].rank = *r;
        }
        self
    }

    /// Set the least common ancestor, which is the root
    /// of the path connecting the edge that will be added 
    /// to the graph.
    /// 
    /// Panics if vertex is not part of graph.
    fn with_least_common_ancestor(mut self, lca: u32) -> Self {
        assert!(self._inner.contains_node(NodeIndex::from(lca)));
        self.least_common_ancestor = lca.into();
        self
    }
}

mod unlayered_graph {
    use petgraph::Direction::{Incoming, Outgoing};

    use super::{Builder, GraphBuilder, EXAMPLE_GRAPH, UnlayeredGraphBuilder};

    #[test]
    fn test_initial_ranking_correct_order() {
        let initial_ranks = Builder::<UnlayeredGraphBuilder>::from_edges(&EXAMPLE_GRAPH)
                             .build()
                             .init_rank();
        
        let g = &initial_ranks.graph;

        for v in g.node_indices() {
            // all incoming neighbors need to have lower ranks,
            for inc in g.neighbors_directed(v, Incoming) {
                assert!(g[v].rank > g[inc].rank);
            }
            // all outgoing higher
            for out in g.neighbors_directed(v, Outgoing) {
                assert!(g[v].rank < g[out].rank);
            }
        }
    }

    #[test]
    fn test_initial_ranking_at_least_minimum_length_2() {
        let actual = Builder::<UnlayeredGraphBuilder>::from_edges(&EXAMPLE_GRAPH)
                             .with_minimum_length(2)
                             .build()
                             .init_rank();

        let g = &actual.graph;

        for v in g.node_indices() {
            for n in g.neighbors_undirected(v) {
                assert!(g[v].rank.abs_diff(g[n].rank) as i32 >= actual.minimum_length )
            }
        }
    }
} 

mod initial_ranks {
    use crate::graphs::p1_layering::{traits::Slack, tests::UnlayeredGraphBuilder};

    use super::{Builder, GraphBuilder, EXAMPLE_GRAPH, EXAMPLE_GRAPH_NON_TIGHT_INITIAL_RANKING};

    #[test]
    fn test_make_tight_is_spanning_tree() {
        let actual = Builder::<UnlayeredGraphBuilder>::from_edges(&EXAMPLE_GRAPH).build().init_rank().make_tight();
        // needs to have exactly n - 1 tree edges 
        assert_eq!(actual.graph.edge_weights().filter(|e| e.is_tree_edge).count(), actual.graph.node_indices().count() - 1);
    }

    #[test]
    fn test_make_tight_is_actually_tight() {
        // all tree edges need to have minimum slack
        let actual = Builder::<UnlayeredGraphBuilder>::from_edges(&EXAMPLE_GRAPH).build().init_rank().make_tight();
        for edge in actual.graph.edge_indices() {
            if actual.graph[edge].is_tree_edge {
                assert_eq!(actual.slack(edge), 0);
            }
        }
    }

    #[test]
    fn test_make_tight_is_spanning_tree_non_tight_initial_ranking() {
        let actual = Builder::<UnlayeredGraphBuilder>::from_edges(&EXAMPLE_GRAPH_NON_TIGHT_INITIAL_RANKING).build().init_rank().make_tight();
        // needs to have exactly n - 1 tree edges 
        assert_eq!(actual.graph.edge_weights().filter(|e| e.is_tree_edge).count(), actual.graph.node_indices().count() - 1);
    }

    #[test]
    fn test_make_tight_is_actually_tight_non_tight_inital_ranking() {
        let actual = Builder::<UnlayeredGraphBuilder>::from_edges(&EXAMPLE_GRAPH_NON_TIGHT_INITIAL_RANKING).build().init_rank().make_tight();
        for edge in actual.graph.edge_indices() {
            if actual.graph[edge].is_tree_edge {
                assert_eq!(actual.slack(edge), 0);
            }
        }
    }
}

mod tight_tree {
    use crate::graphs::p1_layering::tests::{EXAMPLE_GRAPH_FEASIBLE_TREE_NEG_CUT_VALUE, EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE};

    use super::{TightTreeBuilder, Builder, EXAMPLE_GRAPH, GraphBuilder};

    #[test]
    fn test_cut_values_one_negative() {
        let edges = EXAMPLE_GRAPH_FEASIBLE_TREE_NEG_CUT_VALUE; 
        let actual = Builder::<TightTreeBuilder>::from_edges(&EXAMPLE_GRAPH).with_tree_edges(&edges).build().init_cutvalues();
        let g = &actual.graph;
        assert_eq!(g[g.find_edge(edges[0].0.into(), edges[0].1.into()).unwrap()].cut_value, Some(3));
        assert_eq!(g[g.find_edge(edges[1].0.into(), edges[1].1.into()).unwrap()].cut_value, Some(3));
        assert_eq!(g[g.find_edge(edges[2].0.into(), edges[2].1.into()).unwrap()].cut_value, Some(3));
        assert_eq!(g[g.find_edge(edges[3].0.into(), edges[3].1.into()).unwrap()].cut_value, Some(3));
        assert_eq!(g[g.find_edge(edges[4].0.into(), edges[4].1.into()).unwrap()].cut_value, Some(0));
        assert_eq!(g[g.find_edge(edges[5].0.into(), edges[5].1.into()).unwrap()].cut_value, Some(0));
        assert_eq!(g[g.find_edge(edges[6].0.into(), edges[6].1.into()).unwrap()].cut_value, Some(-1));
    }

    #[test]
    fn test_cut_values_all_positive() {
        let edges = EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE;
        let actual = Builder::<TightTreeBuilder>::from_edges(&EXAMPLE_GRAPH).with_tree_edges(&edges).build().init_cutvalues();
        let g = &actual.graph;
        assert_eq!(g[g.find_edge(edges[0].0.into(), edges[0].1.into()).unwrap()].cut_value, Some(2));
        assert_eq!(g[g.find_edge(edges[1].0.into(), edges[1].1.into()).unwrap()].cut_value, Some(1));
        assert_eq!(g[g.find_edge(edges[2].0.into(), edges[2].1.into()).unwrap()].cut_value, Some(2));
        assert_eq!(g[g.find_edge(edges[3].0.into(), edges[3].1.into()).unwrap()].cut_value, Some(2));
        assert_eq!(g[g.find_edge(edges[4].0.into(), edges[4].1.into()).unwrap()].cut_value, Some(2));
        assert_eq!(g[g.find_edge(edges[5].0.into(), edges[5].1.into()).unwrap()].cut_value, Some(1));
        assert_eq!(g[g.find_edge(edges[6].0.into(), edges[6].1.into()).unwrap()].cut_value, Some(0));
    }
}

mod init_low_lim {
    use petgraph::stable_graph::NodeIndex;

    use crate::graphs::p1_layering::Vertex;

    use super::{InitLowLimBuilder, LOW_LIM_GRAPH, Builder, GraphBuilder, EXAMPLE_GRAPH, EXAMPLE_GRAPH_FEASIBLE_TREE_NEG_CUT_VALUE};

    #[test]
    fn init_low_lim_low_lim_graph() {
        let actual = Builder::<InitLowLimBuilder>::from_edges(&LOW_LIM_GRAPH)
                    .with_tree_edges(&LOW_LIM_GRAPH)
                    .build()
                    .init_low_lim();

        let g = &actual.graph;
        let v0 = g[NodeIndex::from(0)];
        let v1 = g[NodeIndex::from(1)];
        let v2 = g[NodeIndex::from(2)];
        let v3 = g[NodeIndex::from(3)];
        let v4 = g[NodeIndex::from(4)];
        let v5 = g[NodeIndex::from(5)];
        let v6 = g[NodeIndex::from(6)];
        let v7 = g[NodeIndex::from(7)];
        let v8 = g[NodeIndex::from(8)];

        assert_eq!(v0, Vertex::new(1, 9, None));
        assert_eq!(v1, Vertex::new(1, 3, Some(0.into())));
        assert_eq!(v2, Vertex::new(1, 1, Some(1.into())));
        assert_eq!(v3, Vertex::new(2, 2, Some(1.into())));
        assert_eq!(v4, Vertex::new(4, 8, Some(0.into())));
        assert_eq!(v5, Vertex::new(4, 5, Some(4.into())));
        assert_eq!(v6, Vertex::new(4, 4, Some(5.into())));
        assert_eq!(v7, Vertex::new(6, 6, Some(4.into())));
        assert_eq!(v8, Vertex::new(7, 7, Some(4.into())));
    }

    #[test]
    fn test_init_low_lim_neg_cut_value() {
        let ft = Builder::<InitLowLimBuilder>::from_edges(&EXAMPLE_GRAPH)
            .with_tree_edges(&EXAMPLE_GRAPH_FEASIBLE_TREE_NEG_CUT_VALUE)
            .build()
            .init_low_lim();

        let g = &ft.graph;
        assert_eq!(g[NodeIndex::from(0)].low, 1);
        assert_eq!(g[NodeIndex::from(0)].lim, 8);
        assert_eq!(g[NodeIndex::from(0)].parent, None);
        assert_eq!(g[NodeIndex::from(1)].low, 1);
        assert_eq!(g[NodeIndex::from(1)].lim, 7);
        assert_eq!(g[NodeIndex::from(1)].parent, Some(0.into()));
        assert_eq!(g[NodeIndex::from(2)].low, 1);
        assert_eq!(g[NodeIndex::from(2)].lim, 6);
        assert_eq!(g[NodeIndex::from(2)].parent, Some(1.into()));
        assert_eq!(g[NodeIndex::from(3)].low, 1);
        assert_eq!(g[NodeIndex::from(3)].lim, 5);
        assert_eq!(g[NodeIndex::from(3)].parent, Some(2.into()));
        assert!(g[NodeIndex::from(4)].low == 1 || g[NodeIndex::from(4)].low == 2);
        assert!(g[NodeIndex::from(4)].lim == 1 || g[NodeIndex::from(4)].lim == 2);
        assert_eq!(g[NodeIndex::from(4)].parent, Some(6.into()));
        assert!(g[NodeIndex::from(5)].low == 1 || g[NodeIndex::from(5)].low == 2);
        assert!(g[NodeIndex::from(5)].lim == 1 || g[NodeIndex::from(5)].lim == 2);
        assert_eq!(g[NodeIndex::from(5)].parent, Some(6.into()));
        assert_eq!(g[NodeIndex::from(6)].low, 1);
        assert_eq!(g[NodeIndex::from(6)].lim, 3);
        assert_eq!(g[NodeIndex::from(6)].parent, Some(7.into()));
        assert_eq!(g[NodeIndex::from(7)].low, 1);
        assert_eq!(g[NodeIndex::from(7)].lim, 4);
        assert_eq!(g[NodeIndex::from(7)].parent, Some(3.into()));
    }
}

mod feasible_tree {
    use petgraph::stable_graph::NodeIndex;

    use crate::graphs::p1_layering::{tests::{FeasibleTreeBuilder, LOW_LIM_GRAPH, LOW_LIM_GRAPH_LOW_LIM_VALUES}, Edge};

    use super::{Builder, EXAMPLE_GRAPH, GraphBuilder, EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE, EXAMPLE_GRAPH_FEASIBLE_TREE_NEG_CUT_VALUE, TightTreeBuilder};

    #[test]
    fn leave_edge_has_negative_cut_value() {
        let feasible_tree = Builder::<TightTreeBuilder>::from_edges(&EXAMPLE_GRAPH)
                                                 .with_tree_edges(&EXAMPLE_GRAPH_FEASIBLE_TREE_NEG_CUT_VALUE)
                                                 .build()
                                                 .init_cutvalues()
                                                 .init_low_lim();
        
        let leave_edge = feasible_tree.leave_edge();
        assert!(leave_edge.is_some());
        let (tail, head) = feasible_tree.graph.edge_endpoints(leave_edge.unwrap()).unwrap();
        assert_eq!(tail, NodeIndex::from(6));
        assert_eq!(head, NodeIndex::from(7));
    }

    #[test]
    fn leave_edge_has_no_negative_cut_value() {
        let feasible_tree = Builder::<TightTreeBuilder>::from_edges(&EXAMPLE_GRAPH)
                                                 .with_tree_edges(&EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE)
                                                 .build()
                                                 .init_cutvalues()
                                                 .init_low_lim();
        
        let leave_edge = feasible_tree.leave_edge();
        assert!(leave_edge.is_none());
    }

    #[test]
    fn test_is_head_to_tail_true_root_in_tail() {
        // u is always considered to be the tail of the edge to be swapped
        let mut ft = Builder::<FeasibleTreeBuilder>::from_edges(&LOW_LIM_GRAPH).with_tree_edges(&LOW_LIM_GRAPH).with_low_lim_values(&LOW_LIM_GRAPH_LOW_LIM_VALUES).build();
        let u = 5;
        let u = *ft.graph.node_weight(u.into()).unwrap();
        let tail = 6;
        let head = 7;
        let edge = ft.graph.add_edge(tail.into(), head.into(), Edge::default());
        assert!(ft.is_head_to_tail(edge, u, false));
    }

    #[test]
    fn test_is_head_to_tail_true_root_in_head() {
        let mut ft = Builder::<FeasibleTreeBuilder>::from_edges(&LOW_LIM_GRAPH).with_tree_edges(&LOW_LIM_GRAPH).with_low_lim_values(&LOW_LIM_GRAPH_LOW_LIM_VALUES).build();
        let u = 4;
        let u = *ft.graph.node_weight(u.into()).unwrap();
        let tail = 3;
        let head = 5;
        let edge = ft.graph.add_edge(tail.into(), head.into(), Edge::default());
        assert!(ft.is_head_to_tail(edge, u, true));
    }

    #[test]
    fn test_is_head_to_tail_false_root_in_tail() {
        let mut ft = Builder::<FeasibleTreeBuilder>::from_edges(&LOW_LIM_GRAPH).with_tree_edges(&LOW_LIM_GRAPH).with_low_lim_values(&LOW_LIM_GRAPH_LOW_LIM_VALUES).build();
        let u = 3;
        let u = *ft.graph.node_weight(u.into()).unwrap();
        let tail = 4;
        let head = 3;
        let edge = ft.graph.add_edge(tail.into(), head.into(), Edge::default());
        assert!(!ft.is_head_to_tail(edge, u, false));
    }

    #[test]
    fn test_is_head_to_tail_false_root_in_head() {
        let mut ft = Builder::<FeasibleTreeBuilder>::from_edges(&LOW_LIM_GRAPH).with_tree_edges(&LOW_LIM_GRAPH).with_low_lim_values(&LOW_LIM_GRAPH_LOW_LIM_VALUES).build();
        let u = 2;
        let u = *ft.graph.node_weight(u.into()).unwrap();
        let tail = 2;
        let head = 0;
        let edge = ft.graph.add_edge(tail.into(), head.into(), Edge::default());
        assert!(!ft.is_head_to_tail(edge, u, true));
    }

    #[test]
    fn enter_edge_find_edge() {
        let mut feasible_tree = Builder::<TightTreeBuilder>::from_edges(&EXAMPLE_GRAPH)
            .with_tree_edges(&EXAMPLE_GRAPH_FEASIBLE_TREE_NEG_CUT_VALUE)
            .build()
            .init_cutvalues()
            .init_low_lim();
        
        let leave_edge = feasible_tree.leave_edge().unwrap();
        let enter_edge = feasible_tree.enter_edge(leave_edge);
        let (tail, head) = feasible_tree.graph.edge_endpoints(enter_edge).unwrap();
        assert!(tail == NodeIndex::from(0));
        assert!(head == NodeIndex::from(4) || head == NodeIndex::from(5));
    }

    #[test]
    fn get_path_in_tree_low_lim_graph() {
        let mut ft = Builder::<FeasibleTreeBuilder>::from_edges(&LOW_LIM_GRAPH)
            .with_tree_edges(&LOW_LIM_GRAPH)
            .with_low_lim_values(&LOW_LIM_GRAPH_LOW_LIM_VALUES)
            .build();

        let tail = 6.into();
        let head = 8.into();
        let edge = ft.graph.add_edge(tail, head, Edge::default());
        let (actual_path, actual_lca) = ft.get_path_in_tree(edge);
        let expected_path = [(5, 6), (4, 5), (4, 8)].into_iter().map(|(t, h)| ft.graph.find_edge(t.into(), h.into()).unwrap()).collect::<Vec<_>>();
        let expected_lca = NodeIndex::from(4_u32);
        assert_eq!(actual_path, expected_path);
        assert_eq!(actual_lca, expected_lca);
    }

    #[test]
    fn get_path_in_tree_low_lim_graph_lca_is_root() {
        let mut ft = Builder::<FeasibleTreeBuilder>::from_edges(&LOW_LIM_GRAPH)
            .with_tree_edges(&LOW_LIM_GRAPH)
            .with_low_lim_values(&LOW_LIM_GRAPH_LOW_LIM_VALUES)
            .build();

        let tail = 3.into();
        let head = 8.into();
        let edge = ft.graph.add_edge(tail, head, Edge::default());
        let (actual_path, actual_lca) = ft.get_path_in_tree(edge);
        let expected_path = [(1, 3), (0, 1), (4, 8), (0, 4)].into_iter().map(|(t, h)| ft.graph.find_edge(t.into(), h.into()).unwrap()).collect::<Vec<_>>();
        let expected_lca = NodeIndex::from(0_u32);
        assert_eq!(actual_path, expected_path);
        assert_eq!(actual_lca, expected_lca);
    }
}

mod update_tree {
    use petgraph::stable_graph::NodeIndex;

    use crate::graphs::p1_layering::Vertex;

    use super::{UpdateTreeBuilder, Builder, GraphBuilder, EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE, EXAMPLE_GRAPH, CUT_VALUES_EXAMPLE_GRAPH_NEG_CUT_VALUE, LOW_LIM_GRAPH_LOW_LIM_VALUES, LOW_LIM_GRAPH_AFTER_UPDATE, EXAMPLE_GRAPH_LOW_LIM_VALUES_NEG_CUT_VALUE};

    #[test]
    fn update_cutvalues_updated_correctly() {
        let mut update = Builder::<UpdateTreeBuilder>::from_edges(&EXAMPLE_GRAPH)
            .with_tree_edges(&EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE)
            .with_cut_values(&CUT_VALUES_EXAMPLE_GRAPH_NEG_CUT_VALUE)
            .with_least_common_ancestor(0)
            .with_connecting_path(&[(4, 6), (6, 7), (3, 7), (2, 3), (1, 2), (0, 1)])
            .with_removed_edge(6, 7)
            .build();

        let swap_edge = update.graph.find_edge(0.into(), 4.into()).unwrap();
        update.graph[swap_edge].is_tree_edge = true;
        update.update_cutvalues();
        let edges = EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE;
        let g = &update.graph;

        assert_eq!(g[g.find_edge(edges[0].0.into(), edges[0].1.into()).unwrap()].cut_value, Some(2));
        assert_eq!(g[g.find_edge(edges[1].0.into(), edges[1].1.into()).unwrap()].cut_value, Some(1));
        assert_eq!(g[g.find_edge(edges[2].0.into(), edges[2].1.into()).unwrap()].cut_value, Some(2));
        assert_eq!(g[g.find_edge(edges[3].0.into(), edges[3].1.into()).unwrap()].cut_value, Some(2));
        assert_eq!(g[g.find_edge(edges[4].0.into(), edges[4].1.into()).unwrap()].cut_value, Some(2));
        assert_eq!(g[g.find_edge(edges[5].0.into(), edges[5].1.into()).unwrap()].cut_value, Some(1));
        assert_eq!(g[g.find_edge(edges[6].0.into(), edges[6].1.into()).unwrap()].cut_value, Some(0));
    }

    #[test]
    fn update_cutvalues_only_tree_edges_have_cut_values() {
        let mut update = Builder::<UpdateTreeBuilder>::from_edges(&EXAMPLE_GRAPH)
            .with_tree_edges(&EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE)
            .with_cut_values(&CUT_VALUES_EXAMPLE_GRAPH_NEG_CUT_VALUE)
            .with_least_common_ancestor(0)
            .with_connecting_path(&[(4, 6), (6, 7), (3, 7), (2, 3), (1, 2), (0, 1)])
            .with_removed_edge(6, 7)
            .build();

        let swap_edge = update.graph.find_edge(0.into(), 4.into()).unwrap();
        update.graph[swap_edge].is_tree_edge = true;
        update.update_cutvalues();

        for e in update.graph.edge_weights() {
            if e.is_tree_edge {
                assert!(e.cut_value.is_some());
            } else {
                assert!(e.cut_value.is_none());
            }
        }
    }

    #[test]
    fn update_low_lim_low_lim_graph() {
        let mut update = Builder::<UpdateTreeBuilder>::from_edges(&LOW_LIM_GRAPH_AFTER_UPDATE)
            .with_tree_edges(&LOW_LIM_GRAPH_AFTER_UPDATE)
            .with_low_lim_values(&LOW_LIM_GRAPH_LOW_LIM_VALUES)
            .with_least_common_ancestor(4)
            .build();

        update.update_low_lim(); 
        let v4 = update.graph[NodeIndex::from(4)];
        let v5 = update.graph[NodeIndex::from(5)];
        let v6 = update.graph[NodeIndex::from(6)];
        let v7 = update.graph[NodeIndex::from(7)];
        let v8 = update.graph[NodeIndex::from(8)];
        assert_eq!(v4, Vertex::new(4, 8, Some(0.into())));
        assert_eq!(v5, Vertex::new(4, 4, Some(6.into())));
        assert_eq!(v6, Vertex::new(4, 5, Some(7.into())));
        assert_eq!(v7, Vertex::new(4, 6, Some(4.into())));
        assert_eq!(v8, Vertex::new(7, 7, Some(4.into())));
    }

    #[test]
    fn update_low_lim_example_graph() {
        let mut update = Builder::<UpdateTreeBuilder>::from_edges(&EXAMPLE_GRAPH)
            .with_tree_edges(&EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE)
            .with_low_lim_values(&EXAMPLE_GRAPH_LOW_LIM_VALUES_NEG_CUT_VALUE)
            .with_least_common_ancestor(0)
            .build();

        update.update_low_lim();
        let v0 = update.graph[NodeIndex::from(0)];
        let v1 = update.graph[NodeIndex::from(1)];
        let v2 = update.graph[NodeIndex::from(2)];
        let v3 = update.graph[NodeIndex::from(3)];
        let v4 = update.graph[NodeIndex::from(4)];
        let v5 = update.graph[NodeIndex::from(5)];
        let v6 = update.graph[NodeIndex::from(6)];
        let v7 = update.graph[NodeIndex::from(7)];
        assert_eq!(v0, Vertex::new(1, 8, None));
        assert_eq!(v1, Vertex::new(1, 4, Some(0.into())));
        assert_eq!(v2, Vertex::new(1, 3, Some(1.into())));
        assert_eq!(v3, Vertex::new(1, 2, Some(2.into())));
        assert_eq!(v4, Vertex::new(5, 7, Some(0.into())));
        assert_eq!(v5, Vertex::new(5, 5, Some(6.into())));
        assert_eq!(v6, Vertex::new(5, 6, Some(4.into())));
        assert_eq!(v7, Vertex::new(1, 1, Some(3.into())));
    }

    #[test]
    fn update_ranks() {
        let mut update = Builder::<UpdateTreeBuilder>::from_edges(&EXAMPLE_GRAPH)
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
        update.update_ranks();
            
        for id in update.graph.node_indices() {
            let rank = update.graph[id].rank;
            let id = id.index();
            assert_eq!(expected[id], (id, rank));
        }
    }
}

mod integration {
    use crate::graphs::p1_layering::{traits::Slack, FeasibleTree};

    use super::{Builder, EXAMPLE_GRAPH, UnlayeredGraphBuilder, GraphBuilder};

    fn is_correct(actual: FeasibleTree) -> bool {
        // all cut values must be positive,
        0 <= actual.graph.edge_indices()
            .filter(|e| actual.graph[*e].is_tree_edge)
            .map(|e| actual.graph[e].cut_value.unwrap())
            .min()
            .unwrap()
        &&
        // tree must be tight
        0 == actual.graph.edge_indices()
            .filter(|e| actual.graph[*e].is_tree_edge)
            .map(|e| actual.slack(e))
            .max()
            .unwrap()
        &&
        // minimum rank must be 0
        0 == actual.graph.node_weights()
            .map(|w| w.rank)
            .min()
            .unwrap()
    }

    #[test]
    fn run_algorithm_example_graph() {
        let graph = Builder::<UnlayeredGraphBuilder>::from_edges(&EXAMPLE_GRAPH).build();
        let actual = graph.init_rank().make_tight().init_cutvalues().init_low_lim().rank();
        assert!(is_correct(actual));
    }
}

mod benchmark {
    use super::{Builder, UnlayeredGraphBuilder, GraphBuilder};

    #[test]
    fn time_10_v() {
        let edges = graph_generator::GraphLayout::new_from_num_nodes(10, 2).build_edges().into_iter().map(|(a, b)| (a as u32, b as u32)).collect::<Vec<_>>();
        let start = std::time::Instant::now();
        let _ = Builder::<UnlayeredGraphBuilder>::from_edges(&edges).build().init_rank().make_tight().init_cutvalues().init_low_lim().rank();
        println!("10 Vertices, 2 Edges per vertice: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn time_100_v() {
        let edges = graph_generator::GraphLayout::new_from_num_nodes(100, 2).build_edges().into_iter().map(|(a, b)| (a as u32, b as u32)).collect::<Vec<_>>();
        let start = std::time::Instant::now();
        let _ = Builder::<UnlayeredGraphBuilder>::from_edges(&edges).build().init_rank().make_tight().init_cutvalues().init_low_lim().rank();
        println!("100 Vertices, 2 Edges per vertice: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn time_1000_v() {
        let edges = graph_generator::GraphLayout::new_from_num_nodes(1000, 2).build_edges().into_iter().map(|(a, b)| (a as u32, b as u32)).collect::<Vec<_>>();
        let start = std::time::Instant::now();
        let _ = Builder::<UnlayeredGraphBuilder>::from_edges(&edges).build().init_rank().make_tight().init_cutvalues().init_low_lim().rank();
        println!("1000 Vertices, 2 Edges per vertice: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn time_10000_v() {
        let edges = graph_generator::GraphLayout::new_from_num_nodes(10000, 2).build_edges().into_iter().map(|(a, b)| (a as u32, b as u32)).collect::<Vec<_>>();
        let start = std::time::Instant::now();
        let _ = Builder::<UnlayeredGraphBuilder>::from_edges(&edges).build().init_rank().make_tight().init_cutvalues().init_low_lim().rank();
        println!("10000 Vertices, 2 Edges per vertice: {}ms", start.elapsed().as_millis());
    }
    #[test]
    fn time_100000_v() {
        let edges = graph_generator::GraphLayout::new_from_num_nodes(100000, 2).build_edges().into_iter().map(|(a, b)| (a as u32, b as u32)).collect::<Vec<_>>();
        let start = std::time::Instant::now();
        let _ = Builder::<UnlayeredGraphBuilder>::from_edges(&edges).build().init_rank().make_tight().init_cutvalues().init_low_lim().rank();
        println!("100000 Vertices, 2 Edges per vertice: {}ms", start.elapsed().as_millis());
    }
    
    #[test]
    fn time_1000000_v() {
        let edges = graph_generator::GraphLayout::new_from_num_nodes(1000000, 2).build_edges().into_iter().map(|(a, b)| (a as u32, b as u32)).collect::<Vec<_>>();
        let start = std::time::Instant::now();
        let _ = Builder::<UnlayeredGraphBuilder>::from_edges(&edges).build().init_rank().make_tight().init_cutvalues().init_low_lim().rank();
        println!("1000000 Vertices, 2 Edges per vertice: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn time_100000_v_3_e() {
        let edges = graph_generator::GraphLayout::new_from_num_nodes(100000, 3).build_edges().into_iter().map(|(a, b)| (a as u32, b as u32)).collect::<Vec<_>>();
        let start = std::time::Instant::now();
        let _ = Builder::<UnlayeredGraphBuilder>::from_edges(&edges).build().init_rank().make_tight().init_cutvalues().init_low_lim().rank();
        println!("100000 Vertices, 3 Edges per vertice: {}ms", start.elapsed().as_millis());
    }
    
    #[test]
    fn time_1000000_v_3_e() {
        let edges = graph_generator::GraphLayout::new_from_num_nodes(1000000, 3).build_edges().into_iter().map(|(a, b)| (a as u32, b as u32)).collect::<Vec<_>>();
        let start = std::time::Instant::now();
        let _ = Builder::<UnlayeredGraphBuilder>::from_edges(&edges).build().init_rank().make_tight().init_cutvalues().init_low_lim().rank();
        println!("1000000 Vertices, 3 Edges per vertice: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn time_100000_v_4_e() {
        let edges = graph_generator::GraphLayout::new_from_num_nodes(100000, 4).build_edges().into_iter().map(|(a, b)| (a as u32, b as u32)).collect::<Vec<_>>();
        let start = std::time::Instant::now();
        let _ = Builder::<UnlayeredGraphBuilder>::from_edges(&edges).build().init_rank().make_tight().init_cutvalues().init_low_lim().rank();
        println!("100000 Vertices, 4 Edges per vertice: {}ms", start.elapsed().as_millis());
    }
    
    #[test]
    fn time_1000000_v_4_e() {
        let edges = graph_generator::GraphLayout::new_from_num_nodes(1000000, 4).build_edges().into_iter().map(|(a, b)| (a as u32, b as u32)).collect::<Vec<_>>();
        let start = std::time::Instant::now();
        let _ = Builder::<UnlayeredGraphBuilder>::from_edges(&edges).build().init_rank().make_tight().init_cutvalues().init_low_lim().rank();
        println!("1000000 Vertices, 4 Edges per vertice: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn time_100000_v_8_e() {
        let edges = graph_generator::GraphLayout::new_from_num_nodes(100000, 8).build_edges().into_iter().map(|(a, b)| (a as u32, b as u32)).collect::<Vec<_>>();
        let start = std::time::Instant::now();
        let _ = Builder::<UnlayeredGraphBuilder>::from_edges(&edges).build().init_rank().make_tight().init_cutvalues().init_low_lim().rank();
        println!("100000 Vertices, 8 Edges per vertice: {}ms", start.elapsed().as_millis());
    }
    
    #[test]
    fn time_1000000_v_8_e() {
        let edges = graph_generator::GraphLayout::new_from_num_nodes(1000000, 8).build_edges().into_iter().map(|(a, b)| (a as u32, b as u32)).collect::<Vec<_>>();
        let start = std::time::Instant::now();
        let _ = Builder::<UnlayeredGraphBuilder>::from_edges(&edges).build().init_rank().make_tight().init_cutvalues().init_low_lim().rank();
        println!("1000000 Vertices, 8 Edges per vertice: {}ms", start.elapsed().as_millis());
    }
}