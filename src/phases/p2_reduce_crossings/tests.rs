use petgraph::stable_graph::{StableDiGraph, NodeIndex};

use super::{Vertex, Edge};

static ONE_DUMMY: [(u32, u32); 9] = [(0, 1), (1, 2), (2, 3), (3, 7), (4, 6), (5, 6), (6, 7), (0, 4), (0, 5)];
static ONE_DUMMY_RANKS: [(u32, u32); 8] = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 1), (5, 1), (6, 2), (7, 4)];

static THREE_DUMMIES: [(u32, u32); 10] = [
    (0, 1), (0, 2),
    (1, 4), (1, 5), (2, 3),
    (3, 8), (4, 6), (5, 7),
    (6, 7),
    (7, 8),
];
static THREE_DUMMIES_RANKS: [(u32, u32); 9] = [(0, 0), (1, 1), (2, 1), (3, 2), (4, 2), (5, 2), (6, 3), (7, 4), (8, 5)];

static COMPLEX_EXAMPLE: [(u32, u32); 21] = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 13), (3, 9), (3, 11),
    (4, 10), (5, 11), (6, 11), (7, 11), (8, 15), 
    (11, 12), (11, 13),
    (12, 14), (12, 15), (12, 13),
];
static COMPLEX_EXAMPLE_RANKS: [(u32, u32); 16] = [
    (0, 0),
    (1, 1), (2, 1), (3, 1),
    (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (9, 2),
    (10, 3), (11, 3),
    (12, 4),
    (14, 5), (15, 5), (13, 5)
];
static _TYPE_2_CONFLICT_2_COLS: [(u32, u32); 8] = [
    (0, 3), (1, 2),
    (2, 5), (3, 4),
    (4, 7), (5, 6),
    (6, 8), (7, 8),
];

static _TYPE_2_CONFLICT_2_COLS_RANKS: [(u32, u32); 9] = [
    (0, 0), (1, 0),
    (2, 1), (3, 1),
    (4, 2), (5, 2),
    (6, 3), (7, 3),
    (8, 4)
];

static _TYPE_2_CONFLICT_2_COLS_DUMMIES: [u32; 4] = [2, 3, 4, 5];


struct GraphBuilder {
    graph: StableDiGraph<Vertex, Edge>,
    minimum_length: i32,
}

impl GraphBuilder {
    fn new_from_edges_with_ranking(edges: &[(u32, u32)], ranks: &[(u32, u32)]) -> Self {
        let mut graph = StableDiGraph::<Vertex, Edge>::from_edges(edges);
        for (v, rank) in ranks {
            graph[NodeIndex::from(*v)].rank = *rank as i32;
        }
        
        Self {
            graph,
            minimum_length: 1,
        }
    }

    #[allow(dead_code)]
    fn with_minimum_length(mut self, minimum_length: i32) -> Self {
        self.minimum_length = minimum_length;
        self
    }

    #[allow(dead_code)]
    fn with_dummies(mut self, dummies: &[u32]) -> Self {
        for dummy in dummies {
            self.graph[NodeIndex::from(*dummy)].is_dummy = true;
        }
        self
    }

    fn build(self) -> (StableDiGraph<Vertex, Edge>, i32) {
        (self.graph, self.minimum_length)
    }
}

#[cfg(test)]
mod insert_dummy_vertices {

    use crate::phases::p2_reduce_crossings::{tests::{ONE_DUMMY, THREE_DUMMIES, THREE_DUMMIES_RANKS, COMPLEX_EXAMPLE, COMPLEX_EXAMPLE_RANKS}, insert_dummy_vertices};

    use super::{ONE_DUMMY_RANKS, GraphBuilder};


    #[test]
    fn insert_dummy_vertices_one_dummy() {
        let (mut graph, minimum_length) = GraphBuilder::new_from_edges_with_ranking(&ONE_DUMMY, &ONE_DUMMY_RANKS).build();
        let n_vertices = graph.node_count();
        insert_dummy_vertices(&mut graph, minimum_length);
        // one dummy vertex
        assert_eq!(graph.node_weights().filter(|w| w.is_dummy).count(), 1);
        // one more vertex
        assert_eq!(n_vertices + 1, graph.node_count())
    }

    #[test]
    fn insert_dummy_vertices_three_dummies() {
        let (mut graph, minimum_length) = GraphBuilder::new_from_edges_with_ranking(&THREE_DUMMIES, &THREE_DUMMIES_RANKS).build();
        let n_vertices = graph.node_count();
        insert_dummy_vertices(&mut graph, minimum_length);
        // one dummy vertex
        assert_eq!(graph.node_weights().filter(|w| w.is_dummy).count(), 3);
        // one more vertex
        assert_eq!(n_vertices + 3, graph.node_count())
    }

    #[test]
    fn insert_dummy_vertices_7_dummies() {
        let (mut graph, minimum_length) = GraphBuilder::new_from_edges_with_ranking(&COMPLEX_EXAMPLE, &COMPLEX_EXAMPLE_RANKS).build();
        let n_vertices = graph.node_count();
        insert_dummy_vertices(&mut graph, minimum_length);
        // one dummy vertex
        assert_eq!(graph.node_weights().filter(|w| w.is_dummy).count(), 7);
        // one more vertex
        assert_eq!(n_vertices + 7, graph.node_count())
    }
}

mod init_order {
    use crate::phases::p2_reduce_crossings::insert_dummy_vertices;

    use super::{GraphBuilder, ONE_DUMMY, ONE_DUMMY_RANKS, THREE_DUMMIES, THREE_DUMMIES_RANKS, COMPLEX_EXAMPLE, COMPLEX_EXAMPLE_RANKS};

    
    #[test]
    fn all_neighbors_must_be_at_adjacent_level_one_dummy() {
        let (mut graph, minimum_length) = GraphBuilder::new_from_edges_with_ranking(&ONE_DUMMY, &ONE_DUMMY_RANKS).build();
        insert_dummy_vertices(&mut graph, minimum_length);
        for v in graph.node_indices() {
            let rank = graph[v].rank;
            for n in graph.neighbors_undirected(v) {
                assert_eq!(rank.abs_diff(graph[n].rank), 1);
            }
        }
    }

    #[test]
    fn all_neighbors_must_be_at_adjacent_level_three_dummies() {
        let (mut graph, minimum_length) = GraphBuilder::new_from_edges_with_ranking(&THREE_DUMMIES, &THREE_DUMMIES_RANKS)
            .build();
        insert_dummy_vertices(&mut graph, minimum_length);
        for v in graph.node_indices() {
            let rank = graph[v].rank;
            for n in graph.neighbors_undirected(v) {
                assert_eq!(rank.abs_diff(graph[n].rank), 1);
            }
        }
    }

    #[test]
    fn all_neighbors_must_be_at_adjacent_level_seven_dummies() {
        let (mut graph, minimum_length) = GraphBuilder::new_from_edges_with_ranking(&COMPLEX_EXAMPLE, &COMPLEX_EXAMPLE_RANKS)
            .build();

        insert_dummy_vertices(&mut graph, minimum_length);
        for v in graph.node_indices() {
            let rank = graph[v].rank;
            for n in graph.neighbors_undirected(v) {
                assert_eq!(rank.abs_diff(graph[n].rank), 1);
            }
        }
    }
}

// TODO: Add new tests for Order crosscount
#[cfg(test)]
mod order {
    use petgraph::stable_graph::StableDiGraph;
    use crate::phases::p2_reduce_crossings::Order;

    static ORDER_TWO_CROSSINGS: [(u32, u32); 3] = [
        (0, 4), (1, 3), (2, 3)
    ];

    #[test]
    fn two_crossings() {
        let order = Order::new(vec![
            vec![0.into(), 1.into(), 2.into()], 
            vec![3.into(), 4.into()]
        ]);
        let graph = StableDiGraph::from_edges(&ORDER_TWO_CROSSINGS);
        assert_eq!(order.bilayer_cross_count(&graph, 0), 2);
    }

    #[test]
    fn four_crossings() {
        let order = Order::new(vec![
            vec![1.into(), 2.into(), 3.into(), 4.into()], 
            vec![5.into(), 6.into(), 7.into(), 8.into()]
        ]);
        let graph = StableDiGraph::from_edges(&[(1, 8), (2, 7), (3, 6), (4, 5)]);
        assert_eq!(order.bilayer_cross_count(&graph, 0), 6);
    }
}