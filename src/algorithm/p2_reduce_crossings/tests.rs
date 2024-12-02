use petgraph::stable_graph::{NodeIndex, StableDiGraph};

use super::{Edge, Vertex};

const ONE_DUMMY: [(u32, u32); 9] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (4, 6),
    (5, 6),
    (6, 7),
    (0, 4),
    (0, 5),
];
const ONE_DUMMY_RANKS: [(u32, u32); 8] = [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 1),
    (5, 1),
    (6, 2),
    (7, 4),
];

const THREE_DUMMIES: [(u32, u32); 10] = [
    (0, 1),
    (0, 2),
    (1, 4),
    (1, 5),
    (2, 3),
    (3, 8),
    (4, 6),
    (5, 7),
    (6, 7),
    (7, 8),
];
const THREE_DUMMIES_RANKS: [(u32, u32); 9] = [
    (0, 0),
    (1, 1),
    (2, 1),
    (3, 2),
    (4, 2),
    (5, 2),
    (6, 3),
    (7, 4),
    (8, 5),
];

const COMPLEX_EXAMPLE: [(u32, u32); 21] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 7),
    (1, 8),
    (1, 13),
    (3, 9),
    (3, 11),
    (4, 10),
    (5, 11),
    (6, 11),
    (7, 11),
    (8, 15),
    (11, 12),
    (11, 13),
    (12, 14),
    (12, 15),
    (12, 13),
];
const COMPLEX_EXAMPLE_RANKS: [(u32, u32); 16] = [
    (0, 0),
    (1, 1),
    (2, 1),
    (3, 1),
    (4, 2),
    (5, 2),
    (6, 2),
    (7, 2),
    (8, 2),
    (9, 2),
    (10, 3),
    (11, 3),
    (12, 4),
    (14, 5),
    (15, 5),
    (13, 5),
];
const _TYPE_2_CONFLICT_2_COLS: [(u32, u32); 8] = [
    (0, 3),
    (1, 2),
    (2, 5),
    (3, 4),
    (4, 7),
    (5, 6),
    (6, 8),
    (7, 8),
];

const _TYPE_2_CONFLICT_2_COLS_RANKS: [(u32, u32); 9] = [
    (0, 0),
    (1, 0),
    (2, 1),
    (3, 1),
    (4, 2),
    (5, 2),
    (6, 3),
    (7, 3),
    (8, 4),
];

const _TYPE_2_CONFLICT_2_COLS_DUMMIES: [u32; 4] = [2, 3, 4, 5];

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

    use petgraph::stable_graph::StableDiGraph;

    use crate::{
        algorithm::p2_reduce_crossings::{
            insert_dummy_vertices,
            tests::{
                COMPLEX_EXAMPLE, COMPLEX_EXAMPLE_RANKS, ONE_DUMMY, THREE_DUMMIES,
                THREE_DUMMIES_RANKS,
            },
        },
        configure::Config,
    };

    use super::{GraphBuilder, ONE_DUMMY_RANKS};

    #[test]
    fn insert_dummy_vertices_one_dummy() {
        let (mut graph, minimum_length) =
            GraphBuilder::new_from_edges_with_ranking(&ONE_DUMMY, &ONE_DUMMY_RANKS).build();
        let n_vertices = graph.node_count();
        insert_dummy_vertices(&mut graph, minimum_length, 0.0);
        // one dummy vertex
        assert_eq!(graph.node_weights().filter(|w| w.is_dummy).count(), 1);
        // one more vertex
        assert_eq!(n_vertices + 1, graph.node_count())
    }

    #[test]
    fn insert_dummy_vertices_three_dummies() {
        let (mut graph, minimum_length) =
            GraphBuilder::new_from_edges_with_ranking(&THREE_DUMMIES, &THREE_DUMMIES_RANKS).build();
        let n_vertices = graph.node_count();
        insert_dummy_vertices(&mut graph, minimum_length, 0.0);
        // one dummy vertex
        assert_eq!(graph.node_weights().filter(|w| w.is_dummy).count(), 3);
        // one more vertex
        assert_eq!(n_vertices + 3, graph.node_count())
    }

    #[test]
    fn insert_dummy_vertices_7_dummies() {
        let (mut graph, minimum_length) =
            GraphBuilder::new_from_edges_with_ranking(&COMPLEX_EXAMPLE, &COMPLEX_EXAMPLE_RANKS)
                .build();
        let n_vertices = graph.node_count();
        insert_dummy_vertices(&mut graph, minimum_length, 0.0);
        // one dummy vertex
        assert_eq!(graph.node_weights().filter(|w| w.is_dummy).count(), 7);
        // one more vertex
        assert_eq!(n_vertices + 7, graph.node_count())
    }

    #[test]
    fn bundle_dummy_vertices_ping_graph() {
        let mut edges = [
            (1, 2),
            (1, 4),
            (1, 5),
            (1, 3),
            (2, 4),
            (2, 5),
            (3, 9),
            (3, 10),
            (3, 8),
            (4, 6),
            (4, 9),
            (4, 8),
            (5, 6),
            (5, 10),
            (5, 8),
            (6, 7),
            (7, 9),
            (7, 10),
            (8, 14),
            (8, 15),
            (8, 13),
            (9, 11),
            (9, 14),
            (9, 13),
            (10, 11),
            (10, 15),
            (10, 13),
            (11, 12),
            (12, 14),
            (12, 15),
            (13, 18),
            (13, 19),
            (13, 20),
            (14, 16),
            (14, 18),
            (14, 20),
            (15, 16),
            (15, 19),
            (15, 20),
            (16, 17),
            (17, 18),
            (17, 19),
            (18, 21),
            (19, 21),
        ];
        for e in &mut edges {
            e.0 -= 1;
            e.1 -= 1;
        }
        let g = StableDiGraph::from_edges(&edges);
        let c = Config::default();
        crate::algorithm::start(g, &c);
    }
}

mod init_order {
    use crate::algorithm::p2_reduce_crossings::insert_dummy_vertices;

    use super::{
        GraphBuilder, COMPLEX_EXAMPLE, COMPLEX_EXAMPLE_RANKS, ONE_DUMMY, ONE_DUMMY_RANKS,
        THREE_DUMMIES, THREE_DUMMIES_RANKS,
    };

    #[test]
    fn all_neighbors_must_be_at_adjacent_level_one_dummy() {
        let (mut graph, minimum_length) =
            GraphBuilder::new_from_edges_with_ranking(&ONE_DUMMY, &ONE_DUMMY_RANKS).build();
        insert_dummy_vertices(&mut graph, minimum_length, 0.0);
        for v in graph.node_indices() {
            let rank = graph[v].rank;
            for n in graph.neighbors_undirected(v) {
                assert_eq!(rank.abs_diff(graph[n].rank), 1);
            }
        }
    }

    #[test]
    fn all_neighbors_must_be_at_adjacent_level_three_dummies() {
        let (mut graph, minimum_length) =
            GraphBuilder::new_from_edges_with_ranking(&THREE_DUMMIES, &THREE_DUMMIES_RANKS).build();
        insert_dummy_vertices(&mut graph, minimum_length, 0.0);
        for v in graph.node_indices() {
            let rank = graph[v].rank;
            for n in graph.neighbors_undirected(v) {
                assert_eq!(rank.abs_diff(graph[n].rank), 1);
            }
        }
    }

    #[test]
    fn all_neighbors_must_be_at_adjacent_level_seven_dummies() {
        let (mut graph, minimum_length) =
            GraphBuilder::new_from_edges_with_ranking(&COMPLEX_EXAMPLE, &COMPLEX_EXAMPLE_RANKS)
                .build();

        insert_dummy_vertices(&mut graph, minimum_length, 0.0);
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
    use crate::algorithm::{p2::order_layer, p2_reduce_crossings::Order, Edge, Vertex};
    use petgraph::stable_graph::StableDiGraph;

    /// Shorthand for creating a default vertex with a specified rank.
    fn vertex_with_rank(rank: i32) -> Vertex {
        Vertex {
            rank,
            ..Default::default()
        }
    }

    #[test]
    fn two_crossings() {
        let mut graph = StableDiGraph::new();
        let n0 = graph.add_node(vertex_with_rank(0));
        let n1 = graph.add_node(vertex_with_rank(0));
        let n2 = graph.add_node(vertex_with_rank(0));
        let s0 = graph.add_node(vertex_with_rank(1));
        let s1 = graph.add_node(vertex_with_rank(1));

        graph.add_edge(n0, s1, Edge::default());
        graph.add_edge(n1, s0, Edge::default());
        graph.add_edge(n2, s0, Edge::default());

        let order = Order::new(vec![vec![n0, n1, n2], vec![s0, s1]]);
        assert_eq!(order.bilayer_cross_count(&graph, 0), 2);
    }

    #[test]
    fn four_crossings() {
        let mut graph = StableDiGraph::new();
        let n0 = graph.add_node(vertex_with_rank(0));
        let n1 = graph.add_node(vertex_with_rank(0));
        let n2 = graph.add_node(vertex_with_rank(0));
        let n3 = graph.add_node(vertex_with_rank(0));
        let s0 = graph.add_node(vertex_with_rank(1));
        let s1 = graph.add_node(vertex_with_rank(1));
        let s2 = graph.add_node(vertex_with_rank(1));
        let s3 = graph.add_node(vertex_with_rank(1));

        graph.add_edge(n0, s3, Edge::default());
        graph.add_edge(n1, s2, Edge::default());
        graph.add_edge(n2, s1, Edge::default());
        graph.add_edge(n3, s0, Edge::default());

        let order = Order::new(vec![vec![n0, n1, n2, n3], vec![s0, s1, s2, s3]]);

        assert_eq!(order.bilayer_cross_count(&graph, 0), 6);
    }

    #[test]
    fn twelve_crossings() {
        let mut g = StableDiGraph::<Vertex, Edge>::new();
        let n0 = g.add_node(vertex_with_rank(0));
        let n1 = g.add_node(vertex_with_rank(0));
        let n2 = g.add_node(vertex_with_rank(0));
        let n3 = g.add_node(vertex_with_rank(0));
        let n4 = g.add_node(vertex_with_rank(0));
        let n5 = g.add_node(vertex_with_rank(0));
        let s0 = g.add_node(vertex_with_rank(1));
        let s1 = g.add_node(vertex_with_rank(1));
        let s2 = g.add_node(vertex_with_rank(1));
        let s3 = g.add_node(vertex_with_rank(1));
        let s4 = g.add_node(vertex_with_rank(1));

        g.add_edge(n0, s0, Edge::default());
        g.add_edge(n1, s1, Edge::default());
        g.add_edge(n1, s2, Edge::default());
        g.add_edge(n2, s0, Edge::default());
        g.add_edge(n2, s3, Edge::default());
        g.add_edge(n2, s4, Edge::default());
        g.add_edge(n3, s0, Edge::default());
        g.add_edge(n3, s3, Edge::default());
        g.add_edge(n4, s3, Edge::default());
        g.add_edge(n5, s2, Edge::default());
        g.add_edge(n5, s4, Edge::default());

        let order = Order::new(vec![vec![n0, n1, n2, n3, n4, n5], vec![s0, s1, s2, s3, s4]]);
        assert_eq!(order.crossings(&g), 12);
    }

    #[test]
    fn test_barycenter() {
        let mut graph = StableDiGraph::new();
        let n0 = graph.add_node(vertex_with_rank(0)); // 33
        let n1 = graph.add_node(vertex_with_rank(0)); // 28
        let n2 = graph.add_node(vertex_with_rank(0)); // 6
        let n3 = graph.add_node(vertex_with_rank(0)); // 42
        let n4 = graph.add_node(vertex_with_rank(0)); // 31
        let n5 = graph.add_node(vertex_with_rank(0)); // 25
        let n6 = graph.add_node(vertex_with_rank(0)); // 38
        let n7 = graph.add_node(vertex_with_rank(0)); // 34
        let s0 = graph.add_node(vertex_with_rank(1)); // 9
        let s1 = graph.add_node(vertex_with_rank(1)); // 43
        let s2 = graph.add_node(vertex_with_rank(1)); // 8
        let s3 = graph.add_node(vertex_with_rank(1)); // 39
        let s4 = graph.add_node(vertex_with_rank(1)); // 35

        graph.add_edge(n0, s0, Edge::default());
        graph.add_edge(n1, s0, Edge::default());
        graph.add_edge(n2, s0, Edge::default());
        graph.add_edge(n2, s2, Edge::default());
        graph.add_edge(n3, s1, Edge::default());
        graph.add_edge(n4, s2, Edge::default());
        graph.add_edge(n5, s2, Edge::default());
        graph.add_edge(n6, s3, Edge::default());
        graph.add_edge(n7, s4, Edge::default());

        let _inner = vec![
            vec![n0, n2, n4, n3, n6, n7, n1, n5],
            vec![s0, s1, s2, s3, s4],
        ];
        let order = Order::new(_inner);
        let expected_order = order_layer(
            &graph,
            false,
            &order,
            crate::algorithm::p2_reduce_crossings::barycenter,
        );
        assert_eq!(
            expected_order._inner[0],
            vec![n0, n1, n2, n3, n4, n5, n6, n7]
        );
    }
}
