use std::collections::HashMap;

use algorithm::{Edge, Vertex};

use configure::Config;
use log::info;
use petgraph::{graph::NodeIndex, stable_graph::StableDiGraph};

mod algorithm;
pub mod configure;
mod util;

type Layout = (Vec<(usize, (f64, f64))>, f64, f64);
type Layouts<T> = Vec<(Vec<(T, (f64, f64))>, f64, f64)>;

/// Creates a graph layout from edges, which are given as a `&[(u32, u32)]`.
///
/// The layouts are returned as a list of disjoint subgraphs containing the
/// subgraph layout, the width, and the height. The layout of a subgraph is a
/// list of the vertex number (as specified in the edges) and its x and y
/// position respectively.
pub fn from_edges(edges: &[(u32, u32)], config: &Config) -> Layouts<usize> {
    info!(target: "initializing", "Creating new layout from edges, containing {} edges", edges.len());
    let graph = StableDiGraph::from_edges(edges);
    algorithm::start(graph, config)
}

/// Creates a graph layout from a preexisting [StableDiGraph<V, E>].
///
/// The layouts are returned as a list of disjoint subgraphs containing the
/// subgraph layout, the width, and the height. The layout of a subgraph is a
/// list of the [NodeIndex] and its x and y position respectively.
pub fn from_graph<V, E>(
    graph: &StableDiGraph<V, E>,
    vertex_size: &impl Fn(NodeIndex, &V) -> (f64, f64),
    config: &Config,
) -> Layouts<NodeIndex> {
    info!(target: "initializing", 
        "Creating new layout from existing graph, containing {} vertices and {} edges.", 
        graph.node_count(), 
        graph.edge_count());

    let graph = graph.map(
        |id, v| Vertex::new(id.index(), vertex_size(id, v)),
        |_, _| Edge::default(),
    );

    algorithm::start(graph, config)
        .into_iter()
        .map(|(l, w, h)| {
            (
                l.into_iter()
                    .map(|(id, coords)| (NodeIndex::from(id as u32), coords))
                    .collect(),
                w,
                h,
            )
        })
        .collect()
}

/// Creates a graph layout from `&[(u32, (f64, f64))]` (vertices as vertex id
/// and vertex size) and `&[(u32, u32)]` (edges).
///
/// The layouts are returned as a list of disjoint subgraphs containing the
/// subgraph layout, the width, and the height. The layout of a subgraph is a
/// list of the vertex number and its x and y position respectively.
///
/// # Panics
///
/// Panics if `edges` contain vertices which are not contained in `vertices`
pub fn from_vertices_and_edges<'a>(
    vertices: &'a [(u32, (f64, f64))],
    edges: &'a [(u32, u32)],
    config: &Config,
) -> Layouts<usize> {
    info!(target: "initializing", 
        "Creating new layout from existing graph, containing {} vertices and {} edges.", 
        vertices.len(), 
        edges.len());

    let mut graph = StableDiGraph::new();
    let mut id_map = HashMap::new();
    for &(v, size) in vertices {
        let id = graph.add_node(Vertex::new(v as usize, size));
        id_map.insert(v, id);
    }

    for (tail, head) in edges {
        graph.add_edge(
            *id_map.get(tail).unwrap(),
            *id_map.get(head).unwrap(),
            Edge::default(),
        );
    }

    algorithm::start(graph, config)
}

#[test]
fn run_algo_empty_graph() {
    let edges = [];
    let g = from_edges(&edges, &Config::default());
    assert!(g.is_empty());
}

#[cfg(test)]
mod benchmark {
    use crate::configure::Config;

    use super::from_edges;

    #[test]
    fn r_100() {
        let edges = graph_generator::RandomLayout::new(100)
            .build_edges()
            .into_iter()
            .map(|(r, l)| (r as u32, l as u32))
            .collect::<Vec<(u32, u32)>>();
        let start = std::time::Instant::now();
        let _ = from_edges(&edges, &Config::default());
        println!("Random 100 edges: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn r_1000() {
        let edges = graph_generator::RandomLayout::new(1000)
            .build_edges()
            .into_iter()
            .map(|(r, l)| (r as u32, l as u32))
            .collect::<Vec<(u32, u32)>>();
        let start = std::time::Instant::now();
        let _ = from_edges(&edges, &Config::default());
        println!("Random 1000 edges: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn r_2000() {
        let edges = graph_generator::RandomLayout::new(2000).build_edges();
        let start = std::time::Instant::now();
        let _ = from_edges(&edges, &Config::default());
        println!("Random 2000 edges: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn r_4000() {
        let edges = graph_generator::RandomLayout::new(4000).build_edges();
        let start = std::time::Instant::now();
        let _ = from_edges(&edges, &Config::default());
        println!("Random 4000 edges: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn l_1000_2() {
        let n = 1000;
        let e = 2;
        let edges = graph_generator::GraphLayout::new_from_num_nodes(n, e).build_edges();
        let start = std::time::Instant::now();
        let _ = from_edges(&edges, &Config::default());
        println!(
            "{n} nodes, {e} edges per node: {}ms",
            start.elapsed().as_millis()
        );
    }

    #[test]
    fn l_2000_2() {
        let n = 2000;
        let e = 2;
        let edges = graph_generator::GraphLayout::new_from_num_nodes(n, e).build_edges();
        let start = std::time::Instant::now();
        let _ = from_edges(&edges, &Config::default());
        println!(
            "{n} nodes, {e} edges per node: {}ms",
            start.elapsed().as_millis()
        );
    }

    #[test]
    fn l_4000_2() {
        let n = 4000;
        let e = 2;
        let edges = graph_generator::GraphLayout::new_from_num_nodes(n, e).build_edges();
        let start = std::time::Instant::now();
        let _ = from_edges(&edges, &Config::default());
        println!(
            "{n} nodes, {e} edges per node: {}ms",
            start.elapsed().as_millis()
        );
    }

    #[test]
    fn l_8000_2() {
        let n = 8000;
        let e = 2;
        let edges = graph_generator::GraphLayout::new_from_num_nodes(n, e).build_edges();
        let start = std::time::Instant::now();
        let _ = from_edges(&edges, &Config::default());
        println!(
            "{n} nodes, {e} edges per node: {}ms",
            start.elapsed().as_millis()
        );
    }
}

#[cfg(test)]
mod check_visuals {

    use crate::{
        configure::{Config, RankingType},
        from_vertices_and_edges,
    };

    use super::from_edges;

    #[test]
    fn test_no_dummies() {
        let vertices = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        ];
        let edges = [
            (1, 2),
            (1, 3),
            (2, 5),
            (2, 16),
            (4, 5),
            (4, 6),
            (4, 7),
            (6, 17),
            (6, 3),
            (6, 18),
            (8, 3),
            (8, 9),
            (8, 10),
            (9, 16),
            (9, 7),
            (9, 19),
            (11, 7),
            (11, 12),
            (11, 13),
            (12, 18),
            (12, 10),
            (12, 20),
            (14, 10),
            (14, 15),
            (15, 19),
            (15, 13),
        ];
        let _ = from_vertices_and_edges(
            &vertices
                .into_iter()
                .map(|v| (v, (0.0, 0.0)))
                .collect::<Vec<_>>(),
            &edges,
            &Config {
                dummy_vertices: true,
                ..Default::default()
            },
        );
    }
    #[test]
    fn verify_looks_good() {
        // NOTE: This test might fail eventually, since the order of lements in a row canot be guaranteed;
        let edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (2, 4),
            (3, 5),
            (3, 6),
            (3, 7),
            (3, 8),
            (4, 5),
            (4, 6),
            (4, 7),
            (4, 8),
            (5, 9),
            (6, 9),
            (7, 9),
            (8, 9),
        ];
        let (layout, width, height) = &mut from_edges(&edges, &Config::default())[0];
        layout.sort_by(|a, b| a.0.cmp(&b.0));

        assert_eq!(*width, 4.0);
        assert_eq!(*height, 6.0);
        println!("{:?}", layout);
    }

    #[test]
    fn root_vertices_on_top_disabled() {
        let edges = [(1, 0), (2, 1), (3, 0), (4, 0)];
        let layout = from_edges(&edges, &Config::default());
        for (id, (_, y)) in layout[0].0.clone() {
            if id == 2 {
                assert_eq!(y, 0.0);
            } else if id == 3 || id == 4 || id == 1 {
                assert_eq!(y, 10.0);
            } else {
                assert_eq!(y, 20.0)
            }
        }
    }

    #[test]
    fn check_coords_2() {
        let edges = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 4),
            (4, 5),
            (5, 6),
            (2, 6),
            (3, 6),
            (3, 7),
            (3, 8),
            (3, 9),
        ];
        let layout = from_edges(&edges, &Config::default());
        println!("{:?}", layout);
    }

    #[test]
    fn hlrs_ping() {
        let _nodes = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        ];
        let edges = [
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
        ]
        .into_iter()
        .map(|(t, h)| (t - 1, h - 1))
        .collect::<Vec<_>>();

        let layout = from_edges(
            &edges,
            &Config {
                ranking_type: RankingType::Up,
                ..Default::default()
            },
        );
        println!("{layout:?}");
    }

    #[test]
    fn run_algo_empty_graph() {
        use super::from_edges;
        let edges = [];
        let g = from_edges(&edges, &Config::default());
        assert!(g.is_empty());
    }

    #[test]
    fn run_algo_with_duplicate_edges() {
        let edges = [
            (1, 2),
            (2, 5),
            (2, 6),
            (2, 3),
            (3, 4),
            (4, 3),
            (4, 8),
            (8, 4),
            (8, 7),
            (3, 7),
            (6, 7),
            (7, 6),
            (5, 6),
            (5, 1),
        ];

        let layout = from_edges(&edges, &Config::default());
        println!("{layout:?}");
    }
}
