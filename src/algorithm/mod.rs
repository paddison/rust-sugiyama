use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableDiGraph};

use crate::{util::into_weakly_connected_components, Layout, Layouts};
use crate::{Config, LayeringType};
use p1_layering as p1;
use p2_reduce_crossings as p2;
use p3_calculate_coordinates as p3;

use self::p1::ranking::move_vertices_up;

mod p1_layering;
mod p2_reduce_crossings;
mod p3_calculate_coordinates;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(super) struct Vertex {
    id: usize,
    rank: i32,
    pos: usize,
    low: u32,
    lim: u32,
    parent: Option<NodeIndex>,
    is_tree_vertex: bool,
    is_dummy: bool,
    root: NodeIndex,
    align: NodeIndex,
    shift: isize,
    sink: NodeIndex,
}

impl Vertex {
    pub(super) fn new(id: usize) -> Self {
        let mut v = Self::default();
        v.id = id;
        v
    }

    #[cfg(test)]
    fn new_test_p1(low: u32, lim: u32, parent: Option<NodeIndex>, is_tree_vertex: bool) -> Self {
        Self {
            id: 0,
            rank: 0,
            pos: 0,
            low,
            lim,
            parent,
            is_tree_vertex,
            is_dummy: false,
            root: 0.into(),
            align: 0.into(),
            shift: isize::MAX,
            sink: 0.into(),
        }
    }

    #[cfg(test)]
    pub fn new_test_p3(align_root_sink: NodeIndex, rank: i32, pos: usize, is_dummy: bool) -> Self {
        Self {
            id: 0,
            rank,
            pos,
            low: 0,
            lim: 0,
            parent: None,
            is_tree_vertex: false,
            is_dummy,
            root: align_root_sink,
            align: align_root_sink,
            shift: isize::MAX,
            sink: align_root_sink,
        }
    }

    #[cfg(test)]
    pub fn new_with_rank(rank: i32) -> Self {
        let mut v = Self::default();
        v.rank = rank;
        v
    }
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            id: 0,
            rank: 0,
            pos: 0,
            low: 0,
            lim: 0,
            parent: None,
            is_tree_vertex: false,
            is_dummy: false,
            root: 0.into(),
            align: 0.into(),
            shift: isize::MAX,
            sink: 0.into(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Edge {
    weight: i32,
    cut_value: Option<i32>,
    is_tree_edge: bool,
    has_type_1_conflict: bool,
}

impl Default for Edge {
    fn default() -> Self {
        Self {
            weight: 1,
            cut_value: None,
            is_tree_edge: false,
            has_type_1_conflict: false,
        }
    }
}

pub(super) fn build_layout_from_edges(edges: &[(u32, u32)], config: Config) -> Layouts<usize> {
    let graph = StableDiGraph::<Vertex, Edge>::from_edges(edges);
    // initialize vertex ids to NodeIndex
    start(graph, config)
}

pub(super) fn build_layout_from_graph<T, E>(
    graph: &StableDiGraph<T, E>,
    config: Config,
) -> Layouts<usize> {
    // does this guarantee that ids will match?
    let algo_graph = graph.map(|_, _| Vertex::default(), |_, _| Edge::default());
    start(algo_graph, config)
}

pub(super) fn start(mut graph: StableDiGraph<Vertex, Edge>, config: Config) -> Layouts<usize> {
    init_graph(&mut graph);
    into_weakly_connected_components(graph)
        .into_iter()
        .map(|g| build_layout(g, config))
        .collect()
}

pub(super) fn map_input_graph<V, E>(graph: &StableDiGraph<V, E>) -> StableDiGraph<Vertex, Edge> {
    graph.map(|_, _| Vertex::default(), |_, _| Edge::default())
}

fn init_graph(graph: &mut StableDiGraph<Vertex, Edge>) {
    for id in graph.node_indices().collect::<Vec<_>>() {
        graph[id].id = id.index();
        graph[id].root = id;
        graph[id].align = id;
        graph[id].sink = id;
    }
}

fn build_layout(mut graph: StableDiGraph<Vertex, Edge>, config: Config) -> Layout {
    execute_phase_1(
        &mut graph,
        config.minimum_length as i32,
        config.layering_type,
    );
    let layers = execute_phase_2(
        &mut graph,
        config.minimum_length as i32,
        config.no_dummy_vertices,
    );
    execute_phase_3(&mut graph, layers, config.vertex_spacing)
}

fn execute_phase_1(
    graph: &mut StableDiGraph<Vertex, Edge>,
    minimum_length: i32,
    layering_type: LayeringType,
) {
    p1::ranking::init_rank(graph, minimum_length);
    match layering_type {
        LayeringType::Up => move_vertices_up(graph, minimum_length),
        _ => p1::rank(graph, minimum_length),
    }
    //p1::rank(graph, minimum_length);
}

fn execute_phase_2(
    graph: &mut StableDiGraph<Vertex, Edge>,
    minimum_length: i32,
    no_dummy_vertices: bool,
) -> Vec<Vec<NodeIndex>> {
    // build layer to test them
    let mut test_layers =
        vec![vec![]; graph.node_weights().map(|w| w.rank as usize).max().unwrap() + 1];
    for v in graph.node_indices() {
        test_layers[graph[v].rank as usize].push(v);
    }
    for l in test_layers {
        //println!("{l:?}");
    }
    p2::insert_dummy_vertices(graph, minimum_length);
    //p2::bundle_dummy_vertices(graph);
    let mut order = p2::ordering(graph);
    if no_dummy_vertices {
        p2::remove_dummy_vertices(graph, &mut order);
    }
    order
}

/// calculate the final coordinates for each vertex, after the graph was layered and crossings where minimized.
fn execute_phase_3(
    graph: &mut StableDiGraph<Vertex, Edge>,
    mut layers: Vec<Vec<NodeIndex>>,
    vertex_spacing: usize,
) -> Layout {
    for n in graph.node_indices().collect::<Vec<_>>() {
        if graph[n].is_dummy {
            graph[n].id = n.index();
        }
    }
    for l in &layers {
        let id = l.iter().map(|v| graph[*v].id).collect::<Vec<_>>();
        println!("{id:?}");
    }
    let width = layers.iter().map(|l| l.len()).max().unwrap_or(0);
    let height = layers.len();
    let mut layouts = p3::create_layouts(graph, &mut layers, vertex_spacing);

    p3::align_to_smallest_width_layout(&mut layouts);
    let mut x_coordinates = p3::calculate_relative_coords(layouts);
    // determine the smallest x-coordinate
    let min = x_coordinates.iter().min_by(|a, b| a.1.cmp(&b.1)).unwrap().1;

    // shift all coordinates so the minimum coordinate is 0
    for (_, c) in &mut x_coordinates {
        *c -= min;
    }

    // format to NodeIndex: (x, y), width, height
    (
        x_coordinates
            .into_iter()
            .filter(|(v, _)| !graph[*v].is_dummy)
            // calculate y coordinate
            .map(|(v, x)| {
                (
                    graph[v].id,
                    (x, graph[v].rank as isize * vertex_spacing as isize * -1),
                )
            })
            .collect::<Vec<_>>(),
        width,
        height,
    )
}

fn slack(graph: &StableDiGraph<Vertex, Edge>, edge: EdgeIndex, minimum_length: i32) -> i32 {
    let (tail, head) = graph.edge_endpoints(edge).unwrap();
    graph[head].rank - graph[tail].rank - minimum_length
}
