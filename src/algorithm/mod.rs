use petgraph::stable_graph::{NodeIndex, StableDiGraph, EdgeIndex};

use p1_layering as p1;
use p2_reduce_crossings as p2;
use p3_calculate_coordinates as p3;
use crate::{util::into_weakly_connected_components, Layouts, Layout};
use crate::Config;

mod p1_layering;
mod p2_reduce_crossings;
mod p3_calculate_coordinates;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct Vertex {
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

#[derive(Clone, Copy)]
struct Edge {
    weight: i32,
    cut_value: Option<i32>,
    is_tree_edge: bool,
    has_type_1_conflict: bool
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

pub(super) fn build_layout_from_graph<T, E>(graph: &StableDiGraph<T, E>, config: Config) -> Layouts<usize> {
    // does this guarantee that ids will match?
    let algo_graph = graph.map(|_, _,| Vertex::default() , |_, _| Edge::default());
    start(algo_graph, config)
}

fn start(mut graph: StableDiGraph<Vertex, Edge>, config: Config) -> Layouts<usize> {
    init_graph(&mut graph);
    into_weakly_connected_components(graph).into_iter().map(|g| build_layout(g, config)).collect()
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
    execute_phase_1(&mut graph, config.minimum_length as i32);
    let layers = execute_phase_2(&mut graph, config.minimum_length as i32);
    execute_phase_3(&mut graph, layers, config.vertex_spacing)
}

fn execute_phase_1(graph: &mut StableDiGraph<Vertex, Edge>, minimum_length: i32) {
    p1::rank(graph, minimum_length);
}

fn execute_phase_2(graph: &mut StableDiGraph<Vertex, Edge>, minimum_length: i32) -> Vec<Vec<NodeIndex>> {
    p2::insert_dummy_vertices(graph, minimum_length);
    p2::ordering(graph)
}

/// calculate the final coordinates for each vertex, after the graph was layered and crossings where minimized.
fn execute_phase_3(graph: &mut StableDiGraph<Vertex, Edge>, mut layers: Vec<Vec<NodeIndex>>, vertex_spacing: usize) -> Layout {
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
        x_coordinates.into_iter()
            .filter(|(v, _)| !graph[*v].is_dummy )
            // calculate y coordinate
            .map(|(v, x)| (graph[v].id, (x, graph[v].rank as isize * vertex_spacing as isize * -1))) 
            .collect::<Vec<_>>(),
        width,
        height
    )
}

fn slack(graph: &StableDiGraph<Vertex, Edge>, edge: EdgeIndex, minimum_length: i32) -> i32 {
    let (tail, head) = graph.edge_endpoints(edge).unwrap();
    graph[head].rank - graph[tail].rank - minimum_length 
}