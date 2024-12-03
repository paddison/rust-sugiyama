//! The implementation roughly follows sugiyamas algorithm for creating
//! a layered graph layout.
//!
//! Usually Sugiyamas algorithm consists of 4 Phases:
//! 1. Remove Cycles
//! 2. Assign each vertex to a rank/layer
//! 3. Reorder vertices in each rank to reduce crossings
//! 4. Calculate the final coordinates.
//!
//! Currently, phase 2 to 4 are implemented, Cycle removal might be added at
//! a later time.
//!
//! The whole algorithm roughly follows the 1993 paper "A technique for drawing
//! directed graphs" by Gansner et al. It can be found
//! [here](https://ieeexplore.ieee.org/document/221135).
//!
//! See the submodules for each phase for more details on the implementation
//! and references used.
use std::collections::HashMap;

use log::{debug, info};
use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableDiGraph};

use crate::configure::{Config, CrossingMinimization, RankingType};
use crate::{util::weakly_connected_components, Layout, Layouts};
use p0_cycle_removal as p0;
use p1_layering as p1;
use p2_reduce_crossings as p2;
use p3_calculate_coordinates as p3;

use self::p3_calculate_coordinates::VDir;

mod p0_cycle_removal;
mod p1_layering;
mod p2_reduce_crossings;
mod p3_calculate_coordinates;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Vertex {
    id: usize,
    size: (f64, f64),
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
    pub(super) fn new(id: usize, size: (f64, f64)) -> Self {
        Self {
            id,
            size,
            ..Default::default()
        }
    }
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            id: 0,
            size: (0.0, 0.0),
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

pub(super) fn start(mut graph: StableDiGraph<Vertex, Edge>, config: &Config) -> Layouts<usize> {
    init_graph(&mut graph);
    weakly_connected_components(graph)
        .into_iter()
        .map(|g| build_layout(g, config))
        .collect()
}

fn init_graph(graph: &mut StableDiGraph<Vertex, Edge>) {
    info!("Initializing graphs vertex weights");
    for id in graph.node_indices().collect::<Vec<_>>() {
        graph[id].id = id.index();
        graph[id].root = id;
        graph[id].align = id;
        graph[id].sink = id;
    }
}

fn build_layout(mut graph: StableDiGraph<Vertex, Edge>, config: &Config) -> Layout {
    info!(target: "layouting", "Start building layout");
    info!(target: "layouting", "Configuration is: {:?}", config);
    // we don't remember the edges that where reversed for now, since they are
    // currently not needed
    let _ = execute_phase_0(&mut graph);

    execute_phase_1(
        &mut graph,
        config.minimum_length as i32,
        config.ranking_type,
    );

    let layers = execute_phase_2(
        &mut graph,
        config.minimum_length as i32,
        config.dummy_vertices,
        config.c_minimization,
        config.transpose,
    );

    let layout = execute_phase_3(&mut graph, layers, config.vertex_spacing, config.dummy_size);
    debug!(target: "layouting", "Coordinates: {:?}\nwidth: {}, height:{}",
        layout.0,
        layout.1,
        layout.2
    );
    layout
}

fn execute_phase_0(graph: &mut StableDiGraph<Vertex, Edge>) -> Vec<EdgeIndex> {
    info!(target: "layouting", "Executing phase 0: Cycle Removal");
    p0::remove_cycles(graph)
}

/// Assign each vertex a rank
fn execute_phase_1(
    graph: &mut StableDiGraph<Vertex, Edge>,
    minimum_length: i32,
    ranking_type: RankingType,
) {
    info!(target: "layouting", "Executing phase 1: Ranking");
    p1::rank(graph, minimum_length, ranking_type);
}

/// Reorder vertices in ranks to reduce crossings
fn execute_phase_2(
    graph: &mut StableDiGraph<Vertex, Edge>,
    minimum_length: i32,
    dummy_vertices: bool,
    crossing_minimization: CrossingMinimization,
    transpose: bool,
) -> Vec<Vec<NodeIndex>> {
    info!(target: "layouting", "Executing phase 2: Crossing Reduction");
    info!(target: "layouting",
        "Has dummy vertices: {}, heuristic for crossing minimization: {:?}, using transpose: {}",
        dummy_vertices,
        crossing_minimization,
        transpose
    );

    p2::insert_dummy_vertices(graph, minimum_length);
    let mut order = p2::ordering(graph, crossing_minimization, transpose);
    if !dummy_vertices {
        p2::remove_dummy_vertices(graph, &mut order);
    }
    order
}

/// calculate the final coordinates for each vertex, after the graph was layered and crossings where minimized.
fn execute_phase_3(
    graph: &mut StableDiGraph<Vertex, Edge>,
    mut layers: Vec<Vec<NodeIndex>>,
    vertex_spacing: usize,
    dummy_size: f64,
) -> Layout {
    info!(target: "layouting", "Executing phase 3: Coordinate Calculation");
    info!(target: "layouting", "Dummy vertices size (if enabled): {dummy_size}");
    for n in graph.node_indices().collect::<Vec<_>>() {
        if graph[n].is_dummy {
            graph[n].id = n.index();
        }
    }
    let width = layers.iter().map(|l| l.len()).max().unwrap_or(0);
    let height = layers.len();
    let mut layouts = p3::create_layouts(graph, &mut layers, vertex_spacing, dummy_size);

    p3::align_to_smallest_width_layout(&mut layouts);
    let mut x_coordinates = p3::calculate_relative_coords(layouts);
    // determine the smallest x-coordinate
    let min = x_coordinates.iter().min_by(|a, b| a.1.cmp(&b.1)).unwrap().1;

    // shift all coordinates so the minimum coordinate is 0
    for (_, c) in &mut x_coordinates {
        *c -= min;
    }

    let mut v = x_coordinates.iter().collect::<Vec<_>>();
    v.sort_by(|a, b| a.0.index().cmp(&b.0.index()));
    // format to NodeIndex: (x, y), width, height
    (
        x_coordinates
            .into_iter()
            .filter(|(v, _)| !graph[*v].is_dummy)
            // calculate y coordinate
            .map(|(v, x)| {
                (
                    graph[v].id,
                    (x, -(graph[v].rank as isize * vertex_spacing as isize)),
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

#[allow(dead_code)]
fn print_to_console(
    dir: VDir,
    graph: &StableDiGraph<Vertex, Edge>,
    layers: &[Vec<NodeIndex>],
    mut coordinates: HashMap<NodeIndex, isize>,
    vertex_spacing: usize,
) {
    let min = *coordinates.values().min().unwrap();
    let str_width = 4;
    coordinates
        .values_mut()
        .for_each(|v| *v = str_width * (*v - min) / vertex_spacing as isize);
    let width = *coordinates.values().max().unwrap() as usize;

    for line in layers {
        let mut v_line = vec!['-'; width + str_width as usize];
        let mut a_line = vec![' '; width + str_width as usize];
        for v in line {
            let pos = *coordinates.get(v).unwrap() as usize;
            if graph[*v].root != *v {
                a_line[pos] = if dir == VDir::Up { 'v' } else { '^' };
            }
            for (i, c) in v.index().to_string().chars().enumerate() {
                v_line[pos + i] = c;
            }
        }
        match dir {
            VDir::Up => {
                println!("{}", v_line.into_iter().collect::<String>());
                println!("{}", a_line.into_iter().collect::<String>());
            }
            VDir::Down => {
                println!("{}", a_line.into_iter().collect::<String>());
                println!("{}", v_line.into_iter().collect::<String>());
            }
        }
    }
    println!();
}
