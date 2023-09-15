#[cfg(test)]
mod tests;

use std::collections::HashMap;

use petgraph::Direction::Incoming;
use petgraph::stable_graph::{StableDiGraph, NodeIndex};
use petgraph::visit::EdgeRef;

use super::{Vertex, Edge};
/// Reprents a Layered Graph, in which the number of crossings of edges between
/// Vertices has been minimized. This implies that the order of vertices will not change
/// in the following steps of the algorithm.
/// 
/// It's then used to mark all type 1 conflicts (a crossing between an inner segment and a non-inner segment)
// #[derive(Clone, Copy)]
// pub struct Vertex {
//     pub(crate) id: usize,
//     rank: usize,
//     pos: usize,
//     is_dummy: bool,
//     root: NodeIndex,
//     align: NodeIndex,
//     shift: isize,
//     sink: NodeIndex,
// }

// impl Default for Vertex {
//     fn default() -> Self {
//         Self {
//             id: 0,
//             rank: usize::default(),
//             pos: usize::default(),
//             is_dummy: false,
//             root: 0.into(),
//             align: 0.into(),
//             shift: isize::MAX,
//             sink: 0.into(),
//         }
//     }
// }

// impl Vertex {
//     pub fn new(id: usize, align_root_sink: NodeIndex, rank: usize, pos: usize, is_dummy: bool) -> Self {
//         Self {
//             id,
//             rank,
//             pos,
//             is_dummy,      
//             root: align_root_sink,
//             align: align_root_sink,
//             shift: isize::MAX,
//             sink: align_root_sink,
//         }
//     }
// }

// #[derive(Clone, Copy)]
// pub struct Edge {
//     has_type_1_conflict: bool
// }

// impl Edge {
//     pub(crate) fn new() -> Self {
//         Self { has_type_1_conflict: false }
//     }
// }

// impl Default for Edge {
//     fn default() -> Self {
//         Self {
//             has_type_1_conflict: false,
//         }
//     }
// }

pub(super) fn calculate_x_coordinates(graph: &mut StableDiGraph<Vertex, Edge>, layers: &mut [Vec<NodeIndex>], vertex_spacing: usize) -> Vec<HashMap<NodeIndex, isize>> {
    let mut layouts = Vec::new();
    mark_type_1_conflicts(graph, layers);
    // calculate the coordinates for each direction
    for _ in [VDir::Down, VDir::Up] {
        for h_dir in [HDir::Right, HDir::Left] {
            create_vertical_alignments(graph, layers);
            let mut layout = do_horizontal_compaction(graph, layers, vertex_spacing);

            // flip x_coordinates if we went from right to left
            if let HDir::Left = h_dir {
                layout.values_mut().for_each(|x| *x = -*x);
            }
            layouts.push(layout);

            // rotate the graph
            for row in layers.iter_mut() {
                row.reverse();
            }
            // reset root, align and sink values
        }
        // rotate the graph
        graph.reverse();
        layers.reverse();
    }
    layouts
}

pub(crate) fn calculate_y_coordinates(layers: &[Vec<NodeIndex>], vertex_spacing: usize) -> HashMap<NodeIndex, isize> {
    layers.iter()
        .enumerate()
        .map(|(rank, row)| row.iter().map(move |v| (*v, rank as isize * vertex_spacing as isize * -1)))
        .flatten()
        .collect::<HashMap<NodeIndex, isize>>() 
}

pub(crate) fn align_to_smallest_width_layout(aligned_layouts: &mut[HashMap<NodeIndex, isize>]) {
    // determine minimum and maximum coordinate of each layout, plus the width
    let min_max: Vec<(isize, isize, isize)> = aligned_layouts.iter()
                                                 .map(|c| {
                                                    let min = *c.values().min().unwrap();
                                                    let max = *c.values().max().unwrap();
                                                    (min, max, max - min)
                                                 }).collect();

    // determine the layout with the minimum width
    let min_width = min_max.iter().enumerate().min_by(|a, b| a.1.2.cmp(&b.1.2)).unwrap().0;

    // align all other layouts to the lowest/highest coordinate of the layout with the minimum width, 
    // depending on the horizontal direction which was chosen to create them
    for (i, layout) in aligned_layouts.iter_mut().enumerate() {
        // if i % 2 == 0, then horizontal direction was left
        let shift = if i % 2 == 0 { 
            min_max[i].0 as isize - min_max[min_width].0 as isize
        } else { 
            min_max[min_width].1  as isize - min_max[i].1 as isize 
        };
        for v in layout.values_mut() {
            let new = *v as isize + shift;
            *v = new as isize;
        }
    }
}

pub(crate) fn set_to_average_median(aligned_layouts: Vec<HashMap<NodeIndex, isize>>) -> Vec<(NodeIndex, isize)> {
    // sort all 4 coordinates per vertex in ascending order
    let mut sorted_layouts = HashMap::new();
    for k in aligned_layouts.get(0).unwrap().keys() {
        let mut vertex_coordinates = [
            *aligned_layouts.get(0).unwrap().get(k).unwrap(),
            *aligned_layouts.get(1).unwrap().get(k).unwrap(),
            *aligned_layouts.get(2).unwrap().get(k).unwrap(),
            *aligned_layouts.get(3).unwrap().get(k).unwrap(),
        ];
        vertex_coordinates.sort();
        sorted_layouts.insert(k, vertex_coordinates);
    }

    // create final layout, by averaging the two median values
    sorted_layouts.into_iter()
                                         .map(|(k, v)| (*k, (v[1] + v[2]) / 2))
                                         .collect::<Vec<_>>()
}

fn is_incident_to_inner_segment(graph: &StableDiGraph<Vertex, Edge>, id: NodeIndex) -> bool {
    graph[id].is_dummy &&
    graph.neighbors_directed(id, Incoming).into_iter().any(|n| graph[n].is_dummy)
}

/// Assumes id is incident to inner segment 
fn get_inner_segment_upper_neighbor(graph: &StableDiGraph<Vertex, Edge>, id: NodeIndex) -> Option<NodeIndex> {
    if is_incident_to_inner_segment(graph, id) {
        graph.neighbors_directed(id, Incoming).next()
    } else {
        None
    }
}

fn mark_type_1_conflicts(graph: &mut StableDiGraph<Vertex, Edge>, layers: &[Vec<NodeIndex>]) {
    for (level, next_level) in layers [..layers.len() - 1]
                                    .iter()
                                    .zip(layers[1..].iter()) 
    {
        let mut left_dummy_index = 0;
        let mut l = 0;
        for (l_1, dummy_candidate) in next_level.iter().enumerate() {
            let right_dummy_index = match get_inner_segment_upper_neighbor(graph, *dummy_candidate) {
                Some(id) => graph[id].pos,
                None => if l_1 == next_level.len()  - 1 { 
                    level.len() 
                } else { 
                    continue;
                }
            };
            while l < l_1 {
                let vertex = next_level[l];
                let mut upper_neighbors = graph.neighbors_directed(vertex, Incoming).collect::<Vec<_>>();
                upper_neighbors.sort_by(|a, b| graph[*a].pos.cmp(&graph[*b].pos));
                for upper_neighbor in upper_neighbors {
                    let vertex_index = graph[upper_neighbor].pos;
                    if vertex_index < left_dummy_index || vertex_index > right_dummy_index {
                        let edge = graph.find_edge(upper_neighbor, vertex).unwrap();
                        graph[edge].has_type_1_conflict = true;
                    }
                }
                l = l + 1;
            }
            left_dummy_index = right_dummy_index;
        }
    }
}

pub(super) fn init_for_alignment(graph: &mut StableDiGraph<Vertex, Edge>, layers: &[Vec<NodeIndex>]) {
    for (rank, row) in layers.iter().enumerate() {
        for (pos, v) in row.iter().enumerate() {
            let weight: &mut Vertex = &mut graph[*v]; 
            weight.rank = rank as i32;
            weight.pos = pos;
            weight.shift = isize::MAX;
            weight.align = *v;
            weight.root = *v;
            weight.sink = *v;
        }
    }
}

// TODO: Change this so the graph gets rotated outside of the function
/// Aligns the graph in so called blocks, which are used in the next step 
/// to determine the x-coordinate of a vertex.
fn create_vertical_alignments(graph: &mut StableDiGraph<Vertex, Edge>, layers: &mut [Vec<NodeIndex>]) {
    init_for_alignment(graph, layers);
    for i in 0..layers.len() {
        let mut r = -1;

        for k in 0..layers[i].len() {
            let v = layers[i][k];
            let mut edges = graph.edges_directed(v, Incoming).map(|e| (e.id(), e.source())).collect::<Vec<_>>();
            if edges.len() == 0 {
                continue;
            }
            edges.sort_by(|e1, e2| graph[e1.1].pos.cmp(&graph[e2.1].pos));

            let d = (edges.len() as f64 + 1.) / 2. - 1.; // need to subtract one because indices are zero based
            let lower_upper_median = [d.floor() as usize, d.ceil() as usize];

            for m in lower_upper_median  {
                if graph[v].align == v {
                    let edge_id = edges[m].0;
                    let median_neighbor = edges[m].1;
                    if !graph[edge_id].has_type_1_conflict && r < graph[median_neighbor].pos as isize {
                        graph[median_neighbor].align = v;
                        graph[v].root = graph[median_neighbor].root;
                        graph[v].align = graph[v].root;
                        r = graph[median_neighbor].pos as isize;
                    }
                }
            }
        }
    }
}

fn do_horizontal_compaction(graph: &mut StableDiGraph<Vertex, Edge>, layers: &[Vec<NodeIndex>], vertex_spacing: usize) -> HashMap<NodeIndex, isize> {
    let mut x_coordinates = place_blocks(graph, layers, vertex_spacing as isize);
    // calculate class shifts 
    for i in 0..layers.len() { 
        let mut v = layers[i][0];
        if graph[v].sink == v {
            if graph[graph[v].sink].shift == isize::MAX {
                let v_sink = graph[v].sink;
                graph[v_sink].shift = 0;
            }
            let mut j = i; // level index
            let mut k = 0; // vertex in level index
            loop {
                v = layers[j][k];

                // traverse one block
                while graph[v].align != graph[v].root {
                    v = graph[v].align;
                    j += 1;

                    if graph[v].pos > 1 {
                        let u = pred(graph[v], layers);
                        let distance_v_u = *x_coordinates.get(&v).unwrap() - (*x_coordinates.get(&u).unwrap() + vertex_spacing as isize);
                        let u_sink = graph[u].sink;
                        graph[u_sink].shift = graph[graph[u].sink].shift.min(graph[graph[v].sink].shift + distance_v_u);
                    }
                }
                k = graph[v].pos + 1;

                if k == layers[j].len() || graph[v].sink != graph[layers[j][k]].sink {
                    break;
                }
            }
        }   
    }

    // calculate absolute x-coordinates
    for v in graph.node_indices() {
        x_coordinates.insert(v, *x_coordinates.get(&v).unwrap() + graph[graph[v].sink].shift);
    }
    x_coordinates
}

fn place_blocks(graph: &mut StableDiGraph<Vertex, Edge>, layers: &[Vec<NodeIndex>], vertex_spacing: isize) -> HashMap<NodeIndex, isize> {
    let mut x_coordinates = HashMap::new();
    // place blocks
    for root in graph.node_indices().filter(|v| graph[*v].root == *v).collect::<Vec<_>>() {
        place_block(graph, layers, root, &mut x_coordinates, vertex_spacing);
    }
    x_coordinates
}
fn place_block(
    graph: &mut StableDiGraph<Vertex, Edge>,
    layers: &[Vec<NodeIndex>],
    root: NodeIndex, 
    x_coordinates: &mut HashMap<NodeIndex, isize>, 
    vertex_spacing: isize
) {
    if x_coordinates.get(&root).is_some() {
        return;
    }
    x_coordinates.insert(root, 0);
    let mut w = root;
    loop {
        if graph[w].pos > 0 {
            let u = graph[pred(graph[w], layers)].root;
            place_block(graph, layers, u, x_coordinates, vertex_spacing);
            // initialize sink of current node to have the same sink as the root
            if graph[root].sink == root { 
                graph[root].sink = graph[u].sink; 
            }
            if graph[root].sink == graph[u].sink {
                    x_coordinates.insert(root, *x_coordinates.get(&root).unwrap().max(&(x_coordinates.get(&u).unwrap() + vertex_spacing)));

                }
            }
            w = graph[w].align;
            if w == root {
                break
            }
    }
    // align all other vertices in this block to the x-coordinate of the root
    while graph[w].align != root {
        w = graph[w].align;
        x_coordinates.insert(w, *x_coordinates.get(&root).unwrap());
        graph[w].sink = graph[root].sink;
    }
}

fn pred(vertex: Vertex, layers: &[Vec<NodeIndex>]) -> NodeIndex {
    layers[vertex.rank as usize][vertex.pos - 1]
}
/// Represents a layered graph whose vertices have been aligned in blocks.
/// A root is the highest node in a block, depending on the direction.
/// 
/// It is used to determine classes of a block, calculate the x-coordinates of a block
/// in regard to its class and shift classes together as close as possible.

/// Represents the horizontal direction in which the algorithm is run
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum HDir {
    Left,
    Right,
}

/// Represents the vertical direction in which the algorithm is run
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum VDir {
    Up,
    Down,
}