use std::collections::{HashMap, HashSet};

use petgraph::{stable_graph::{NodeIndex, StableDiGraph}, graph::Node};

use crate::{phases::{p3_calculate_coordinates::{MinimalCrossings, VDir, HDir, MarkedTypeOneConflicts}, p1_layering::{start, Vertex, Edge}, p2_reduce_crossings::InsertDummyVertices}, util::into_weakly_connected_components};

type Layouts = Vec<(Vec<((usize, usize), (isize, isize))>, usize, usize)>;
type Layout = (Vec<((usize, usize), (isize, isize))>, usize, usize);

pub fn build_layout_from_edges(edges: &[(u32, u32)], minimum_length: u32, vertex_spacing: usize) -> Layouts {
    let mut graph = StableDiGraph::<Vertex, Edge>::from_edges(edges);
    // initialize vertex ids to NodeIndex
    let indices = graph.node_indices().collect::<Vec<_>>();
    for i in indices {
        graph[i].id = i.index();
    }
    into_weakly_connected_components(graph).into_iter()
        .map(|graph| build_layout(graph, minimum_length, vertex_spacing))
        .collect()
}

pub fn build_layout_from_vertices_and_edges(vertices: &[u32], edges: &[(u32, u32)], minimum_length: u32, vertex_spacing: usize) -> Layouts {
    // add all edges
    
    let mut graph = StableDiGraph::<Vertex, Edge>::new();
    
    // add all vertices which have no edges
    for v in vertices {
        graph.add_node(Vertex::from_id(*v as usize));
    }

    for (from, to) in edges {
        graph.add_edge(NodeIndex::from(*from), NodeIndex::from(*to), Edge::default());
    }

    // initialize vertex ids to NodeIndex
    let indices = graph.node_indices().collect::<Vec<_>>();
    for i in indices {
        graph[i].id = i.index();
    }
    into_weakly_connected_components(graph).into_iter()
        .map(|graph| build_layout(graph, minimum_length, vertex_spacing))
        .collect()

}

pub fn build_layout_from_graph(graph: StableDiGraph<usize, usize>, minimum_length: u32, vertex_spacing: usize) -> Layout {
    let graph = graph.map(|_, w| Vertex::from_id(*w), |_, _| Edge::default());
    // build into subgraphs
    build_layout(graph, minimum_length, vertex_spacing)
}

fn build_layout(graph: StableDiGraph<Vertex, Edge>, minimum_length: u32, vertex_spacing: usize) -> Layout {
    let proper_graph = rank(graph, minimum_length);
    let minimal_crossings = minimize_crossings(proper_graph);
    calculate_coordinates(minimal_crossings, vertex_spacing)
}

fn rank(graph: StableDiGraph<Vertex, Edge>, minimum_length: u32) -> InsertDummyVertices {
    start(graph, minimum_length).init_rank().make_tight().init_cutvalues().init_low_lim().rank().into()
}

fn minimize_crossings(graph: InsertDummyVertices) -> MinimalCrossings {
    graph.prepare_for_initial_ordering().ordering()
}
            
fn squeeze_layout(layout: &mut HashMap<NodeIndex, isize>, marked: MarkedTypeOneConflicts) {
    // remove all dummies from layout and move nodes to the right
}

// TODO: Put this in p3 module
/// Calculates the final x-coordinates for each vertex, after the graph was layered and crossings where minimized.
fn calculate_coordinates(graph: MinimalCrossings, vertex_spacing: usize) -> Layout {
    let y_coordinates = graph.layers.iter()
        .enumerate()
        .map(|(rank, row)| row.iter().map(move |v| (*v, rank as isize * vertex_spacing as isize * -1)))
        .flatten()
        .collect::<HashMap<NodeIndex, isize>>(); 
    let mut layouts = Vec::new();
    let marked = graph.mark_type_1_conflicts();
    
    // calculate the coordinates for each direction
    for vertical_direction in &[VDir::Up, VDir::Down] {
        for horizontal_direction in &[HDir::Left, HDir::Right] {
            let layout = marked.clone()
                               .create_vertical_alignments(*vertical_direction, *horizontal_direction)
                               .do_horizontal_compaction(vertex_spacing, *horizontal_direction);
            layouts.push(layout);
        }
    }

    // remove all dummies
    let dummies = marked.node_indices().filter(|v| marked[*v].is_dummy).collect::<HashSet<_>>();
    for l in &mut layouts {
        for d in &dummies {
            l.remove(d);
        }
    }
    
    // determine minimum and maximum coordinate of each layout, plus the width
    let min_max: Vec<(isize, isize, isize)> = layouts.iter()
                                                 .map(|c| {
                                                    let min = *c.values().min().unwrap();
                                                    let max = *c.values().max().unwrap();
                                                    (min, max, max - min)
                                                 }).collect();

    // determine the layout with the minimum width
    let min_width = min_max.iter().enumerate().min_by(|a, b| a.1.2.cmp(&b.1.2)).unwrap().0;

    // align all other layouts to the lowest/highest coordinate of the layout with the minimum width, 
    // depending on the horizontal direction which was chosen to create them
    for (i, layout) in layouts.iter_mut().enumerate() {
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

    // sort all 4 coordinates per vertex in ascending order
    let mut sorted_layouts = HashMap::new();
    for k in layouts.get(0).unwrap().keys() {
        let mut vertex_coordinates = [
            *layouts.get(0).unwrap().get(k).unwrap(),
            *layouts.get(1).unwrap().get(k).unwrap(),
            *layouts.get(2).unwrap().get(k).unwrap(),
            *layouts.get(3).unwrap().get(k).unwrap(),
        ];
        vertex_coordinates.sort();
        sorted_layouts.insert(k, vertex_coordinates);
    }

    // create final layout, by averaging the two median values
    let mut final_layout = sorted_layouts.into_iter()
                                         .map(|(k, v)| (*k, (v[1] + v[2]) / 2))
                                         .collect::<Vec<_>>();
    // determine the smallest x-coordinate
    let min = final_layout.iter().min_by(|a, b| a.1.cmp(&b.1)).unwrap().1;

    // shift all coordinates so the minimum coordinate is 0
    for (_, c) in &mut final_layout {
        *c -= min;
    }
    let width = marked.layers.iter().map(|l| l.len()).max().unwrap_or(0);
    let height = marked.layers.len();
    let layout = final_layout.into_iter()
        .filter(|(v, _)| !marked[*v].is_dummy )
        .map(|(v, x)| (
            (v.index(), marked[v].id), 
            (x, *y_coordinates.get(&v).unwrap()), 
            ))
        .collect::<Vec<_>>();
    (layout, width, height)

}

#[cfg(test)]
mod benchmark {
    use super::build_layout_from_edges;

    #[test]
    fn r_100() {
        let edges = graph_generator::RandomLayout::new(100).build_edges().into_iter().map(|(r, l)| (r as u32, l as u32)).collect::<Vec<(u32, u32)>>();
        let start = std::time::Instant::now();
        let _ = build_layout_from_edges(&edges, 1, 10);
        println!("Random 100 edges: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn r_1000() {
        let edges = graph_generator::RandomLayout::new(1000).build_edges().into_iter().map(|(r, l)| (r as u32, l as u32)).collect::<Vec<(u32, u32)>>();
        let start = std::time::Instant::now();
        let _ = build_layout_from_edges(&edges, 1, 10);
        println!("Random 1000 edges: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn r_2000() {
        let edges = graph_generator::RandomLayout::new(2000).build_edges();
        let start = std::time::Instant::now();
        let _ = build_layout_from_edges(&edges, 1, 10);
        println!("Random 2000 edges: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn r_4000() {
        let edges = graph_generator::RandomLayout::new(2000).build_edges();
        let start = std::time::Instant::now();
        let _ = build_layout_from_edges(&edges, 1, 10);
        println!("Random 4000 edges: {}ms", start.elapsed().as_millis());
    }
    #[test]
    fn r_8000() {
        let edges = graph_generator::RandomLayout::new(8000).build_edges();
        let start = std::time::Instant::now();
        let _ = build_layout_from_edges(&edges, 1, 10);
        println!("Random 8000 edges: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn l_1000_2() {
        let n = 1000;
        let e = 2;
        let edges = graph_generator::GraphLayout::new_from_num_nodes(n, e).build_edges();
        let start = std::time::Instant::now();
        let _ = build_layout_from_edges(&edges, 1, 10);
        println!("{n} nodes, {e} edges per node: {}ms", start.elapsed().as_millis());
    }
}

#[cfg(test)]
mod check_visuals {
    use std::default;

    use petgraph::stable_graph::StableDiGraph;

    use crate::{phases::p1_layering::{Vertex, Edge}, algorithm::{build_layout_from_graph, build_layout}, util::into_weakly_connected_components};

    use super::build_layout_from_edges;
    
    #[test]
    fn verify_looks_good() {
        let edges = [
                (0, 1), 
                (1, 2), 
                (2, 3), (2, 4), 
                (3, 5), (3, 6), (3, 7), (3, 8), (4, 5), (4, 6), (4, 7), (4, 8),
                (5, 9), (6, 9), (7, 9), (8, 9)
        ];
        let layout = build_layout_from_edges(&edges, 1, 10); 
        println!("{:?}", layout);
    }

    #[test]
    fn check_coords() {
        let mut graph = StableDiGraph::<Vertex, Edge>::new();
        let _0 = graph.add_node(Vertex::from_id(0));
        let _1 = graph.add_node(Vertex::from_id(1));
        let _2 = graph.add_node(Vertex::from_id(2));
        let _3 = graph.add_node(Vertex::from_id(3));
        let _4 = graph.add_node(Vertex::from_id(4));
        graph.add_edge(_1, _0, Edge::default());
        graph.add_edge(_2, _1, Edge::default());
        graph.add_edge(_3, _0, Edge::default());
        graph.add_edge(_4, _0, Edge::default());
        let edges = [(1, 0), (2, 1), (3, 0), (4, 0)];
        let layout = into_weakly_connected_components(graph).into_iter()
        .map(|graph| build_layout(graph, 1, 10))
        .collect::<Vec<_>>();
        println!("{:?}", layout);
    }

    #[test]
    fn check_coords_2() {
        let mut graph = StableDiGraph::<Vertex, Edge>::from_edges([(0, 1), (0, 2), (0, 3), (1, 4), (4, 5), (5, 6), (2, 6), (3, 6), (3, 7), (3, 8), (3, 9)]);
        for n in graph.node_indices().collect::<Vec<_>>() {
            graph[n].id = n.index()
        }

        let layout = into_weakly_connected_components(graph).into_iter()
        .map(|graph| build_layout(graph, 1, 10))
        .collect::<Vec<_>>();
        println!("{:?}", layout);
    }

}
