use std::collections::HashMap;

use petgraph::stable_graph::NodeIndex;

use crate::phases::{p3_calculate_coordinates::{MinimalCrossings, VDir, HDir}, p1_layering::start, p2_reduce_crossings::InsertDummyVertices};

pub fn build_layout(edges: &[(u32, u32)], minimum_length: u32) -> Vec<(usize, (isize, isize))> {
    let proper_graph = rank(edges, minimum_length);
    let minimal_crossings = minimize_crossings(proper_graph);
    calculate_coordinates(minimal_crossings, 10)
}

pub fn rank(edges: &[(u32, u32)], minimum_length: u32) -> InsertDummyVertices {
    start(edges, minimum_length).init_rank().make_tight().init_cutvalues().init_low_lim().rank().into()
}

pub fn minimize_crossings(graph: InsertDummyVertices) -> MinimalCrossings {
    graph.prepare_for_initial_ordering().ordering()
}

/// Calculates the final x-coordinates for each vertex, after the graph was layered and crossings where minimized.
pub fn calculate_coordinates(graph: MinimalCrossings, vertex_spacing: usize) -> Vec<(usize, (isize, isize))>{
    let y_coordinates = graph.layers.iter()
        .enumerate()
        .map(|(rank, row)| row.iter().map(move |v| (*v, rank as isize * vertex_spacing as isize)))
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

    // min max width
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

    final_layout.into_iter().map(|(v, x)| (v.index(), (x, *y_coordinates.get(&v).unwrap()))).collect()

}

#[cfg(test)]
mod benchmark {
    use super::build_layout;

    #[test]
    fn r_100() {
        let edges = graph_generator::RandomLayout::new(100).build_edges().into_iter().map(|(r, l)| (r as u32, l as u32)).collect::<Vec<(u32, u32)>>();
        let start = std::time::Instant::now();
        let _ = build_layout(&edges, 1);
        println!("Random 100 edges: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn r_1000() {
        let edges = graph_generator::RandomLayout::new(1000).build_edges().into_iter().map(|(r, l)| (r as u32, l as u32)).collect::<Vec<(u32, u32)>>();
        let start = std::time::Instant::now();
        let _ = build_layout(&edges, 1);
        println!("Random 1000 edges: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn r_2000() {
        let edges = graph_generator::RandomLayout::new(2000).build_edges();
        let start = std::time::Instant::now();
        let _ = build_layout(&edges, 1);
        println!("Random 2000 edges: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn r_4000() {
        let edges = graph_generator::RandomLayout::new(2000).build_edges();
        let start = std::time::Instant::now();
        let _ = build_layout(&edges, 1);
        println!("Random 4000 edges: {}ms", start.elapsed().as_millis());
    }
    #[test]
    fn r_8000() {
        let edges = graph_generator::RandomLayout::new(8000).build_edges();
        let start = std::time::Instant::now();
        let _ = build_layout(&edges, 1);
        println!("Random 8000 edges: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn l_1000_2() {
        let n = 1000;
        let e = 2;
        let edges = graph_generator::GraphLayout::new_from_num_nodes(n, e).build_edges();
        let start = std::time::Instant::now();
        let _ = build_layout(&edges, 1);
        println!("{n} nodes, {e} edges per node: {}ms", start.elapsed().as_millis());
    }
}

#[cfg(test)]
mod check_visuals {
    use super::build_layout;
    
    #[test]
    fn verify_looks_good() {
        let edges = [
                (0, 1), 
                (1, 2), 
                (2, 3), (2, 4), 
                (3, 5), (3, 6), (3, 7), (3, 8), (4, 5), (4, 6), (4, 7), (4, 8),
                (5, 9), (6, 9), (7, 9), (8, 9)
        ];
        let layout = build_layout(&edges, 1); 
        println!("{:?}", layout);
    }

}
