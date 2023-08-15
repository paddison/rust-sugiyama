use std::collections::HashMap;

use petgraph::stable_graph::NodeIndex;

use crate::graphs::{p3_calculate_coordinates::{MinimalCrossings, VDir, HDir}, p1_create_layers::{FeasibleTree, UnlayeredGraph, self}};

fn feasible_tree<T: Default>(edges: &[(u32, u32)], minimum_length: usize) -> FeasibleTree<T> {
    let graph = petgraph::stable_graph::StableDiGraph::<Option<T>, usize>::from_edges(edges);
    p1_create_layers::start_layering(graph)
                     .initial_ranking(minimum_length)
                     .make_tight()
                     .init_cutvalues()
}

/// Calculates the final x-coordinates for each vertex, after the graph was layered and crossings where minimized.
pub fn calculate_coordinates<T: Default + Clone>(graph: MinimalCrossings<T>, vertex_spacing: usize) -> Vec<(NodeIndex, isize)>{
    let mut layouts = Vec::new();
    let marked = graph.mark_type_1_conflicts();
    
    // calculate the coordinates for each direction
    for vertical_direction in &[VDir::Up, VDir::Down] {
        for horizontal_direction in &[HDir::Left, HDir::Right] {
            let layout = marked.clone()
                               .create_vertical_alignments(*vertical_direction, *horizontal_direction)
                               .do_horizontal_compaction(*vertical_direction, *horizontal_direction, vertex_spacing);
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

    final_layout
}

#[cfg(test)]
mod test {
    use petgraph::stable_graph::{NodeIndex, StableDiGraph};

    use crate::{graphs::p3_calculate_coordinates::MinimalCrossings, util::layers::Layers, algorithm::calculate_coordinates};

    pub fn g_levels(levels: usize) -> MinimalCrossings<usize>{
        let mut edges = Vec::new();
        let mut layers = Vec::new();
        let mut id = 0;
        for l in 0..levels {

            let mut level = Vec::new();
            for _ in 0..2_usize.pow(l as u32) {
                level.push(NodeIndex::from(id as u32));
                id += 1;
            }
            layers.push(level);
        } 

        for level in &layers[0..layers.len() - 1] {
            for n in level {
                edges.push((n.index() as u32, n.index() as u32 * 2 + 1));
                edges.push((n.index() as u32, n.index() as u32 * 2 + 2));
            }
        }

        let g = StableDiGraph::from_edges(&edges);
        let layers = Layers::new(layers, &g);

        MinimalCrossings::new(layers, g)
    }
    
    fn _test() -> MinimalCrossings<usize> {
        let edges: [(usize, usize); 29] = [(0, 2), (0, 6), (1, 16), (1, 17), 
                        (3, 8), (16, 8), (4, 8), (17, 19), (18, 20), (5, 8), (5, 9), (6, 8), (6, 21),
                        (7, 10), (7, 11), (7, 12), (19, 23), (20, 24), (21, 12), (9, 22), (9, 25),
                        (10, 13), (10, 14), (11, 14), (22, 13), (23, 15), (24, 15), (12, 15), (25, 15)];

        let layers_raw: Vec<Vec<NodeIndex>> = [
            vec![0, 1],
            vec![2, 3, 16, 4, 17, 18, 5, 6],
            vec![7, 8, 19, 20, 21, 9],
            vec![10, 11, 22, 23, 24, 12, 25],
            vec![13, 14, 15],
        ].into_iter().map(|row| row.into_iter().map(|id| id.into()).collect())
        .collect();
        

        let mut graph = StableDiGraph::new();

        for _ in 0..16 {
            graph.add_node(Some(usize::default()));
        }
        for _ in 0..10 {
            graph.add_node(None);
        }

        for (a, b) in edges {
            graph.add_edge(NodeIndex::new(a), NodeIndex::new(b), usize::default());
        }

        let layers = Layers::new::<usize>(layers_raw, &graph);
        MinimalCrossings::new(layers, graph)
    }

    fn _g() -> MinimalCrossings<usize>{
        let edges: Vec<(u32, u32)> = vec![(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)];
        let layers:  Vec<Vec<NodeIndex>> = [
            vec![0],
            vec![1, 2], 
            vec![3, 4, 5, 6]
        ].into_iter().map(|r| r.into_iter().map(|id| id.into()).collect()).collect();

        let g = StableDiGraph::from_edges(edges);
        let layers = Layers::new(layers, &g);

        MinimalCrossings::new(layers, g)
    }

    #[test]
    fn benchmark() {
        let stack_size = 128 * 1024 * 1024;
        let child = std::thread::Builder::new()
            .stack_size(stack_size)
            .spawn(|| {
                let g = g_levels(13);
                let start = std::time::Instant::now();
                let _ = calculate_coordinates(g, 10);
                println!("{}ms", start.elapsed().as_millis());
            }).unwrap();

        // Wait for thread to join
        child.join().unwrap();
    }

    #[test]
    fn cmp_with_temanejo() {
        let mut g = StableDiGraph::from_edges(&[
            (0, 1), 
            (1, 2), 
            (2, 3), (2, 4), 
            (3, 5), (3, 6), (3, 7), (3, 8), (4, 5), (4, 6), (4, 7), (4, 8),
            (5, 9), (6, 9), (7, 9), (8, 9)]);
        
        for n in 0..10 {
            let w: &mut Option<usize> = g.node_weight_mut(n.into()).unwrap();
            w.replace(1_usize);
        }
        
        let layers_raw = vec![
            vec![0.into()],
            vec![1.into()],
            vec![2.into()],
            vec![3.into(), 4.into()],
            vec![5.into(), 6.into(), 7.into(), 8.into()],
            vec![9.into()],
        ];
        let layers = Layers::new(layers_raw, &g);

        let mc = MinimalCrossings::<usize>::new(layers, g);
        let mut coords = calculate_coordinates(mc, 20);
        coords.sort_by(|a, b| a.0.cmp(&b.0));
        println!("{coords:?}");
    }
}