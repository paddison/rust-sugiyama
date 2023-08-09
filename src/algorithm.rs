use std::collections::HashMap;

use petgraph::stable_graph::{NodeIndex, StableDiGraph};

use crate::{graphs::calculate_coordinates::{MinimalCrossings, VDir, HDir}, util::layers::Layers};

pub fn g_levels(levels: usize) -> MinimalCrossings<usize>{
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

    let mut edges = Vec::new();
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

pub fn calculate_coordinates<T: Default + Clone>(graph: MinimalCrossings<T>) {
    let start = std::time::Instant::now();
    let marked = graph.mark_type_1_conflicts();
    println!("mark: {}", start.elapsed().as_millis());
    let mut all_coords = Vec::new();
    for vertical_direction in &[VDir::Up, VDir::Down] {
        for horizontal_direction in &[HDir::Left, HDir::Right] {
            let clone = marked.clone();
            let start = std::time::Instant::now();
            println!("clone: {}", start.elapsed().as_millis());
            let start = std::time::Instant::now();
            let coords = clone
                            .create_vertical_alignments(*vertical_direction, *horizontal_direction)
                            .do_horizontal_compaction(*vertical_direction, *horizontal_direction);
            println!("algo: {}", start.elapsed().as_millis());
            all_coords.push(coords);
        }
    }
    // min max width
    let min_max: Vec<(isize, isize, isize)> = all_coords.iter()
                                                 .map(|c| {
                                                    let min = *c.values().min().unwrap();
                                                    let max = *c.values().max().unwrap();
                                                    (min, max, max - min)
                                                 }).collect();

    let min_width = min_max.iter().enumerate().min_by(|a, b| a.1.2.cmp(&b.1.2)).unwrap().0;

    for (i, coords) in all_coords.iter_mut().enumerate() {
        let shift = if i % 2 == 0 { min_max[i].0 as isize - min_max[min_width].0 as isize} else { min_max[min_width].1  as isize - min_max[i].1 as isize };
        for v in coords.values_mut() {
            let new = *v as isize + shift;
            *v = new as isize;
        }
    }

    let mut c = HashMap::new();
    for k in all_coords.get(0).unwrap().keys() {
        c.insert(k, [
            *all_coords.get(0).unwrap().get(k).unwrap(),
            *all_coords.get(1).unwrap().get(k).unwrap(),
            *all_coords.get(2).unwrap().get(k).unwrap(),
            *all_coords.get(3).unwrap().get(k).unwrap(),
        ]);
    }

    for v in c.values_mut() {
        v.sort();
    }

    let mut final_coords = c.into_iter().map(|(k, v)| (k, (v[1] + v[2]) / 2)).collect::<Vec<_>>();
    let min = final_coords.iter().min_by(|a, b| a.1.cmp(&b.1)).unwrap().1;
    for (_, c) in &mut final_coords {
        *c -= min;
    }
    let end = start.elapsed();
    println!("total: {}", end.as_millis());
    final_coords.sort_by(|(l, _), (r, _)| l.cmp(r));

    
    // println!("{:?}", final_coords);
    // dbg!(final_coords);
}

#[cfg(test)]
mod test {
    use petgraph::stable_graph::{NodeIndex, StableDiGraph};

    use crate::{graphs::calculate_coordinates::MinimalCrossings, util::layers::Layers};

    
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

}