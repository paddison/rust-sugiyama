use std::collections::HashMap;

use petgraph::stable_graph::{NodeIndex, StableDiGraph};

use crate::{graphs::calculate_coordinates::{MinimalCrossings, VerticalDirection, HorizontalDirection}, util::layers::Layers};

fn calculate_coordinates() {
    let marked = test().mark_type_1_conflicts();
    let mut all_coords = Vec::new();
    for vertical_direction in [VerticalDirection::Up, VerticalDirection::Down] {
        for horizontal_direction in [HorizontalDirection::Left, HorizontalDirection::Right] {
            let coords = marked.clone()
                               .create_vertical_alignments(vertical_direction, horizontal_direction)
                               .do_horizontal_compaction();
            all_coords.push(coords);
        }
    }
    let maxs = [
        *all_coords.get(0).unwrap().values().max().unwrap(),
        *all_coords.get(1).unwrap().values().max().unwrap(),
        *all_coords.get(2).unwrap().values().max().unwrap(),
        *all_coords.get(3).unwrap().values().max().unwrap(),
    ];
    let mins = [
        *all_coords.get(0).unwrap().values().min().unwrap(),
        *all_coords.get(1).unwrap().values().min().unwrap(),
        *all_coords.get(2).unwrap().values().min().unwrap(),
        *all_coords.get(3).unwrap().values().min().unwrap(),
    ];
    let min_width = mins.iter()
                        .zip(maxs)
                        .enumerate()
                        .map(|(i, (min, max))| (i, min.abs_diff(max)))
                        .min_by(|(_, width), (_, width2)| width.cmp(width2))
                        .unwrap().0;

    dbg!(&all_coords);

    for (i, coords) in all_coords.iter_mut().enumerate() {
        let shift = if i % 2 == 0 { mins[i] as isize - mins[min_width] as isize} else { maxs[min_width]  as isize - maxs[i] as isize };
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
    final_coords.sort_by(|(l, _), (r, _)| l.cmp(r));
    
    println!("{:?}", final_coords);
}

#[test]
fn vbla() {
    calculate_coordinates();
}

fn test() -> MinimalCrossings<usize> {
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
    
    let layers = Layers::new_from_layers(layers_raw);

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

    MinimalCrossings::new(layers, graph)
}