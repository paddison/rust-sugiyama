use petgraph::stable_graph::{NodeIndex, StableDiGraph};
use rust_sugiyama::from_graph;
use std::collections::HashMap;

fn main() {
    let mut g: StableDiGraph<String, usize> = StableDiGraph::new();

    let rick = g.add_node("Rick".to_string());
    let morty = g.add_node("Morty".to_string());
    let beth = g.add_node("Beth".to_string());
    let jerry = g.add_node("Jerry".to_string());
    let summer = g.add_node("Summer".to_string());

    g.add_edge(rick, beth, 1);
    g.add_edge(rick, jerry, 1);
    g.add_edge(beth, summer, 1);
    g.add_edge(jerry, summer, 1);
    g.add_edge(beth, morty, 1);
    g.add_edge(jerry, morty, 1);

    let layouts = from_graph(&g)
        .build()
        .into_iter()
        .map(|(layout, width, height)| {
            let mut new_layout = HashMap::new();
            for (id, coords) in layout {
                new_layout.insert(g[NodeIndex::from(id)].clone(), coords);
            }
            (new_layout, width, height)
        })
        .collect::<Vec<_>>();

    for (layout, width, height) in layouts {
        println!("Coordinates: {:?}", layout);
        println!("width: {width}, height: {height}");
    }
}
