
fn main() {
    use rust_sugiyama::from_graph;
    let mut g = petgraph::stable_graph::StableDiGraph::<String, usize>::new();

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

    let vertex_size_fn = |idx: petgraph::stable_graph::NodeIndex| (g.node_weight(idx).unwrap().len() as f32 * 15.0, 25.0);

    let layouts = from_graph(&g)
        .vertex_spacing(40)
        .vertex_sizing_fn(&vertex_size_fn)
        .build()
        .into_iter()
        .map(|(layout, width, height)| {
            println!("{}", rust_sugiyama::to_svg(&g, &layout, &vertex_size_fn, (10.0, 10.0)));
            
            let mut new_layout = std::collections::HashMap::new();
            for (id, coords) in layout {
                new_layout.insert(g[petgraph::stable_graph::NodeIndex::from(id)].clone(), coords);
            }
            (new_layout, width, height)
        })
        .collect::<Vec<_>>(); 

    for (layout, width, height) in layouts {
        println!("Coordinates: {:?}", layout);
        println!("width: {width}, height: {height}");
    }
}