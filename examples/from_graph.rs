use petgraph::{
    stable_graph::{NodeIndex, StableDiGraph},
    visit::EdgeRef,
    Direction,
};
use rust_sugiyama::{configure::Config, from_graph};
use std::collections::HashMap;
use svg::{
    node::{
        element::{Line, Rectangle, Text},
        Comment,
    },
    Document,
};

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

    let layouts = from_graph(
        &g,
        &Config {
            vertex_spacing: 100,
            ..Config::default()
        },
    )
    .into_iter()
    .map(|(layout, width, height)| {
        println!("{}", create_svg(&g, &layout, 15.0, 25.0, (10.0, 10.0)));

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

/// Generates an SVG string of the resulting layout. This is an example of
/// rendering the layouts.
fn create_svg(
    graph: &StableDiGraph<String, usize>,
    layout: &[(NodeIndex, (isize, isize))],
    character_width: f32,
    node_height: f32,
    padding: (f32, f32),
) -> String {
    // Figure out the extents of the SVG.
    let node_size = |idx: NodeIndex| {
        (
            graph.node_weight(idx).unwrap().len() as f32 * character_width,
            node_height,
        )
    };
    let f32_cmp = |a: &f32, b: &f32| a.partial_cmp(b).unwrap();
    let min_x = layout
        .iter()
        .map(|(idx, (x, _))| *x as f32 - node_size(*idx).0 * 0.5)
        .min_by(f32_cmp)
        .unwrap();
    let max_x = layout
        .iter()
        .map(|(idx, (x, _))| *x as f32 + node_size(*idx).0 * 0.5)
        .max_by(f32_cmp)
        .unwrap();
    let min_y = layout
        .iter()
        .map(|(idx, (_, y))| *y as f32 - node_size(*idx).1 * 0.5)
        .min_by(f32_cmp)
        .unwrap();
    let max_y = layout
        .iter()
        .map(|(idx, (_, y))| *y as f32 + node_size(*idx).1 * 0.5)
        .max_by(f32_cmp)
        .unwrap();

    let mut document = Document::new()
        .set("width", (max_x - min_x) as f32 + 2.0 * padding.0)
        .set("height", (max_y - min_y) as f32 + 2.0 * padding.1);

    let origin = (padding.0 - min_x as f32, padding.1 - min_y as f32);

    // Generate a line for each edge in the graph.
    for (idx, pos) in layout {
        let vertex_name = graph.node_weight(*idx).unwrap();
        for e in graph.edges_directed(*idx, Direction::Outgoing) {
            let other_vertex_index = e.target();
            let other_vertex_name = graph.node_weight(other_vertex_index).unwrap();
            let (_, other_vertex_pos) = layout.iter().find(|e| e.0 == other_vertex_index).unwrap();

            let line = Line::new()
                .set("x1", pos.0 as f32 + origin.0)
                .set("y1", pos.1 as f32 + origin.1)
                .set("x2", other_vertex_pos.0 as f32 + origin.0)
                .set("y2", other_vertex_pos.1 as f32 + origin.1)
                .set("style", "stroke:black;")
                .add(Comment::new(format!(
                    "({vertex_name}, {other_vertex_name})"
                )));
            document = document.add(line);
        }
    }

    // Generate a rectangle for each edge in the graph.
    for (idx, pos) in layout {
        let vertex_name = graph.node_weight(*idx).unwrap();

        let size = (vertex_name.len() as f32 * character_width, node_height);

        let rect = Rectangle::new()
            .set("x", pos.0 as f32 - size.0 * 0.5 + origin.0)
            .set("y", pos.1 as f32 - size.1 * 0.5 + origin.1)
            .set("width", size.0)
            .set("height", size.1)
            .set("style", "fill:white;stroke:black;");

        let text = Text::new(vertex_name)
            .set("x", pos.0 as f32 + origin.0)
            .set("y", pos.1 as f32 + origin.1)
            .set("dominant-baseline", "middle")
            .set("text-anchor", "middle");
        document = document.add(rect).add(text);
    }

    document.to_string()
}
