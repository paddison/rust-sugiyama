use rust_sugiyama::{configure::Config, from_edges};

fn main() {
    let edges = [
        (0, 1),
        //
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
        //
        (3, 7),
        (3, 8),
        //
        (4, 7),
        (4, 8),
        //
        (5, 7),
        (5, 8),
        //
        (6, 7),
        (6, 8),
        //
        (7, 9),
        //
        (8, 9),
    ];

    let layouts = from_edges(
        &edges,
        &Config {
            vertex_spacing: 20.0,
            ..Default::default()
        },
    );

    for (layout, width, height) in layouts {
        println!("Coordinates: {:?}", layout);
        println!("width: {width}, height: {height}");
    }
}
