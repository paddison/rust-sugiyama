# Rust Sugiyama
![example worklfow](https://github.com/paddison/rust-sugiyama/actions/workflows/rust_ci.yml/badge.svg)

## Description

An implementation of Sugiyamas algorithm for displaying a layered graph.

This crate heavily uses the crate [petgraph](https://crates.io/crates/petgraph) under the hood.

Cycle Removal is implemented by using the `greedy_feedback_arc_set` function of petgraph and then reversing the edges from the set.

The rank assignment algorithm is implemented according to the paper `A Technique for Drawing Directed Graphs` by Gansner et al. which can be found [here](https://ieeexplore.ieee.org/document/221135). It first assigns a node a layer and creates an optimal feasible tree for rank assignment.

Crossing Reduction follows the weighted median heuristic which is also descriped in the above paper, it is also possible to use the barycenter heuristic for crossing reduction via configuration. In order to count crossings, the Bilayer Cross Count algorithm as described in the paper `Simple and Efficient Bilayer Cross Counting` by Wilhelm Barth and Petra Mutzel and Michael Juenger. It can also be found [online](http://ls11-www.cs.tu-dortmund.de/downloads/papers/BJM04.pdf).

Finally, the implementation for coordinate assignment follows the algorithm provided by Brandes and Koepf, which can be found in this [paper](https://www.semanticscholar.org/paper/Fast-and-Simple-Horizontal-Coordinate-Assignment-Brandes-K%C3%B6pf/69cb129a8963b21775d6382d15b0b447b01eb1f8).

Bugs or feature requests can be either submitted via a github issue or by contacting patrickbaumann579@gmail.com.

## Usage

Currently, there are three options to create a layout: 
1. `from_edges`, which takes a `&[(u32, u32)]`
2. `from_vertices_and_edges`, which takes a `&[u32]` and a `&[(u32, u32)]`
3. `from_graph`, which takes a `petgraph::StableDiGraph<V, E>`

They will divide the graph into its connected components and calculate the coordinates seperately for each component.
The API is implemented via the builder pattern, where a user may specify values like the minimum spacing between vertices etc.

### build_layout_from_edges
This takes a `&[u32, u32]` slice and calculates the x and y coordinates, the height of the graph, and the width.

```rust
use rust_sugiyama::{configure::Config, from_edges};

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
        vertex_spacing: 20,
        ..Default::default()
    },
);

for (layout, width, height) in layouts {
    println!("Coordinates: {:?}", layout);
    println!("width: {width}, height: {height}");
}
```

### build_layout_from_graph
Takes as input a `&StableDiGraph<V, E>` and calculates the x and y coordinates, the height and width of the graph.
`NodeIndices` are preserved between layouts and map directly to the input graph.

```rust
use rust_sugiyama::{configure::Config, from_graph};

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
        ..Default::default()
    },
)
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
```

### configuration via envs
It is also possible to configure the algorithm via environment variables, using the method `configure_from_env()`. 

Environment variables that can be set are:

|ENV|values|default|description|
|---|------|-------|-------|
| RUST_GRAPH_MIN_LEN    | integer, > 0                | 1          | minimum edge length between layers |
| RUST_GRAPH_V_SPACING  | integer, > 0                | 10         | minimum spacing between vertices on the same layer |
| RUST_GRAPH_DUMMIES    | (y\|n)                       | y          | if dummy vertices are included in the final layout |
| RUST_GRAPH_R_TYPE     | (original\|minimize\|up\|down) | minimize   | defines how vertices are places vertically |
| RUST_GRAPH_CROSS_MIN  | (barycenter\|median)         | barycenter | which heuristic to use for crossing reduction |
| RUST_GRAPH_TRANSPOSE  | (y\|n)                       | y          | if transpose function is used to further try to reduce crossings (may increase runtime significally for large graphs) |
| RUST_GRAPH_DUMMY_SIZE | float, > 0, <= 1            | 1.0        |size of dummy vertices in final layout, if dummy vertices are included. this will squish the graph horizontally |



