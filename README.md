# Rust Sugiyama

## Description

An implementation of Sugiyamas algorithm for displaying a layered graph.

Currently, the implementation is complete, except for graphs which contain cycles. 

The rank assignment algorithm is implemented according to the paper `A Technique for Drawing Directed Graphs` by Gansner et al. which can be found [here](https://ieeexplore.ieee.org/document/221135). It first assigns a node a layer and creates an optimal feasible tree for rank assignment.

Crossing Reduction follows the weighted median heuristic which is also descriped in the above paper. In order to count crossings, the Bilayer Cross Count algorithm as described in the paper `Simple and Efficient Bilayer Cross Counting` by Wilhelm Barth and Petra Mutzel and Michael Juenger. It can also be found [online](http://ls11-www.cs.tu-dortmund.de/downloads/papers/BJM04.pdf).

Finally, the implementation for coordinate assignment follows the algorithm provided by Brandes and Koepf, which can be found in this [paper](https://www.semanticscholar.org/paper/Fast-and-Simple-Horizontal-Coordinate-Assignment-Brandes-K%C3%B6pf/69cb129a8963b21775d6382d15b0b447b01eb1f8).

## Usage

Currently, there are two options to create a layout, both take as input a minimum edge length and a minimum spacing between the vertices.
They will divide the graph into its connected components and calculate the coordinates seperately for each component.

### build_layout_from_edges
This takes a `&[u32, u32]` slice and calculates the x and y coordinates, the height of the graph, and the width.

```rust
from rust_sugiyama import build_layout_from_edges;

let edges = [
    (0, 1), 
    (1, 2), 
    (1, 3), (1, 4), (1, 5), (1, 6), 
    (3, 7), (3, 8), (4, 7), (4, 8), 
    (5, 7), (5, 8), (6, 7), (6, 8), 
    (7, 9), (8, 9)
];
let layouts = build_layout_from_edges(&edges, 1, 10);
for (layout, width, height) in layouts {
    println!("Coordinates: {:?}", layouts);
    println!("width: {width}, height: {height}");
}
```

### build_layout_from_graph
Takes as input a `&StableDiGraph<V, E>` and calculates the x and y coordinates, the height and width of the graph.
`NodeIndices` are preserved between layouts and map directly to the input graph.

```rust
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

let layouts = build_layout_from_edges(&edges, 1, 10)
    .into_iter()
    .map(|(layout, width, height)| {
        let mut new_layout = HashMap::new();
        for (id, coords) in layout {
            new_layout.insert(g[NodeIndex::from(id)], coords);
        }
        (new_layout, width, height)
    })
    .collect::<Vec<_>>(); 

for (layout, width, height) in layouts {
    println!("Coordinates: {:?}", layouts);
    println!("width: {width}, height: {height}");
}
```
