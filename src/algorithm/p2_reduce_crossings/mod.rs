#[cfg(test)]
mod tests;
use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::ops::{Deref, DerefMut};

use petgraph::algo::toposort;
use petgraph::stable_graph::{NodeIndex, StableDiGraph};
use petgraph::Direction::{Incoming, Outgoing};

use crate::util;
use crate::util::{radix_sort, IterDir};

use super::{slack, Edge, Vertex};

// later test if its better to access neighbors via graph or to store them separately
#[derive(Clone)]
struct Order {
    _inner: Vec<Vec<NodeIndex>>,
    positions: HashMap<NodeIndex, usize>,
}

impl Display for Order {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        for row in &self._inner {
            for c in row {
                s.push_str(&c.index().to_string());
                s.push(',')
            }
            s.push('\n');
        }
        f.write_str(&s)
    }
}

impl Order {
    fn new(layers: Vec<Vec<NodeIndex>>) -> Self {
        let mut positions = HashMap::new();
        for l in &layers {
            for (pos, v) in l.iter().enumerate() {
                positions.insert(*v, pos);
            }
        }
        Self {
            _inner: layers,
            positions,
        }
    }

    fn max_rank(&self) -> usize {
        self.len()
    }

    fn crossings(&self, graph: &StableDiGraph<Vertex, Edge>) -> usize {
        let mut cross_count = 0;
        for rank in 0..self.max_rank() - 1 {
            cross_count += self.bilayer_cross_count(graph, rank);
        }
        cross_count
    }

    fn bilayer_cross_count(&self, graph: &StableDiGraph<Vertex, Edge>, rank: usize) -> usize {
        // find initial edge order
        let north = &self[rank];
        let south = &self[rank + 1];
        let mut len = south.len();
        let mut key_length = 0;
        while len > 0 {
            len /= 10;
            key_length += 1;
        }
        let edge_endpoint_positions = north
            .iter()
            .map(|v| {
                radix_sort(
                    graph
                        .neighbors_directed(*v, Outgoing)
                        .filter(|n| graph[*v].rank.abs_diff(graph[*n].rank) == 1)
                        .filter_map(|n| self.positions.get(&n))
                        .copied()
                        .collect(),
                    key_length,
                )
            })
            .flatten()
            .collect::<Vec<_>>();

        Self::count_crossings(edge_endpoint_positions, south.len())
    }

    fn count_crossings(endpoints: Vec<usize>, south_len: usize) -> usize {
        // build the accumulator tree
        let mut c = 0;
        while 1 << c < south_len {
            c += 1
        }
        let tree_size = (1 << (c + 1)) - 1;
        let first_index = (1 << c) - 1;
        let mut tree = vec![0; tree_size];

        let mut cross_count = 0;

        // traverse through the positions and adjust tree nodes
        for pos in endpoints {
            let mut index = pos + first_index;
            tree[index] += 1;
            while index > 0 {
                // traverse up the tree, incrementing the nodes of the tree
                // each time we visit them.
                //
                // When visiting a left node, add the value of the node on the right to
                // the cross count;
                if index % 2 == 1 {
                    cross_count += tree[index + 1]
                }
                index = (index - 1) / 2;
                tree[index] += 1;
            }
        }
        cross_count
    }
}

impl Deref for Order {
    type Target = Vec<Vec<NodeIndex>>;

    fn deref(&self) -> &Self::Target {
        &self._inner
    }
}

impl DerefMut for Order {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self._inner
    }
}

pub(super) fn insert_dummy_vertices(graph: &mut StableDiGraph<Vertex, Edge>, minimum_length: i32) {
    // find all edges that have slack of greater than 0.
    // and insert dummy vertices
    for edge in graph.edge_indices().collect::<Vec<_>>() {
        if slack(graph, edge, minimum_length) > 0 {
            let (mut tail, head) = graph.edge_endpoints(edge).unwrap();
            // we don't need to remember edges that where removed
            graph.remove_edge(edge);
            for rank in (graph[tail].rank + 1)..graph[head].rank {
                // usize usize::MAX id as reserved value for a dummy vertex
                let mut d = Vertex::default();
                d.is_dummy = true;
                let new = graph.add_node(d);
                graph[new].align = new;
                graph[new].root = new;
                graph[new].sink = new;
                graph[new].rank = rank;
                graph.add_edge(tail, new, Edge::default());
                tail = new;
            }
            graph.add_edge(tail, head, Edge::default()); // add last dummy edge connecting to the head
        }
    }
}

pub(super) fn remove_dummy_vertices(
    graph: &mut StableDiGraph<Vertex, Edge>,
    order: &mut [Vec<NodeIndex>],
) {
    // go through all nodes in topological order
    // see if any outgoing neighbors are dummies
    // follow them until the other non dummy node is found
    // insert old edge
    // remove all dummy nodes
    let vertices = toposort(&*graph, None).unwrap();
    for v in vertices {
        let mut edges = Vec::new();
        for mut n in graph.neighbors_directed(v, Outgoing) {
            if graph[n].is_dummy {
                while graph[n].is_dummy {
                    let dummy_neighbors = graph.neighbors_directed(n, Outgoing).collect::<Vec<_>>();
                    assert_eq!(dummy_neighbors.len(), 1);
                    n = dummy_neighbors[0];
                }
                edges.push((v, n));
            }
        }
        for (tail, head) in edges {
            graph.add_edge(tail, head, Edge::default());
        }
    }
    // remove from order
    for mut l in order {
        *l = l
            .into_iter()
            .filter(|v| !graph[**v].is_dummy)
            .map(|v| *v)
            .collect();
    }
    graph.retain_nodes(|g, v| !g[v].is_dummy);
}

// TODO: Maybe write store all upper neighbors on vertex directly
pub(super) fn ordering(graph: &mut StableDiGraph<Vertex, Edge>) -> Vec<Vec<NodeIndex>> {
    let mut order = init_order(graph);
    println!("c: {}", order.crossings(graph));
    // move downwards for crossing reduction
    let order = reduce_crossings_bilayer_sweep(graph, order);
    order._inner
}

fn init_order(graph: &StableDiGraph<Vertex, Edge>) -> Order {
    fn dfs(
        v: NodeIndex,
        order: &mut Vec<Vec<NodeIndex>>,
        graph: &StableDiGraph<Vertex, Edge>,
        visited: &mut HashSet<NodeIndex>,
    ) {
        if !visited.contains(&v) {
            visited.insert(v);
            order[graph[v].rank as usize].push(v);
            graph
                .neighbors_directed(v, Outgoing)
                .for_each(|n| dfs(n, order, graph, visited))
        }
    }

    let max_rank = graph
        .node_weights()
        .map(|v| v.rank as usize)
        .max_by(|r1, r2| r1.cmp(&r2))
        .expect("Got invalid ranking");
    let mut order = vec![Vec::new(); max_rank + 1];
    let mut visited = HashSet::new();

    // build initial order via dfs
    graph
        .node_indices()
        .for_each(|v| dfs(v, &mut order, &graph, &mut visited));

    Order::new(order)
}

fn reduce_crossings_bilayer_sweep(graph: &StableDiGraph<Vertex, Edge>, mut order: Order) -> Order {
    let mut best_crossings = order.crossings(graph);
    let mut last_best = 0;
    let mut best = order.clone();
    for i in 0.. {
        order = wmedian(graph, i % 2 == 0, &best);
        let crossings = order.crossings(graph);
        if crossings < best_crossings {
            best_crossings = crossings;
            best = order.clone();
            last_best = 0;
        } else {
            last_best += 1;
        }
        if last_best == 4 {
            return best;
        }
    }
    best
}

fn wmedian(graph: &StableDiGraph<Vertex, Edge>, move_down: bool, current: &Order) -> Order {
    let mut o = vec![Vec::new(); current.max_rank()];
    let mut positions = current.positions.clone();
    let dir: Vec<usize> = if move_down {
        o[0] = current._inner[0].clone();
        (1..current.max_rank()).collect()
    } else {
        o[current.max_rank() - 1] = current._inner[current.max_rank() - 1].clone();
        (0..current.max_rank() - 1).collect()
    };

    for rank in dir {
        o[rank] = current[rank].clone();
        //println!("{:?}", self.order[rank]);
        o[rank].sort_by(|a, b| {
            barycenter(graph, *a, move_down, &positions)
                .total_cmp(&barycenter(graph, *b, move_down, &positions))
        });
        o[rank].iter().enumerate().for_each(|(pos, v)| {
            positions.insert(*v, pos);
        });
    }

    Order::new(o)
}

fn median_value(
    graph: &StableDiGraph<Vertex, Edge>,
    vertex: NodeIndex,
    move_down: bool,
    positions: &HashMap<NodeIndex, usize>,
) -> f64 {
    let neighbors = if move_down {
        graph.neighbors_directed(vertex, Incoming)
    } else {
        graph.neighbors_directed(vertex, Outgoing)
    };
    // Only look at direct neighbors
    let mut adjacent = neighbors
        .filter(|n| graph[vertex].rank.abs_diff(graph[*n].rank) == 1)
        .map(|n| *positions.get(&n).unwrap())
        .collect::<Vec<_>>();

    adjacent.sort();

    let length_p = adjacent.len();
    let m = length_p / 2;
    if length_p == 0 {
        f64::MAX
    } else if length_p % 2 == 1 {
        adjacent[m] as f64
    } else if length_p == 2 {
        (adjacent[0] + adjacent[1]) as f64 / 2.
    } else {
        let left = adjacent[m - 1] - adjacent[0];
        let right = adjacent[length_p - 1] - adjacent[m];
        (adjacent[m - 1] * right + adjacent[m] * left) as f64 / (left + right) as f64
    }
}

fn barycenter(
    graph: &StableDiGraph<Vertex, Edge>,
    vertex: NodeIndex,
    move_down: bool,
    positions: &HashMap<NodeIndex, usize>,
) -> f64 {
    let neighbors: Vec<_> = if move_down {
        graph.neighbors_directed(vertex, Incoming).collect()
    } else {
        graph.neighbors_directed(vertex, Outgoing).collect()
    };

    if neighbors.len() == 0 {
        return 0.;
    }

    // Only look at direct neighbors
    let adjacent = neighbors
        .into_iter()
        .filter(|n| graph[vertex].rank.abs_diff(graph[*n].rank) == 1)
        .map(|n| *positions.get(&n).unwrap())
        .collect::<Vec<usize>>();

    adjacent.iter().sum::<usize>() as f64 / adjacent.len() as f64
}
