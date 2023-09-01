mod tests;
use std::collections::{ HashSet, HashMap };
use std::fmt::Display;
use std::ops::{Deref, DerefMut};
use std::time::Instant;

use petgraph::Direction::{Incoming, Outgoing};
use petgraph::algo::toposort;
use petgraph::stable_graph::{StableDiGraph, NodeIndex};
use petgraph::visit::IntoNeighborsDirected;

use crate::util::lookup_maps::NodeLookupMap;
use crate::{impl_slack, util};
use crate::util::{IterDir, radix_sort};
use crate::{util::layers::Layers, impl_layer_graph};
use crate::util::traits::LayerGraph;
use crate::graphs::p1_layering::Vertex as P1Vertex;
use crate::graphs::p1_layering::Edge as P1Edge;
use crate::graphs::p1_layering::traits::Slack;

use super::p1_layering::FeasibleTree;
use super::p1_layering::traits::Ranked;
use super::p3_calculate_coordinates::MinimalCrossings;

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Vertex {
    id: usize,
    rank: u32,
    pos: usize,
    is_dummy: bool,
    upper_neighbors: Vec<NodeIndex>, // store positions of neighbors on adjacent ranks, since we need to acces them very often
    lower_neighbors: Vec<NodeIndex>
}

impl Vertex {
    fn new(rank: u32) -> Self {
        Self {
            id: 0,
            rank,
            pos: 0,
            is_dummy: false,
            upper_neighbors: Vec::new(),
            lower_neighbors: Vec::new(),
        }
    }
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            id: 0,
            rank: 0,
            pos: 0,
            is_dummy: false,
            upper_neighbors: Vec::new(),
            lower_neighbors: Vec::new(),
        }
    }
}

impl Ranked for Vertex {
    fn rank(&self) -> i32 {
        self.rank as i32
    }
}

impl From<P1Vertex> for Vertex {
    fn from(vertex: P1Vertex) -> Self {
        Vertex { id: vertex.id, rank: vertex.rank as u32, pos: 0, is_dummy: false, upper_neighbors: Vec::new(), lower_neighbors: Vec::new() }
    }
}

pub struct Edge;

impl From<P1Edge> for Edge {
    fn from(_: P1Edge) -> Self {
        Self
    }
}

impl Default for Edge {
    fn default() -> Self {
        Edge
    }
}

// later test if its better to access neighbors via graph or to store them separately
#[derive(Clone)]
struct Order {
    _inner: Vec<Vec<NodeIndex>>,
    positions: HashMap<NodeIndex, usize>,
    crossings: usize,
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
        Self{ 
            _inner: layers,
            positions,
            crossings: usize::MAX,
        }
    }

    fn new_empty(max_rank: usize) -> Self {
        Self {
            _inner: vec![Vec::new(); max_rank],
            positions: HashMap::new(),
            crossings: usize::MAX,
        }
    }

    fn max_rank(&self) -> usize {
        self.len()
    }

    fn exchange(&mut self, i: usize, rank: usize) {
        self[rank].swap(i, i + 1);
        self.positions.insert(self[rank][i], i);
        self.positions.insert(self[rank][i + 1], i + 1);
    }

    fn crossing(&self, v: NodeIndex, w: NodeIndex, graph: &StableDiGraph<Vertex, Edge>, rank: usize) -> usize {
        let mut crossings = 0;
        if rank > 0 {
            let mut v_adjacent = graph[v].upper_neighbors.iter().map(|n| *self.positions.get(n).unwrap()).collect::<Vec<_>>();
            let mut w_adjacent = graph[w].upper_neighbors.iter().map(|n| *self.positions.get(n).unwrap()).collect::<Vec<_>>();
            v_adjacent.sort();
            w_adjacent.sort();
            crossings += Self::cross_count(&v_adjacent, &w_adjacent);
        }
        crossings
    }

    fn cross_count(v_adjacent: &[usize], w_adjacent: &[usize]) -> usize {
        let mut all_crossings = 0;
        let mut k = 0;
        for i in v_adjacent {
            let i = *i;
            let mut crossings = k;
            while k < w_adjacent.len() && w_adjacent[k] < i {
                let j = w_adjacent[k];
                if i > j {
                    crossings += 1;
                }
                k += 1;
            }
            all_crossings += crossings;
        }

        all_crossings
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
        let edge_endpoint_positions = north.iter()
            .map(|v| radix_sort(
                graph.neighbors_directed(*v, Outgoing)
                            .filter_map(|n| self.positions.get(&n))
                            .copied()
                            .collect(), key_length)
            ).flatten()
            .collect::<Vec<_>>();

        Self::count_crossings(edge_endpoint_positions, south.len())
    }

    fn count_crossings(endpoints: Vec<usize>, south_len: usize) -> usize {
        // build the accumulator tree
        let mut c = 0;
        while 1 << c < south_len { c += 1 };
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
                if index % 2 == 1 { cross_count += tree[index + 1] }
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

// Note: Do we need to remember the minimum length?
pub struct InsertDummyVertices {
    pub(crate) graph: StableDiGraph<Vertex, Edge>,
    pub(crate) minimum_length: i32
}

impl_slack!(InsertDummyVertices, Vertex, Edge);

// Keep Vertex
impl InsertDummyVertices {
    pub fn prepare_for_initial_ordering(mut self) -> ReduceCrossings {
        self.insert_dummy_vertices();
        self.fill_in_neighbors();

        ReduceCrossings { graph: self.graph }
    }
    fn insert_dummy_vertices(&mut self) {
        // find all edges that have slack of greater than 0.
        // and insert dummy vertices
        for edge in self.graph.edge_indices().collect::<Vec<_>>() {
            if self.slack(edge) > 0 {
                let (mut tail, head) = self.graph.edge_endpoints(edge).unwrap();
                // we don't need to remember edges that where removed
                self.graph.remove_edge(edge);
                for rank in (self.graph[tail].rank + 1)..self.graph[head].rank {
                    // usize usize::MAX id as reserved value for a dummy vertex
                    let d = Vertex::new(rank);
                    let new = self.graph.add_node(d);
                    self.graph.add_edge(tail, new, Edge);
                    tail = new;
                }
                self.graph.add_edge(tail, head, Edge); // add last dummy edge connecting to the head
            }
        }
    }

    fn fill_in_neighbors(&mut self) {
        for v in self.graph.node_indices().collect::<Vec<_>>() {
            let Vertex { rank, .. } = self.graph[v];
            // fill in upper neighbors
            if rank > 0 {
                let mut walker = self.graph.neighbors_directed(v, Incoming).detach();
                while let Some(n) = walker.next_node(&self.graph) {
                    assert!(self.graph[n].rank == rank - 1);
                    self.graph[v].upper_neighbors.push(n);
                }
            }
            // fill in lower neighbors
            let mut walker = self.graph.neighbors_directed(v, Outgoing).detach();
            while let Some(n) = walker.next_node(&self.graph) {
                assert!(self.graph[n].rank == rank + 1);
                self.graph[v].lower_neighbors.push(n);
            }
        }
    }
}

impl From<FeasibleTree> for InsertDummyVertices {
    fn from(ft: FeasibleTree) -> Self {
        let FeasibleTree { graph, minimum_length } = ft;

        Self {
            graph: graph.map(|_id, n| (*n).into(), |_id, e| (*e).into()),
            minimum_length,
        } 
    }
}

pub struct ReduceCrossings {
    graph: StableDiGraph<Vertex, Edge>,
}

impl Deref for ReduceCrossings {
    type Target = StableDiGraph<Vertex, Edge>;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl ReduceCrossings {
    pub fn ordering<T: Default>(mut self) -> MinimalCrossings<T> {
        let order = self.init_order();
        let max_iterations = 24;
        let min_improvement = 0.05;
        // move downwards for crossing reduction
        let order = self.reduce_crossings_bilayer_sweep(order);
        // self.reduce_crossings(max_iterations, min_improvement, IterDir::Forward);
        // self.reduce_crossings(max_iterations, min_improvement, IterDir::Backward);
        // move upwards for crossing reduction
        let Self { graph } = self;
        let g = graph.map(|_, w| if w.is_dummy { None } else { Some(T::default()) }, |_, _| usize::default());
        let layers = Layers::new(order._inner, &g);
        MinimalCrossings::new(layers, g)
    }

    fn init_order(&mut self) -> Order {
        fn dfs(
            v: NodeIndex, 
            order: &mut Vec<Vec<NodeIndex>>,
            graph: &StableDiGraph<Vertex, Edge>, 
            visited: &mut HashSet<NodeIndex>) {
            if !visited.contains(&v) {
                visited.insert(v);
                order[graph[v].rank as usize].push(v);
                graph.neighbors_directed(v, Outgoing).for_each(|n| dfs(n, order, graph, visited)) 
            }
        }

        let max_rank = self.graph.node_weights()
            .map(|v| v.rank as usize)
            .max_by(|r1, r2| r1.cmp(&r2))
            .expect("Got invalid ranking");
        let mut order = vec![Vec::new(); max_rank + 1];
        let mut visited = HashSet::new();

        // build initial order via dfs
        self.graph.node_indices()
            .filter(|v| self.graph[*v].rank == 0)
            .for_each(|v| dfs(v, &mut order, &self.graph, &mut visited));

        // fill in initial position
        order.iter().for_each(|r| 
            r.iter().enumerate().for_each(|(pos, v)| self.graph[*v].pos = pos)
        );
        Order::new(order)
    }

    fn reduce_crossings_bilayer_sweep(&mut self, mut order: Order) -> Order {
        let mut best_crossings = order.crossings(&self);
        println!("bc: {best_crossings}");
        let mut last_best = 0;
        let mut best = order.clone();
        for i in 0.. {
            order = self.wmedian(i % 2 == 0, &order);
            let crossings = order.crossings(&self);
            if crossings < best_crossings {
                best_crossings = crossings;
                println!("bc: {best_crossings}");
                best = order.clone();
                last_best = 0; 
            } else {
                last_best += 1;
            }
            if last_best == 4 {
                return best;
            }
        }
            /*
        for i in 0..24 {
            order = self.wmedian(i % 2 == 0, &order);
            self.transpose(&mut order, IterDir::Forward);
            let crossings = order.crossings(&self);
            if crossings < best_crossings {
                println!("{crossings}");
                best_crossings = crossings;
                best = order.clone();
            }
        }
            */
        best
    }

    fn wmedian(&self, move_down: bool, current: &Order) -> Order {
        let dir = if move_down { IterDir::Forward  } else { IterDir::Backward };
        let mut o = vec![Vec::new(); current.max_rank()];
        let mut positions = HashMap::new();

        for rank in util::iterate(dir, current.max_rank()) {
            o[rank] = current[rank].clone();
            //println!("{:?}", self.order[rank]);
            o[rank].sort_by(|a, b| 
                            self.median_value(*a, move_down, &positions)
                                .cmp(&self.median_value(*b, move_down, &positions)));
            o[rank].iter().enumerate().for_each(|(pos, v)| { positions.insert(*v, pos); });
        }

        Order::new(o)
    }

    fn median_value(&self, vertex: NodeIndex, move_down: bool, positions: &HashMap<NodeIndex, usize>) -> usize {
        let neighbors = if move_down { 
            &self.graph[vertex].upper_neighbors 
        } else { 
            &self.graph[vertex].lower_neighbors 
        };
        let mut adjacent = neighbors.iter()
            .map(|n| *positions.get(n).unwrap())
            .collect::<Vec<_>>();

        adjacent.sort();

        let length_p = adjacent.len();
        let m = length_p / 2;
        if length_p == 0 {
            usize::MAX
        } else if length_p % 2 == 1 {
            adjacent[m]
        } else if length_p == 2 {
            (adjacent[0] + adjacent[1]) / 2
        } else {
            let left = adjacent[m - 1] - adjacent[0];
            let right = adjacent[length_p - 1] - adjacent[m];
            (adjacent[m - 1] * right + adjacent[m] * left) / (left + right) 
        }
    }

    fn transpose(&self, order: &mut Order, direction: IterDir) {
        let mut improved = true;
        Self::cross_count_old(order, &self.graph);
        while improved {
            let initial_crossings = order.crossings as f64;
            improved = false;
            for rank in util::iterate(direction, order.max_rank()) {
                for i in 0..order[rank].len() - 1 {
                    let v = order[rank][i];
                    let w = order[rank][i + 1];
                    if order.crossing(v, w, &self.graph, rank) > order.crossing(w, v, &self.graph, rank) {
                        //println!("swapped");
                        improved = true;
                        order.exchange(i, rank);
                    }
                }
            }
            /*
            Self::cross_count_old(order, &self.graph);
            let new_crossings = order.crossings as f64;
            if new_crossings / initial_crossings > 0.99 { 
                println!("didn't improve");
                improved = false; 
            } else {
                println!("{}", new_crossings / initial_crossings);
            }
            */
        }
    }

    fn cross_count_old(order: &mut Order, graph: &StableDiGraph<Vertex, Edge>) -> usize {
            order.crossings = 0;
            for rank in 0..order.max_rank() {
                for i in 0..order[rank].len() - 1 {
                    let v = order[rank][i];
                    let w = order[rank][i + 1];
                    order.crossings += order.crossing(v, w, graph, rank);
                }
            }
            order.crossings
    }

}
