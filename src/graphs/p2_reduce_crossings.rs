mod tests;
use std::collections::{ HashSet, VecDeque };
use std::ops::{Deref, DerefMut};
use std::time::Instant;

use petgraph::Direction::{Incoming, Outgoing};
use petgraph::algo::toposort;
use petgraph::stable_graph::{StableDiGraph, NodeIndex};
use petgraph::visit::IntoNeighborsDirected;

use crate::{impl_slack, util};
use crate::util::{IterDir, radix_sort};
use crate::{util::layers::Layers, impl_layer_graph};
use crate::util::traits::LayerGraph;
use crate::graphs::p1_layering::Vertex as P1Vertex;
use crate::graphs::p1_layering::Edge as P1Edge;
use crate::graphs::p1_layering::traits::Slack;

use super::p1_layering::FeasibleTree;
use super::p1_layering::traits::Ranked;

#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct Vertex {
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

pub(crate) struct Edge;

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
struct Order {
    _inner: Vec<Vec<NodeIndex>>,
    crossings: usize,
}

impl Order {
    fn new(layers: Vec<Vec<NodeIndex>>) -> Self {
        Self{ 
            _inner: layers,
            crossings: usize::MAX,
        }
    }

    fn new_empty(max_rank: usize) -> Self {
        Self {
            _inner: vec![Vec::new(); max_rank],
            crossings: usize::MAX,
        }
    }

    fn max_rank(&self) -> usize {
        self.len()
    }

    fn exchange(&mut self, i: usize, rank: usize) {
        self[rank].swap(i, i + 1);
    }

    fn get_neighbor_positions(&self, neighbors: &[NodeIndex], rank: usize) -> Vec<usize> {
        neighbors.iter()
            .filter_map(|n| self[rank].iter().position(|v| v == n))
            .collect()
    }


    fn crossing(&self, v: NodeIndex, w: NodeIndex, graph: &StableDiGraph<Vertex, Edge>, rank: usize) -> usize {
        // check upper lower neighbor crossings or only the one for the direction?
        let mut crossings = 0;
        if rank > 0 {
            let v_adjacent = self.get_neighbor_positions(&graph[v].upper_neighbors, rank - 1);
            let w_adjacent = self.get_neighbor_positions(&graph[w].upper_neighbors, rank - 1);
            crossings += Self::cross_count(&v_adjacent, &w_adjacent);
        }
        /*
        if rank < self.max_rank() - 1 {
            // let neighbors_ms = Instant::now();
            let v_adjacent = self.get_neighbor_positions(&graph[v].lower_neighbors, rank + 1);
            let w_adjacent = self.get_neighbor_positions(&graph[w].lower_neighbors, rank + 1);
            // println!("lower_neighbors: {}ms", neighbors_ms.elapsed().as_millis());
            // let cross_count_ms = Instant::now();
            crossings += Self::cross_count(&v_adjacent, &w_adjacent);
            // println!("crosscount: {}ms", cross_count_ms.elapsed().as_millis());
        }
         */
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
    fn prepare_for_initial_ordering(mut self) -> InitOrdering {
        self.insert_dummy_vertices();
        self.fill_in_neighbors();

        InitOrdering { graph: self.graph }
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


pub struct InitOrdering {
    graph: StableDiGraph<Vertex, Edge>,
}

impl InitOrdering {
    fn init_order(mut self) -> ReduceCrossings {
        fn dfs(v: NodeIndex, order: &mut Vec<Vec<NodeIndex>>, graph: &StableDiGraph<Vertex, Edge>, visited: &mut HashSet<NodeIndex>) {
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
        let mut order = vec![Vec::new(); max_rank];
        let mut visited = HashSet::new();

        // build initial order via dfs
        self.graph.node_indices()
            .filter(|v| self.graph[*v].rank == 0)
            .for_each(|v| dfs(v, &mut order, &self.graph, &mut visited));

        // fill in initial position
        order.iter().for_each(|r| 
            r.iter().enumerate().for_each(|(pos, v)| self.graph[*v].pos = pos)
        );

        ReduceCrossings { 
            graph: self.graph, 
            order: Order::new(order),
        }
    }
}

pub(crate) struct ReduceCrossings {
    graph: StableDiGraph<Vertex, Edge>,
    order: Order,
}

impl ReduceCrossings {
    fn ordering(&mut self) {
        let max_iterations = 24;
        let min_improvement = 0.05;
        // move downwards for crossing reduction
        self.reduce_crossings(max_iterations, min_improvement, IterDir::Forward);
        self.reduce_crossings(max_iterations, min_improvement, IterDir::Backward);
        // move upwards for crossing reduction
    }

    fn reduce_crossings(&mut self, max_iterations: usize, min_improvement: f64, direction: IterDir) {
        let mut median_ms = 0;
        let mut transpose_ms = 0;
        let now = Instant::now();
        for i in 0..max_iterations {
            let median = Instant::now();
            let mut order = self.wmedian(i % 2 == 0);
            median_ms += median.elapsed().as_millis();

            let transpose = Instant::now();
            self.transpose(&mut order, direction);
            transpose_ms += transpose.elapsed().as_millis();
            // abort if we don't improve significantly anymore
            if (Self::cross_count_old(&mut order, &self.graph) as f64 / Self::cross_count_old(&mut self.order, &self.graph) as f64) < min_improvement {
                return;
            }
            if Self::cross_count_old(&mut order, &self.graph) < Self::cross_count_old(&mut self.order, &self.graph) {
                // println!("{}", order.crossings);
                self.order = order;
            }
        }
    }

    fn wmedian(&mut self, move_down: bool) -> Order {
        let dir = if move_down { IterDir::Forward  } else { IterDir::Backward };
        let mut new_order = Order::new_empty(self.order.max_rank());

        for rank in util::iterate(dir, self.order.max_rank()) {
            new_order[rank] = self.order[rank].clone();
            new_order[rank].sort_by(|a, b| self.median_value(*a, move_down, rank).cmp(&self.median_value(*b, move_down, rank)));
        }

        new_order
    }

    fn median_value(&self, vertex: NodeIndex, move_down: bool, rank: usize) -> usize {
        let neighbors = if move_down { &self.graph[vertex].upper_neighbors } else { &self.graph[vertex].lower_neighbors };
        let adjacent = self.order.get_neighbor_positions(neighbors, rank);
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
            let right = adjacent[length_p] - adjacent[m];
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
                        improved = true;
                        order.exchange(i, rank);
                    }
                }
            }
            Self::cross_count_old(order, &self.graph);
            let new_crossings = order.crossings as f64;
            if new_crossings / initial_crossings > 0.99 { 
                println!("didn't improve");
                improved = false; 
            } else {
                println!("{}", new_crossings / initial_crossings);
            }
        }
    }

    fn crossings(&self) -> usize {
        let mut cross_count = 0;
        for i in 0..self.order.max_rank() - 1 {
            cross_count += self.bilayer_cross_count(&self.order[i], &self.order[i + 1]);
        }
        cross_count 
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

    fn bilayer_cross_count(&self, north: &[NodeIndex], south: &[NodeIndex]) -> usize {
        // find initial edge order
        let mut len = south.len();
        let mut key_length = 0;
        while len > 0 {
            len /= 10;
            key_length += 1;
        }
        let edge_endpoint_positions = north.iter()
            .map(|v| radix_sort(
                self.graph.neighbors_directed(*v, Outgoing)
                            .map(|n| self.graph[n].pos)
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
