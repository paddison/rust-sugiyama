mod tests;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use petgraph::Direction::{Incoming, Outgoing};
use petgraph::algo::toposort;
use petgraph::stable_graph::{StableDiGraph, NodeIndex};
use petgraph::visit::IntoNeighborsDirected;

use crate::impl_slack;
use crate::util::layers::IterDir;
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
    is_dummy: bool,
    upper_neighbors: Vec<NodeIndex>, // store positions of neighbors on adjacent ranks, since we need to acces them very often
    lower_neighbors: Vec<NodeIndex>
}

impl Vertex {
    #[cfg(test)]
    fn new(rank: u32) -> Self {
        Self {
            id: 0,
            rank,
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
        Vertex { id: vertex.id, rank: vertex.rank as u32, is_dummy: false, upper_neighbors: Vec::new(), lower_neighbors: Vec::new() }
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
        if rank < self.max_rank() - 1 {
            let v_adjacent = self.get_neighbor_positions(&graph[v].lower_neighbors, rank + 1);
            let w_adjacent = self.get_neighbor_positions(&graph[w].lower_neighbors, rank + 1);
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
                    let d = Vertex { id: 0, rank, is_dummy: true, upper_neighbors: Vec::new(), lower_neighbors: Vec::new() };
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
    fn init_order(self) -> ReduceCrossings {
        let mut order = Vec::new();

        // fill in initial position
        for v in toposort(&self.graph, None).unwrap() {
            let weight = &self.graph[v];
            let rank = weight.rank as usize;

            while order.len() <= rank {
                order.push(Vec::new());
            }

            order[rank].push(v);
        }

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
        println!("{:?}", self.order.crossings);
    }

    fn reduce_crossings(&mut self, max_iterations: usize, min_improvement: f64, direction: IterDir) {
        for i in 0..max_iterations {
            let mut order = self.wmedian(i % 2 == 0);
            self.transpose(&mut order, direction);
            // abort if we don't improve significantly anymore
            if (Self::crossings(&mut order, &self.graph) as f64 / Self::crossings(&mut self.order, &self.graph) as f64) < min_improvement {
                return;
            }
            if Self::crossings(&mut order, &self.graph) < Self::crossings(&mut self.order, &self.graph) {
                self.order = order;
            }
        }
    }

    fn wmedian(&mut self, move_down: bool) -> Order {
        let dir = if move_down { IterDir::Forward  } else { IterDir::Backward };
        let mut new_order = Order::new_empty(self.order.max_rank());

        for rank in Layers::iterate(dir, self.order.max_rank()) {
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
        while improved {
            improved = false;
            for rank in Layers::iterate(direction, order.max_rank()) {
                for i in 0..order[rank].len() - 1 {
                    let v = order[rank][i];
                    let w = order[rank][i + 1];
                    if order.crossing(v, w, &self.graph, rank) > order.crossing(w, v, &self.graph, rank) {
                        improved = true;
                        order.exchange(i, rank);
                    }
                }
            }
        }
    }

    fn crossings(order: &mut Order, graph: &StableDiGraph<Vertex, Edge>) -> usize {
        if order.crossings == usize::MAX {
            order.crossings = 0;
            for rank in 0..order.max_rank() {
                for i in 0..order[rank].len() - 1 {
                    let v = order[rank][i];
                    let w = order[rank][i + 1];
                    order.crossings += order.crossing(v, w, graph, rank);
                }
            }
        }

        order.crossings
    }
}