use std::collections::{HashSet, VecDeque};

use petgraph::{stable_graph::{StableDiGraph, NodeIndex, EdgeIndex}, Direction::{Incoming, Outgoing, self}, visit::EdgeRef};

use super::{Vertex, Edge, NeighborhoodInfo};

pub(super) trait Slack {
    fn slack(&self, edge: EdgeIndex) -> i32 {
        let graph = self.graph();
        let (tail, head) = graph.edge_endpoints(edge).unwrap();
        graph[head].rank - graph[tail].rank - self.minimum_length()
    }

    fn graph(&self) -> &StableDiGraph<Vertex, Edge>;
    fn minimum_length(&self) -> i32;
}

#[macro_export]
macro_rules! impl_slack {
    ($t:ty) => {
        impl Slack for $t {
            fn graph(&self) -> &StableDiGraph<Vertex, Edge> {
                &self.graph
            }

            fn minimum_length(&self) -> i32 {
                self.minimum_length
            }
        } 
    };
}
pub(super) trait LowLimDFS {
    fn dfs_low_lim(&mut self, next: NodeIndex, parent: Option<NodeIndex>, max_lim: &mut u32, visited: &mut HashSet<NodeIndex>) {
        visited.insert(next);
        self.graph_mut()[next].lim = *max_lim;
        self.graph_mut()[next].parent = parent;
        while let Some(n) = self.graph_mut().neighbors_undirected(next).detach().next_node(self.graph_mut()) {
            if visited.contains(&n) {
                continue;
            }
            *max_lim -= 1;
            self.dfs_low_lim(n, Some(next), max_lim, visited);
            self.graph_mut()[next].low = *max_lim;
        }
    }
    fn graph_mut(&mut self) -> &mut StableDiGraph<Vertex, Edge>;
    fn graph(&self) -> &StableDiGraph<Vertex, Edge>;
}

#[macro_export]
macro_rules! impl_low_lim_dfs {
    ($t:ty) => {
        impl LowLimDFS for $t {
            fn graph_mut(&mut self) -> &mut StableDiGraph<Vertex, Edge> {
                &mut self.graph
            }

            fn graph(&self) -> &StableDiGraph<Vertex, Edge> {
                &self.graph
            }
        }
    };
}

pub(super) trait CalculateCutValues {
    fn calculate_cut_values(&mut self, mut queue: VecDeque<NodeIndex>) {
        while let Some(vertex) = queue.pop_front() {
            let incoming = self.get_neighborhood_info(vertex, Incoming); 
            let outgoing = self.get_neighborhood_info(vertex, Outgoing); 

            // if we can't calculate cut value yet, or the value is already known
            let (mut incoming, mut outgoing) = match (incoming, outgoing) {
                (Some(inc), Some(out)) => (inc, out),
                _ => continue,
            };

            let missing = match (incoming.missing, outgoing.missing) {
                (Some(u), None) => u,
                (None, Some(v)) => v,
                _ => continue,
            };

            let edge = match self.graph_mut().find_edge(vertex, missing) {
                Some(e) => {
                    // switch direction, if vertex is tail component of edge
                    std::mem::swap(&mut incoming, &mut outgoing);
                    e
                },
                None => self.graph_mut().find_edge(missing, vertex).unwrap()

            };

            self.graph_mut()[edge].cut_value = Some(self.calculate_cut_value(edge, incoming, outgoing));
            // continue traversing tree in direction of edge whose vertex was missing before
            queue.push_back(missing);
        }
    }

    fn calculate_cut_value(&self, edge: EdgeIndex, incoming: NeighborhoodInfo, outgoing: NeighborhoodInfo) -> i32 {
        self.graph()[edge].weight 
        + incoming.non_tree_edge_weight_sum - incoming.cut_value_sum + incoming.tree_edge_weight_sum
        - outgoing.non_tree_edge_weight_sum + outgoing.cut_value_sum - outgoing.tree_edge_weight_sum
    }

    fn get_neighborhood_info(&self, vertex: NodeIndex, direction: Direction) -> Option<NeighborhoodInfo> {
        // return the sum of all cut values,
        // sum of weights of cut value edges
        // missing cut value (only if there is one)
        // sum of weights of edges who are not tree edges
        let mut cut_value_sum = 0;
        let mut tree_edge_weight_sum = 0;
        let mut non_tree_edge_weight_sum = 0;
        let mut missing = None;

        for edge in self.graph().edges_directed(vertex, direction) {
            let (tail, head) = (edge.source(), edge.target());
            let edge = *edge.weight();
            if !edge.is_tree_edge {
                non_tree_edge_weight_sum += edge.weight;
            } else if let Some(cut_value) = edge.cut_value {
                cut_value_sum += cut_value;
                tree_edge_weight_sum += edge.weight;
            } else if missing.is_none() {
                missing = Some(if tail == vertex { head } else { tail });
            } else {
                return None;
            }
        }
        Some(
            NeighborhoodInfo {
                cut_value_sum,
                tree_edge_weight_sum,
                non_tree_edge_weight_sum,
                missing,
            }
        )
    }

    fn graph_mut(&mut self) -> &mut StableDiGraph<Vertex, Edge>;
    fn graph(&self) -> &StableDiGraph<Vertex, Edge>;
}

#[macro_export]
macro_rules! impl_calculate_cut_values {
    ($t:ty) => {
        impl CalculateCutValues for $t {
            fn graph_mut(&mut self) -> &mut StableDiGraph<Vertex, Edge> {
                &mut self.graph
            }

            fn graph(&self) -> &StableDiGraph<Vertex, Edge> {
                &self.graph
            }
        }
    };
}