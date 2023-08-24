mod tight_tree_dfs;
mod traits;
#[cfg(test)]
mod tests;

use std::collections::{HashSet, VecDeque};

use petgraph::Direction::{*, self};
use petgraph::stable_graph::{StableDiGraph, NodeIndex, EdgeIndex};
use petgraph::visit::IntoNodeIdentifiers;

use crate::{impl_slack, impl_low_lim_dfs, impl_calculate_cut_values};

use self::traits::{LowLimDFS, CalculateCutValues, Slack};
use self::tight_tree_dfs::{TightTreeDFS};

#[derive(Clone, Copy)]
pub(crate) struct Vertex {
    id: usize,
    rank: i32,
    low: u32,
    lim: u32,
    parent: Option<NodeIndex>,
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            id: 0,
            rank: 0,
            low: 0,
            lim: 0,
            parent: None,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Edge {
    weight: i32,
    cut_value: Option<i32>,
    is_tree_edge: bool,
}

impl Default for Edge {
    fn default() -> Self {
        Self {
            weight: 1,
            cut_value: None,   
            is_tree_edge: false,
        }
    }
}

struct NeighborhoodInfo {
    cut_value_sum: i32,
    tree_edge_weight_sum: i32,
    non_tree_edge_weight_sum: i32,
    missing: Option<NodeIndex>,
}


pub(crate) fn start(edges: &[(u32, u32)], minimum_length: u32) -> UnlayeredGraph {
    let graph = StableDiGraph::<Vertex, Edge>::from_edges(edges);
    UnlayeredGraph { graph, minimum_length: minimum_length as i32 }
}


pub(crate) struct UnlayeredGraph {
    graph: StableDiGraph<Vertex, Edge>,
    minimum_length: i32
}

impl UnlayeredGraph {
    pub(crate) fn init_rank(self) -> InitialRanks {
        let Self { mut graph, minimum_length } = self;

        // Sort nodes topologically so we don't need to verify that we've assigned
        // a rank to all incoming neighbors
        // assume graphs contain no circles for now
        for v in petgraph::algo::toposort(&graph, None).unwrap() {
            let rank = graph.neighbors_directed(v, Incoming)
                                 .map(|n| graph[n].rank + self.minimum_length)
                                 .max();

            if let Some(rank) = rank {
                graph[v].rank = rank;
            }
        }

        InitialRanks { graph, minimum_length }
    }
}

pub(crate) struct InitialRanks {
    graph: StableDiGraph<Vertex, Edge>,
    minimum_length: i32
}

impl_slack!(InitialRanks);

impl InitialRanks {
    pub(crate) fn make_tight(mut self) -> TightTree {
        // let Self { mut graph, minimum_length } = self;

        // take a random edge to start the tree from
        // split edges into sets consisting of tree and non tree edges.
        // in the beginning, all edges are non tree edges, and they are added
        // with each call to dfs.

        // build a new graph which is a tree. 
        // Remember only edges which where part of the original graph
        // each time we add an edge to the tree, we remove it from the graph
        let num_nodes = self.graph.node_count();
        let mut nodes = self.graph.node_indices().collect::<Vec<_>>().into_iter();
        let mut dfs = TightTreeDFS::new();
        
        while dfs.tight_tree(&self, nodes.next().unwrap(), &mut HashSet::new()) < num_nodes {
            let edge = self.find_non_tight_edge(&dfs);
            let (_, head) = self.graph.edge_endpoints(edge).unwrap();
            let mut delta = self.slack(edge);

            if dfs.contains_vertex(&head) {
                delta = -delta;
            }

            self.tighten_edge(&dfs, delta)
        }

        self.mark_tree_edges(dfs);

        TightTree { graph: self.graph, minimum_length: self.minimum_length }
    }

    fn find_non_tight_edge(&self, dfs: &TightTreeDFS) -> EdgeIndex {
        self.graph.edge_indices()
            .filter(|e| !dfs.contains_edge(*e) && dfs.is_incident_edge(e, &self.graph))
            .min_by(|e1, e2| self.slack(*e1).cmp(&self.slack(*e2))).unwrap()
    }

    fn tighten_edge(&mut self, dfs: &TightTreeDFS, delta: i32) {
        for e in &dfs.edges {
            let (tail, head) = self.graph.edge_endpoints(*e).unwrap();
            self.graph[tail].rank += delta;
            self.graph[head].rank += delta;
        }
    }

    fn mark_tree_edges(&mut self, dfs: TightTreeDFS) {
        for e in dfs.edges {
            self.graph[e].is_tree_edge = true;
        }
    }
}

pub(crate) struct TightTree {
    graph: StableDiGraph<Vertex, Edge>,
    minimum_length: i32,
}

impl CalculateCutValues for TightTree {
    fn graph_mut(&mut self) -> &mut StableDiGraph<Vertex, Edge> {
        &mut self.graph
    }

    fn graph(&self) -> &StableDiGraph<Vertex, Edge> {
        &self.graph
    }
}

impl TightTree {
    pub(crate) fn init_cutvalues(mut self) -> InitTightTreeData {
        // TODO: check if it is faster to collect tree edges or to do unecessary iterations
        // let tree_edges = self.graph.edge_indices().filter(|e| self.graph[*e].is_tree_edge).collect::<HashSet<_>>();
        let queue = self.leaves();
        // traverse tree inward via breadth first starting from leaves
        self.calculate_cut_values(queue);
        InitTightTreeData { graph: self.graph, minimum_length: self.minimum_length }
    }

    fn leaves(&self) -> VecDeque<NodeIndex> {
        self.graph.node_identifiers()
                  .filter(|v| 
                    1 == self.graph.edges_directed(*v, Incoming)
                              .chain(self.graph.edges_directed(*v, Outgoing))
                              .filter(|e| e.weight().is_tree_edge)
                              .count())
                  .collect::<VecDeque<_>>()
    }
}


pub(crate) struct InitTightTreeData {
    graph: StableDiGraph<Vertex, Edge>,
    minimum_length: i32,
}

impl_low_lim_dfs!(InitTightTreeData);

impl InitTightTreeData {
    pub(crate) fn init_low_lim(mut self) -> FeasibleTree {
        // start at arbitrary root node
        let root = self.graph.node_indices().next().unwrap();
        let mut max_lim = self.graph.node_count() as u32;
        self.dfs_low_lim(root, None, &mut max_lim, &mut HashSet::new());
        FeasibleTree { graph: self.graph, minimum_length: self.minimum_length }
    }
}

pub(crate) struct FeasibleTree {
    graph: StableDiGraph<Vertex, Edge>,
    minimum_length: i32,
}

impl_slack!(FeasibleTree);

impl FeasibleTree {
    pub(crate) fn rank(mut self) {

        while let Some(edge) = self.leave_edge() {
            // swap edges and calculate cut value
            let swap_edge = self.enter_edge(edge);
            self = self.exchange(edge, swap_edge).update();
        }

        // don't balance ranks since we want maximum width to 
        // give indication about number of parallel processes running
        self.normalize();
    }

    fn leave_edge(&self) -> Option<EdgeIndex> {
        for edge in self.graph.edge_indices() {
            if let Some(cut_value) = self.graph[edge].cut_value {
                if cut_value < 0 {
                    return Some(edge);
                }
            }
        }
        None
    }

    fn enter_edge(&mut self, edge: EdgeIndex) -> EdgeIndex {
        // find a non-tree edge to replace e.
        // remove e from tree
        // consider all edges going from head to tail component.
        // choose edge with minimum slack.
        let (mut u, mut v) = self.graph.edge_endpoints(edge).map(|(t, h)| (self.graph[t], self.graph[h])).unwrap();

        if !(u.lim < v.lim) {
            std::mem::swap(&mut u, &mut v); 
        }

        self.graph.edge_indices()
            .filter(|e| !self.graph[*e].is_tree_edge)
            .filter(|e| self.is_head_to_tail(*e, u, u.lim < v.lim))
            .min_by(|e1, e2| self.slack(*e1).cmp(&self.slack(*e2)))
            .unwrap()
    }

    fn is_head_to_tail(&self, edge: EdgeIndex, u: Vertex, root_is_in_head: bool) -> bool {
        // edge needs to go from head to tail. e.g. tail neads to be in head component, and head in tail component
        let (tail, head) = self.graph.edge_endpoints(edge).map(|(t, h)| (self.graph[t], self.graph[h])).unwrap();
        // check if head is in tail component
        root_is_in_head == (u.low <= head.lim && head.lim <= u.lim) &&
        // check if tail is in head component
        root_is_in_head != (u.low <= tail.lim && tail.lim <= u.lim)
    }

    fn exchange(mut self, removed_edge: EdgeIndex, swap_edge: EdgeIndex) -> UpdateTree {
        // get path connecting the head and tail of swap_edge in the tree
        let (connecting_path, least_common_ancestor) = self.get_path_in_tree(swap_edge);

        // swap edges 
        self.graph[removed_edge].is_tree_edge = false;
        self.graph[swap_edge].is_tree_edge = true;

        // destructure self, since we need to build the tree anew:
        let Self { graph, minimum_length } = self;
        UpdateTree { graph, minimum_length, connecting_path, removed_edge, least_common_ancestor }
    }

    fn get_path_in_tree(&self, edge: EdgeIndex) -> (Vec<EdgeIndex>, NodeIndex) {
        assert!(!self.graph[edge].is_tree_edge);
        let (w_id, x_id)  = self.graph.edge_endpoints(edge).unwrap();
        let (w, x) = (self.graph[w_id], self.graph[x_id]);
        let mut path_w_l = Vec::new();
        let mut path_l_x = VecDeque::new();
        // follow path back until least common ancestor is found
        // record path from w to l
        let mut l_id = w_id;
        let least_common_ancestor = loop {
            let l = self.graph[l_id];
            path_w_l.push(self.graph.find_edge_undirected(l_id, l.parent.unwrap()).unwrap().0);
            if l.low <= w.lim && x.lim <= l.lim {
                break l_id;
            }
            l_id = l.parent.unwrap();
        };

        // record path from x to l
        let mut l_id = x_id;
        while l_id != least_common_ancestor {
            let parent = self.graph[l_id].parent.unwrap();
            path_l_x.push_front(self.graph.find_edge_undirected(l_id, parent).unwrap().0);
            l_id = parent;
        }

        (path_w_l.into_iter().chain(path_l_x.into_iter()).collect::<Vec<_>>(), least_common_ancestor)
    }

    fn normalize(&mut self) {
        let min_rank = self.graph.node_identifiers().map(|v| self.graph[v].rank).min().unwrap();
        for v in self.graph.node_weights_mut() {
            v.rank -= min_rank;
        }
    }
}

struct UpdateTree {
    graph: StableDiGraph<Vertex, Edge>,
    minimum_length: i32,
    connecting_path: Vec<EdgeIndex>,
    removed_edge: EdgeIndex,
    least_common_ancestor: NodeIndex,
}

impl_calculate_cut_values!(UpdateTree);
impl_low_lim_dfs!(UpdateTree);

impl UpdateTree {
    fn update(mut self) -> FeasibleTree {
        self.update_cutvalues();
        self.update_low_lim();
        self.update_ranks();
        FeasibleTree { graph: self.graph, minimum_length: self.minimum_length }
    }

    fn update_cutvalues(&mut self) {
        self.remove_outdated_cutvalues();
        let queue = VecDeque::from([self.graph.edge_endpoints(self.removed_edge).unwrap().0]);
        self.calculate_cut_values(queue);
    }

    fn remove_outdated_cutvalues(&mut self) {
        // starting from the first node, we know all adjacent cutvalues except for one.
        // thus we should be able to update every cut value efficiently by going through the path.
        // the last thing we need to do is calculate the cut value for the edge that was added.
        // remove all the cutvalues on the path:
        self.graph[self.removed_edge].cut_value = None;
        for edge in &self.connecting_path {
            self.graph[*edge].cut_value = None;
        }
    }

    fn update_low_lim(&mut self) {
        let parent = self.graph[self.least_common_ancestor].parent;
        let mut visited = match &parent {
            Some(parent) => HashSet::from([*parent]),
            None => HashSet::new()
        };
        let mut max_lim = self.graph[self.least_common_ancestor].lim;
        self.dfs_low_lim(self.least_common_ancestor, parent, &mut max_lim, &mut visited);
    }

    fn update_ranks(&mut self) {
        let node = self.graph.node_identifiers().next().unwrap();
        let mut visited = HashSet::from([node]);
        self.graph[node].rank = 0;
        let mut queue = VecDeque::from([node]);

        while let Some(parent) = queue.pop_front() {
            self.update_neighbor_ranks(parent, Outgoing, 1, &mut queue, &mut visited);
            self.update_neighbor_ranks(parent, Incoming, -1, &mut queue, &mut visited);
        }
    }

    fn update_neighbor_ranks(&mut self, parent: NodeIndex, direction: Direction, coefficient: i32, queue: &mut VecDeque<NodeIndex>,  visited: &mut HashSet<NodeIndex>) {
        while let Some((edge, other)) = self.graph.neighbors_directed(parent, direction).detach().next(&self.graph) {
            if !self.graph[edge].is_tree_edge || visited.contains(&other) {
                continue;
            }
            self.graph[other].rank = self.graph[parent].rank + self.minimum_length * coefficient;
            queue.push_back(other);
            visited.insert(other);
        }
    }
}