mod rank;
mod tree;
#[cfg(test)]
mod tests;

use std::collections::{HashSet, HashMap, VecDeque};

use petgraph::Direction::{*, self};
use petgraph::graph::Node;
use petgraph::stable_graph::{StableDiGraph, NodeIndex};
use petgraph::visit::{EdgeRef, IntoNeighborsDirected, IntoNodeIdentifiers};

use crate::util::layers::Layers;

use self::rank::Ranks;
use self::tree::{TighTreeDFS};

use super::p2_reduce_crossings::ProperLayeredGraph;

pub(crate) fn start_layering<T: Default>(graph: StableDiGraph<Option<T>, usize>) -> UnlayeredGraph<T> {
    UnlayeredGraph { graph }
}

// create from input graph
pub(crate) struct UnlayeredGraph<T: Default> {
    graph: StableDiGraph<Option<T>, usize>
}

impl<T: Default> UnlayeredGraph<T> {
    pub(crate) fn initial_ranking(self, minimum_length: usize) -> TightTreeBuilder<T> {
        let mut scanned = HashSet::<(NodeIndex, NodeIndex)>::new();
        let mut ranks = HashMap::<NodeIndex, isize>::new();

        // Sort nodes topologically so we don't need to verify that we've assigned
        // a rank to all incoming neighbors
        // assume graphs contain no circles for now
        for v in petgraph::algo::toposort(&self.graph, None).unwrap() {
            self.graph.neighbors_directed(v, Incoming).for_each(|u| assert!(scanned.contains(&(u, v))));
            
            let rank = self.graph.neighbors_directed(v, Incoming)
                                 .filter_map(|n| ranks.get(&n).and_then(|r| Some(r + 1)))
                                 .max()
                                 .unwrap_or(0);

            for n in self.graph.neighbors_directed(v, Outgoing) {
                scanned.insert((v, n));
            }

            ranks.insert(v, rank);
        }

        let ranks = Ranks::new(ranks, &self.graph, minimum_length);
        TightTreeBuilder { graph: self.graph, ranks }
    }
}

pub(crate) struct TightTreeBuilder<T: Default> {
    graph: StableDiGraph<Option<T>, usize>,
    ranks: Ranks,
}

impl<T: Default> TightTreeBuilder<T> {
    #[cfg(test)]
    fn new(graph: StableDiGraph<Option<T>, usize>, ranks: Ranks) -> Self {
        Self { graph, ranks }
    }

    pub(crate) fn make_tight(mut self) -> FeasibleTreeBuilder<T> {
        // take a random edge to start the tree from
        // split edges into sets consisting of tree and non tree edges.
        // in the beginning, all edges are non tree edges, and they are added
        // with each call to dfs.

        // build a new graph which is a tree. 
        // Remember only edges which where part of the original graph
        // each time we add an edge to the tree, we remove it from the graph
        let num_nodes = self.graph.node_count();
        let mut nodes = self.graph.node_indices().into_iter();
        let mut dfs = TighTreeDFS::new();
        
        while dfs.build_tight_tree(&self.graph, &self.ranks, nodes.next().unwrap(), &mut HashSet::new()) < num_nodes {
            let (tail, head) = self.find_non_tight_edge(&dfs);
            let mut delta = self.ranks.slack(tail, head);

            if dfs.contains_vertex(&head) {
                delta = -delta;
            }

            self.ranks.tighten_edge(&dfs, delta)
        }

        // remove all edges which are contained in tree from graph
        dfs.make_edges_disjoint(&mut self.graph);

        FeasibleTreeBuilder { graph: self.graph, ranks: self.ranks, tree: dfs.into_tree_subgraph() }
    }
    
    fn find_non_tight_edge(&self, tree: &TighTreeDFS) -> (NodeIndex, NodeIndex) {
        self.graph.edge_indices()
            .filter_map(|e| self.graph.edge_endpoints(e))
            .filter(|(tail, head)| !tree.contains_edge(*tail, *head) && tree.is_incident_edge(tail, head))
            .min_by(|a, b| self.ranks.slack(a.0, a.1).cmp(&self.ranks.slack(b.0, b.1))).unwrap()
    }
}


pub(crate) struct FeasibleTreeBuilder<T: Default> {
    graph: StableDiGraph<Option<T>, usize>,
    ranks: Ranks,
    tree: StableDiGraph<Option<T>, usize>,
}

impl<T: Default> FeasibleTreeBuilder<T> {
    fn calculate_cutvalues(&self, mut queue: VecDeque<NodeIndex>, cut_values: &mut HashMap<(NodeIndex, NodeIndex), isize>) {
        while let Some(vertex) = queue.pop_front() {
            // terminate early if all cutvalues are known
            if cut_values.len() == self.tree.edge_count() {
                println!("done early");
                break;
            }
            let (mut cut_values_incoming, mut missing_cut_values_incoming) = 
                self.get_neighborhood_info(vertex, cut_values, Incoming); 
            let (mut cut_values_outgoing, mut missing_cut_values_outgoing) = 
                self.get_neighborhood_info(vertex, cut_values, Outgoing); 
            let (mut incoming, mut outgoing) = (Direction::Incoming, Direction::Outgoing);

            // if we can't calculate cut value yet, or the value is already known
            if missing_cut_values_incoming.len() > 1 || missing_cut_values_outgoing.len() > 1 || 
                missing_cut_values_incoming.len() == 0 && missing_cut_values_outgoing.len() == 0 {
                continue;
            } 

            // switch direction, if vertex is tail component of edge
            let edge = if missing_cut_values_outgoing.len() == 1 {
                std::mem::swap(&mut cut_values_incoming, &mut cut_values_outgoing);
                std::mem::swap(&mut missing_cut_values_incoming, &mut missing_cut_values_outgoing);
                std::mem::swap(&mut incoming, &mut outgoing);
                (vertex, missing_cut_values_incoming[0])
            } else {
                (missing_cut_values_incoming[0], vertex)
            };

            let cut_value = 1 + self.graph.neighbors_directed(vertex, incoming).count() as isize - 
                cut_values_incoming.iter().sum::<isize>() + cut_values_incoming.len() as isize - 
                self.graph.neighbors_directed(vertex, outgoing).count() as isize + 
                cut_values_outgoing.iter().sum::<isize>() - cut_values_outgoing.len() as isize;
            
            cut_values.insert(edge, cut_value);
            // continue traversing tree in direction of edge whose vertex was missing before
            queue.push_back(missing_cut_values_incoming[0]);
        }
    }

    fn update_cutvalues(self, mut cut_values: HashMap<(NodeIndex, NodeIndex), isize>, starting_vertex: NodeIndex) -> FeasibleTree<T> {
        let queue = VecDeque::from([starting_vertex]);
        self.calculate_cutvalues(queue, &mut cut_values);
        FeasibleTree { graph: self.graph, tree: self.tree, ranks: self.ranks, cut_values }
    }

    pub(crate) fn init_cutvalues(self) -> FeasibleTree<T> {
        // assumes all edges have a weight of one
        let mut cut_values = HashMap::<(NodeIndex, NodeIndex), isize>::new();
        let queue = self.leaves();

        // traverse tree inward via breadth first starting from leaves
        self.calculate_cutvalues(queue, &mut cut_values);

        FeasibleTree { graph: self.graph, tree: self.tree, ranks: self.ranks, cut_values }
    }

    fn get_neighborhood_info(
        &self, 
        vertex: NodeIndex, 
        cut_values: &mut HashMap<(NodeIndex, NodeIndex), isize>, 
        direction: Direction
    ) -> (Vec<isize>, Vec<NodeIndex>) {
        let mut cuts = Vec::new(); 
        let mut missing = Vec::new();
        for edge in self.tree.edges_directed(vertex, direction) {
            let (tail, head) = (edge.source(), edge.target());
            if let Some(cut_value) = cut_values.get(&(tail, head)) {
                cuts.push(*cut_value);
            } else {
                missing.push(if tail == vertex { head } else { tail });
            }
        }
        (cuts, missing)
    }

    fn leaves(&self) -> VecDeque<NodeIndex> {
        self.tree.node_indices().filter(|v| self.tree.neighbors_undirected(*v).count() < 2).collect::<VecDeque<_>>()
    }
}

// TODO: make an add an extra type which is used to init low lim values.
pub(crate) struct FeasibleTree<T: Default> {
    graph: StableDiGraph<Option<T>, usize>,
    tree: StableDiGraph<Option<T>, usize>,
    ranks: Ranks,
    pub cut_values: HashMap<(NodeIndex, NodeIndex), isize>,
}

impl<T: Default> FeasibleTree<T> {
    fn rank(mut self) -> ProperLayeredGraph<T> {
        let mut low_lim = self.initialize_low_lim();

        while let Some(edge) = self.leave_edge() {
            // swap edges and calculate cut value
            let swap_edge = self.enter_edge(edge, &low_lim);
            self = self.exchange(&mut low_lim, edge, swap_edge);
            self.update_ranks();
        }

        // TODO maybe destcructure this or turn it into Layers.
        self.ranks.normalize();
        // don't balance ranks since we want maximum width to 
        // give indication about number of parallel processes running
        ProperLayeredGraph::new(Layers::new_empty(), self.graph)
    }

    fn leave_edge(&self) -> Option<(NodeIndex, NodeIndex)> {
        for (edge, cut_value) in self.cut_values.iter() {
            if cut_value < &0 {
                return Some(*edge);
            }
        }
        None
    }

    fn enter_edge(&mut self, edge: (NodeIndex, NodeIndex), low_lim: &HashMap<NodeIndex, TreeData>) -> (NodeIndex, NodeIndex) {
        // find a non-tree edge to replace e.
        // remove e from tree
        // consider all edges going from head to tail component.
        // choose edge with minimum slack.
        let (u, v) = edge;
        let mut u = *low_lim.get(&u).unwrap();
        let mut v = *low_lim.get(&v).unwrap();
        if !(u.lim < v.lim) {
            std::mem::swap(&mut u, &mut v); 
        }

        self.graph.edge_indices()
            .filter_map(|e| self.graph.edge_endpoints(e))
            .filter(|(tail, head)| { 
                Self::is_head_to_tail(low_lim, *tail, *head, u, u.lim < v.lim)
            })
            .min_by(|(tail_a, head_a), (tail_b, head_b)| self.ranks.slack(*tail_a, *head_a).cmp(&self.ranks.slack(*tail_b, *head_b)))
            .unwrap()
    }

    fn exchange(mut self, low_lim: &mut HashMap<NodeIndex, TreeData>, edge: (NodeIndex, NodeIndex), swap_edge: (NodeIndex, NodeIndex)) -> Self {
        // get path connecting the head and tail of swap_edge in the tree
        let (connecting_path, least_common_ancestor) = Self::get_path_in_tree(low_lim, swap_edge);

        // swap edges 
        self.tree.remove_edge(self.tree.find_edge(edge.0, edge.1).unwrap());
        self.tree.add_edge(swap_edge.0, swap_edge.1, usize::default());
        self.graph.remove_edge(self.graph.find_edge(swap_edge.0, swap_edge.1).unwrap());
        // is it a good idea to add the edge that was removed back to the graph or should we keep a separate list of removed edges?
        self.graph.add_edge(edge.0, edge.1, usize::default()); 

        // update cut values
        // the only cut values that need to be updated are those in the path connecting the swap edge in the tree
        // find the first cut value in the path which can be updated,
        // then update all cut values that are missing along the way.
        // the first value which can be updated is alway one of the vertices of the edge that was removed.
        self.remove_outdated_cutvalues(connecting_path, edge);
        // destructure self, since we need to build the tree anew:

        let Self { graph, tree, ranks, cut_values } = self;
        self = FeasibleTreeBuilder { graph, ranks, tree }.update_cutvalues(cut_values, edge.0);

        self.update_low_lim(low_lim, least_common_ancestor);

        self
    }

    fn initialize_low_lim(&self) -> HashMap<NodeIndex, TreeData> {
        // start at arbitrary root node
        let root = self.tree.node_indices().next().unwrap();
        let mut low_lim = HashMap::new();
        self.dfs_low_lim(&mut low_lim, root, 1, None);

        low_lim
    }

    fn dfs_low_lim(&self, low_lim: &mut HashMap<NodeIndex, TreeData>, next: NodeIndex, mut least_lim: usize, parent: Option<NodeIndex>) -> usize {
        low_lim.insert(next, TreeData::default()); // dummy values
        let mut low = least_lim;
        for n in self.tree.neighbors_undirected(next) {
            if low_lim.contains_key(&n) {
                continue;
            }
            least_lim = self.dfs_low_lim(low_lim, n, least_lim, Some(next)) + 1;
            low = low.min(low_lim.get(&n).unwrap().low);
        }
        low_lim.insert(next, TreeData { low, lim: least_lim, parent });

        least_lim
    }

    fn get_path_in_tree(low_lim: &HashMap<NodeIndex, TreeData>, edge: (NodeIndex, NodeIndex)) -> (Vec<NodeIndex>, NodeIndex) {
        let (w, x) = edge;
        let w_data = low_lim.get(&w).unwrap();
        let x_data = low_lim.get(&x).unwrap();
        let mut path_w_l = VecDeque::new();
        let mut path_l_x = VecDeque::new();
        // follow path back until least common ancestor is found
        // record path from w to l
        let least_common_ancestor = loop {
            let l = w_data.parent.unwrap();
            path_w_l.push_back(l);
            let l_data = low_lim.get(&l).unwrap();
            if l_data.low <= w_data.lim && x_data.lim <= l_data.lim {
                break l;
            }
        };

        // record path from x to l
        loop {
            let l = x_data.parent.unwrap();
            let l_data = low_lim.get(&l).unwrap();
            if l_data.low <= w_data.lim && x_data.lim <= l_data.lim {
                assert_eq!(l, least_common_ancestor); // for debugging check that roots are identical
                break
            }

            path_l_x.push_front(l);
        }

        path_w_l.append(&mut path_l_x);

        (path_w_l.into_iter().collect::<Vec<_>>(), least_common_ancestor)
    }

    fn is_head_to_tail(low_lim: &HashMap<NodeIndex, TreeData>, tail: NodeIndex, head: NodeIndex, u: TreeData, root_is_in_head: bool) -> bool {
        // edge needs to go from head to tail. e.g. tail neads to be in head component, and head in tail component
        let tail = low_lim.get(&tail).unwrap();
        let head = low_lim.get(&head).unwrap();
        // check if head is in tail component
        root_is_in_head == (u.low <= head.lim && head.lim <= u.lim) &&
        // check if tail is in head component
        root_is_in_head != (u.low <= tail.lim && tail.lim <= u.lim)
    }

    fn remove_outdated_cutvalues(&mut self, connecting_path: Vec<NodeIndex>, removed_edge: (NodeIndex, NodeIndex)) {
        // starting from the first node, we know all adjacent cutvalues except for one.
        // thus we should be able to update every cut value efficiently by going through the path.
        // the last thing we need to do is calculate the cut value for the edge that was added.
        // remove all the cutvalues on the path:
        self.cut_values.remove(&removed_edge);
        for (tail, head) in connecting_path[..connecting_path.len() - 1].iter().copied().zip(connecting_path[1..].iter().copied()) {
            if self.cut_values.remove(&(tail, head)).is_none() {
                self.cut_values.remove(&(head, tail));
            }
        }
    }

    fn update_ranks(&mut self) {
        let node = self.tree.node_identifiers().next().unwrap();
        let mut new_ranks = HashMap::from([(node, 0)]);
        // start at arbitrary node and traverse the tree
        let mut queue = VecDeque::from([self.tree.node_identifiers().next().unwrap()]);
        let minimum_length = self.ranks.get_minimum_length() as isize;

        while let Some(parent) = queue.pop_front() {
            for n in self.tree.neighbors_directed(parent, Incoming) {
                if new_ranks.contains_key(&n) {
                    continue;
                }
                new_ranks.insert(n, new_ranks.get(&parent).unwrap() - minimum_length);
                queue.push_back(n);
            }

            for n in self.tree.neighbors_directed(parent, Outgoing) {
                if new_ranks.contains_key(&n) {
                    continue;
                }
                new_ranks.insert(n, new_ranks.get(&parent).unwrap() + minimum_length);
                queue.push_back(n);
            }
        }
        self.ranks = Ranks::new(new_ranks, &self.tree, self.ranks.get_minimum_length())
    }

    fn update_low_lim(&self, low_lim: &mut HashMap<NodeIndex, TreeData>, least_common_ancestor: NodeIndex) {
        let lca_data = low_lim.get(&least_common_ancestor).unwrap();
        let mut visited = match lca_data.parent {
            Some(parent) => HashSet::from([parent]),
            None => HashSet::new()
        };
        let mut max_lim = lca_data.lim;
        self.reverse_dfs_low_lim(low_lim, least_common_ancestor, lca_data.parent, &mut max_lim, &mut visited)
    }

    /// Used to update low, lim and parent values, by traversing the subtree under parent.
    /// The difference to initialzing the low lim values is that we will decrement lim instead of incrementing it.
    fn reverse_dfs_low_lim(&self, low_lim: &mut HashMap<NodeIndex, TreeData>, next: NodeIndex, parent: Option<NodeIndex>, max_lim: &mut usize, visited: &mut HashSet<NodeIndex>) {
        visited.insert(next);
        low_lim.entry(next).and_modify(|e| { e.lim = *max_lim; e.parent = parent; });

        for n in self.tree.neighbors_undirected(next) {
            if visited.contains(&n) {
                continue;
            }
            *max_lim -= 1;
            self.reverse_dfs_low_lim(low_lim, n, Some(next), max_lim, visited);
            low_lim.entry(n).and_modify(|e| e.low = *max_lim);
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct TreeData {
    lim: usize,
    low: usize,
    parent: Option<NodeIndex>
}

impl TreeData {
    #[cfg(test)]
    fn new(lim: usize, low: usize, parent: Option<NodeIndex>) -> Self {
        Self { lim, low, parent }
    }
}

impl Default for TreeData {
    fn default() -> Self {
        TreeData{ lim: 0, low: 0, parent: None }
    }
}

