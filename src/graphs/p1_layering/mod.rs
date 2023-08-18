mod rank;
mod tree;
#[cfg(test)]
mod tests;

use std::collections::{HashSet, HashMap, VecDeque};

use petgraph::Direction::*;
use petgraph::stable_graph::{StableDiGraph, NodeIndex};
use petgraph::visit::{EdgeRef, IntoNeighborsDirected};

use self::rank::Ranks;
use self::tree::Tree;

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
        let mut tree = Tree::new();
        
        // TODO: Do this starting from the node in the topmost layer
        while self.tight_tree_dfs(&mut tree, nodes.next().unwrap(), &mut HashSet::new()) < num_nodes {
            let (tail, head) = self.find_non_tight_edge(&tree);
            let mut delta = self.ranks.slack(tail, head);

            if tree.contains_vertex(&head) {
                delta = -delta;
            }

            self.ranks.tighten_edge(&tree, delta)
        }

        // remove all edges which are contained in tree from graph
        self.graph.retain_edges(|graph, edge| {
            let (tail, head) = graph.edge_endpoints(edge).unwrap();
            !tree.contains_edge(tail, head)
        });

        FeasibleTreeBuilder { graph: self.graph, ranks: self.ranks, tree }
    }
    
    fn find_non_tight_edge(&self, tree: &Tree) -> (NodeIndex, NodeIndex) {
        self.graph.edge_indices()
            .filter_map(|e| self.graph.edge_endpoints(e))
            .filter(|(tail, head)| !tree.contains_edge(*tail, *head) && tree.is_incident_edge(tail, head))
            .min_by(|a, b| self.ranks.slack(a.0, a.1).cmp(&self.ranks.slack(b.0, b.1))).unwrap()
    }

    fn tight_tree_dfs(&self, tree: &mut Tree, vertex: NodeIndex, visited: &mut HashSet<(NodeIndex, NodeIndex)>) -> usize {
        // start from topmost nodes.
        // then for each topmost node add nodes to tree until done. Then continue with next node until no more nodes are found.
        let mut node_count = 1;
        if !tree.contains_vertex(&vertex) {
            tree.add_vertex(vertex);
        }
        for connected_edge in self.graph.edges_directed(vertex, Incoming).chain(self.graph.edges_directed(vertex, Outgoing)) {
            let (tail, head) = (connected_edge.source(), connected_edge.target());
            // find out if the other vertex of the edge is the head or the tail
            let other = if connected_edge.source() == vertex { head } else { tail };

            if !visited.contains(&(tail, head)) {
                visited.insert((tail, head));
                if tree.contains_edge(tail, head) {
                    node_count += self.tight_tree_dfs(tree, other, visited);
                } else if !tree.contains_vertex(&other) && self.ranks.slack(tail, head) == 0 {
                    tree.add_vertex(other);
                    tree.add_edge(tail, head);
                    node_count += self.tight_tree_dfs(tree, other, visited);
                }
            }
        }
        node_count
    }
}


pub(crate) struct FeasibleTreeBuilder<T: Default> {
    graph: StableDiGraph<Option<T>, usize>,
    ranks: Ranks,
    tree: Tree,
}

impl<T: Default> FeasibleTreeBuilder<T> {
    pub(crate) fn init_cutvalues(self) -> FeasibleTree<T> {
        let mut queue = VecDeque::new();
        let mut cut_values = HashMap::<(NodeIndex, NodeIndex), isize>::new();
        // assumes all edges have a weight of one
        // start calculating cutvalues from leaves inward:
        for leave in self.tree.leaves() {
            // get the connecting edge:
            let (tail, head) = self.tree.connected_edges(leave).next().unwrap();
            let is_tail = leave == tail;
            if is_tail {
                let opposite = head;
                let cut_value = (1 + self.graph.neighbors_directed(leave, Outgoing).count()) as isize - self.graph.neighbors_directed(leave, Incoming).count() as isize;
                cut_values.insert((tail, head), cut_value);
                queue.push_back(opposite);
            } else {
                let opposite = tail;
                let cut_value = (1 + self.graph.neighbors_directed(leave, Incoming).count()) as isize - self.graph.neighbors_directed(leave, Outgoing).count() as isize;
                cut_values.insert((tail, head), cut_value);
                queue.push_back(opposite);
            }
        }

        while let Some(vertex) = queue.pop_front() {
            let (inc_cut_values, inc_missing) = self.get_neighborhood_info(vertex, &mut cut_values, true); 
            let (out_cut_values, out_missing) = self.get_neighborhood_info(vertex, &mut cut_values, false); 
            if inc_missing.len() == 1 && out_missing.len() == 0  {
                // calculate remaining cut values and continue traversing tree
                let cut_value = 1 + self.graph.neighbors_directed(vertex, Incoming).count() as isize - inc_cut_values.iter().sum::<isize>() + inc_cut_values.len() as isize - 
                                self.graph.neighbors_directed(vertex, Outgoing).count() as isize + out_cut_values.iter().sum::<isize>() - out_cut_values.len() as isize;
                cut_values.insert((inc_missing[0], vertex), cut_value);
                queue.push_back(inc_missing[0]);
            } else if inc_missing.len() == 0 && out_missing.len() == 1 {
                // vertex is tail
                let cut_value = 1 + self.graph.neighbors_directed(vertex, Outgoing).count() as isize + inc_cut_values.iter().sum::<isize>() - inc_cut_values.len() as isize - 
                                self.graph.neighbors_directed(vertex, Incoming).count() as isize - out_cut_values.iter().sum::<isize>() + out_cut_values.len() as isize;
                cut_values.insert((vertex, out_missing[0]), cut_value);
                queue.push_back(out_missing[0]);
            } else if inc_missing.len() == 0 && out_missing.len() == 0 {
                continue;
            } else {
                println!("push back on queue");
                queue.push_back(vertex);
            } 
        }

        FeasibleTree { graph: self.graph, tree: self.tree, ranks: self.ranks, cut_values }
    }

    fn get_neighborhood_info(&self, vertex: NodeIndex, cut_values: &mut HashMap<(NodeIndex, NodeIndex), isize>, is_incoming: bool) -> (Vec<isize>, Vec<NodeIndex>) {
        let mut cuts = Vec::new(); 
        let mut missing = Vec::new();
        let neighbors = if is_incoming { self.tree.incoming(vertex) } else { self.tree.outgoing(vertex) };
        for n in neighbors {
            let (tail, head) = if is_incoming { (n, vertex) } else { (vertex, n) };
            if let Some(cut_value) = cut_values.get(&(tail, head)) {
                cuts.push(*cut_value);
            } else {
                missing.push(n);
            }
        }
        (cuts, missing)
    }
}

pub(crate) struct FeasibleTree<T: Default> {
    graph: StableDiGraph<Option<T>, usize>,
    tree: Tree,
    ranks: Ranks,
    pub cut_values: HashMap<(NodeIndex, NodeIndex), isize>,
}