use std::collections::{HashSet, HashMap, VecDeque};
use std::hash::Hash;

use petgraph::Direction::*;
use petgraph::graph::Node;
use petgraph::stable_graph::{StableDiGraph, NodeIndex};
use petgraph::visit::EdgeRef;

use self::ranks::Ranks;

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
    fn new(graph: StableDiGraph<Option<T>, usize>, ranks: Ranks) -> Self {
        Self { graph, ranks }
    }

    // Returns true if exactly one vertex is a member of the tree.
    fn is_incident_edge(&self, tail: NodeIndex, head: NodeIndex) -> bool {
        self.graph.contains_node(tail) ^ self.graph.contains_node(head)
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
        let mut nodes = self.graph.node_indices();
        let mut tree = Tree::new();
        
        while self.tight_tree_dfs(&mut tree, nodes.next().unwrap(), &mut HashSet::new()) < num_nodes {
            let (tail, head) = self.graph.edge_indices()
                                         .filter_map(|e| self.graph.edge_endpoints(e))
                                         .filter(|(tail, head)| !tree.contains_edge(*tail, *head) && self.is_incident_edge(*tail, *head))
                                         .min_by(|a, b| self.ranks.slack(a.0, a.1).cmp(&self.ranks.slack(b.0, b.1))).unwrap();
            let mut delta = self.ranks.slack(tail, head);
            if tree.contains_node(&head) {
                delta = -delta;
            }
            for vertex in tree.vertices.iter() {
                self.ranks.update(*vertex, delta);
            }
        }
        FeasibleTreeBuilder { _inner: self.graph, ranks: self.ranks }
    }
    
    fn tight_tree_dfs(&self, tree: &mut Tree<NodeIndex>, vertex: NodeIndex, visited: &mut HashSet<(NodeIndex, NodeIndex)>) -> usize {
        let mut node_count = 1;
        tree.add_node(vertex);
        for head in self.graph.neighbors_directed(vertex, Outgoing) {
            if visited.insert((vertex, head)) {
                if tree.contains_edge(vertex, head) {
                    node_count += self.tight_tree_dfs(tree, head, visited);
                } else if !tree.contains_node(&head) && self.ranks.slack(vertex, head) == 0 {
                    tree.add_node(head);
                    tree.add_edge(vertex, head);
                    node_count += self.tight_tree_dfs(tree, head, visited);
                }
            }
        }
        for tail in self.graph.neighbors_directed(vertex, Incoming) {
            if visited.insert((tail, vertex)) {
                if tree.contains_edge(tail, vertex) {
                    node_count += self.tight_tree_dfs(tree, tail, visited);
                } else if !tree.contains_node(&tail) && self.ranks.slack(tail, vertex) == 0 {
                    tree.add_node(tail);
                    tree.add_edge(tail, vertex);
                    node_count += self.tight_tree_dfs(tree, tail, visited);
                }
            }
        }

    node_count
    }


}

struct Tree<T> {
    vertices: HashSet<T>,
    edges: HashSet<(T, T)>,
}

impl<T: Eq + PartialEq + Hash> Tree<T> {
    fn new() -> Self {
        Self { 
            vertices: HashSet::new(), 
            edges: HashSet::new() 
        }
    }

    fn add_node(&mut self, node: T) {
        self.vertices.insert(node);
    }

    fn add_edge(&mut self, tail: T, head: T) {
        self.edges.insert((tail, head));
    }

    fn contains_node(&self, node: &T) -> bool {
        self.vertices.contains(node)
    }

    fn contains_edge(&self, tail: T, head: T) -> bool {
        self.edges.contains(&(tail, head))
    }
}

#[cfg(test)]
mod test_dfs {
    use core::num;
    use std::collections::HashSet;

    use petgraph::{stable_graph::NodeIndex, visit::IntoNodeIdentifiers};


    use crate::graphs::p1_create_layers::{Tree, UnlayeredGraph, TightTreeBuilder};

    use super::tests::create_test_graph;

    #[test]
    fn test_dfs() {
        // let mut tree = Tree::<NodeIndex>{ vertices: HashSet::new(), edges: HashSet::new() };
        // let graph = create_test_graph::<i32>();
        // let number_of_nodes = graph.node_count();
        // let graph2 = create_test_graph::<i32>();
        // let mut nodes = graph2.node_indices().collect::<Vec<_>>().into_iter();
        // let ranks = UnlayeredGraph{ graph }.initial_ranking(1);
        // while builder.tight_tree_dfs(&mut tree, nodes.next().unwrap(), &mut HashSet::new()) < number_of_nodes {

        // }
        // assert!(tree.edges.len() < number_of_nodes);
        // assert_eq!(tree.vertices.len(), number_of_nodes);
        // builder.make_tight();
    }
}

pub(crate) struct FeasibleTreeBuilder<T: Default> {
    _inner: StableDiGraph<Option<T>, usize>,
    ranks: Ranks
}

impl<T: Default> FeasibleTreeBuilder<T> {
    pub(crate) fn init_cutvalues(self) -> FeasibleTree<T> {
        FeasibleTree { graph: StableDiGraph::new() }
    }
}

mod ranks {
    use std::{collections::HashMap, ops::Index};

    use petgraph::stable_graph::{NodeIndex, StableDiGraph};

    /// Ranks of the vertices of a graph.
    /// Needs to contain all the vertices of a graph
    #[derive(Debug, Clone)]
    pub struct Ranks {
        _inner: HashMap<NodeIndex, isize>,
        minimum_length: usize
    }

    impl Ranks {
        pub fn new<T>(ranks: HashMap<NodeIndex, isize>, graph: &StableDiGraph<Option<T>, usize>, minimum_length: usize) -> Self {
            assert!(Self::is_valid(&ranks, graph));
            Ranks { _inner: ranks, minimum_length }
        }
        
        fn is_valid<T>(ranks: &HashMap<NodeIndex, isize>, graph: &StableDiGraph<Option<T>, usize>) -> bool {
            for v in graph.node_indices() {
                if !ranks.contains_key(&v) {
                    return false;
                }
            } 

            true
        }

        // tail = predecessor, head = successor
        pub(super) fn slack(&self, tail: NodeIndex, head: NodeIndex) -> isize {
            self._inner.get(&head).unwrap() - self._inner.get(&tail).unwrap() - self.minimum_length as isize
        }

        pub(super) fn get_minimum_length(&self) -> usize {
            self.minimum_length
        }

        pub(super) fn update(&mut self, vertex: NodeIndex, delta: isize) {
            self._inner.entry(vertex).and_modify(|rank| *rank += delta);
        }
    }

    impl Index<NodeIndex> for Ranks {
        type Output = isize;

        fn index(&self, index: NodeIndex) -> &Self::Output {
           self._inner.get(&index).unwrap()
        }
    }
}

pub(crate) struct FeasibleTree<T: Default> {
    graph: StableDiGraph<Option<T>, usize>,
}

#[cfg(test)]
mod tests {
    use petgraph::stable_graph::StableDiGraph;

    use super::UnlayeredGraph;

    pub(crate) fn create_test_graph<T: Default>() -> StableDiGraph<Option<T>, usize> {
        petgraph::stable_graph::StableDiGraph::from_edges(&[(0, 1), (1, 2), (2, 3), (0, 4), (0, 5), (4, 6), (5, 6), (3, 7), (6, 7)])
    }

    #[test]
    fn test_initial_ranking() {
        let graph = create_test_graph::<isize>();
        let ul_graph = UnlayeredGraph { graph };
        let ranks = ul_graph.initial_ranking(1);
        dbg!(&ranks.ranks);
    }
}