use std::collections::{HashSet, HashMap, VecDeque};

use petgraph::Direction::*;
use petgraph::graph::Node;
use petgraph::stable_graph::{StableDiGraph, NodeIndex};
use petgraph::visit::{IntoNeighbors, IntoNeighborsDirected};

use crate::{util::layers::Layers, impl_layer_graph};
use crate::util::traits::LayerGraph;

use self::ranks::Ranks;

// create from input graph
struct UnlayeredGraph<T: Default> {
    graph: StableDiGraph<Option<T>, usize>
}

impl<T: Default> UnlayeredGraph<T> {
    // create feasible tree
    fn create_feasible_tree(self) -> FeasibleTree<T> {
        // initialize ranks
        let mut ranks = self.initial_ranking();
        let num_vertices = self.graph.node_count();

        let mut tight_tree = HashSet::from([self.graph.edge_indices().next().unwrap()]);
        while tight_tree.len() - 1 < num_vertices {
            
        }




        FeasibleTree { graph: self.graph }
    }

    fn initial_ranking(&self) -> Ranks {
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

        Ranks::new(ranks, &self.graph) 
    }

}

mod ranks {
    use std::{collections::HashMap, ops::Index};

    use petgraph::stable_graph::{NodeIndex, StableDiGraph};

    /// Ranks of the vertices of a graph.
    /// Needs to contain all the vertices of a graph
    #[derive(Debug)]
    pub struct Ranks {
        _inner: HashMap<NodeIndex, isize>
    }

    impl Ranks {
        pub fn new<T>(ranks: HashMap<NodeIndex, isize>, graph: &StableDiGraph<Option<T>, usize>) -> Self {
            assert!(Self::is_valid(&ranks, graph));
            Ranks { _inner: ranks}
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
        fn slack(&self, tail: NodeIndex, head: NodeIndex) -> isize {
            self._inner.get(&tail).unwrap() - self._inner.get(&head).unwrap()
        }
    }

    impl Index<NodeIndex> for Ranks {
        type Output = isize;

        fn index(&self, index: NodeIndex) -> &Self::Output {
           self._inner.get(&index).unwrap()
        }
    }
}

struct FeasibleTree<T: Default> {
    graph: StableDiGraph<Option<T>, usize>,
}

#[cfg(test)]
mod tests {
    use petgraph::stable_graph::StableDiGraph;

    use super::UnlayeredGraph;

    fn create_test_graph<T: Default>() -> StableDiGraph<Option<T>, usize> {
        petgraph::stable_graph::StableDiGraph::from_edges(&[(0, 1), (1, 2), (2, 3), (0, 4), (0, 5), (4, 6), (5, 6), (3, 7), (6, 7)])
    }

    #[test]
    fn test_initial_ranking() {
        let graph = create_test_graph::<isize>();
        let ul_graph = UnlayeredGraph { graph };
        let ranks = ul_graph.initial_ranking();
        dbg!(&ranks);
    }
}