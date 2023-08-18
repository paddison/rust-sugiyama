use std::collections::{ HashSet, hash_set::Iter, VecDeque };

use petgraph::{stable_graph::{NodeIndex, StableDiGraph, Neighbors, NodeIndices}, Direction::{Incoming, Outgoing}, visit::EdgeRef};

use super::rank::Ranks;

#[derive(Debug)]
pub(super) struct TighTreeDFS {
    pub(super) vertices: HashSet<NodeIndex>,
    pub(super) edges: HashSet<(NodeIndex, NodeIndex)>,
}

impl TighTreeDFS {
    pub(super) fn new() -> Self {
        Self {
            vertices: HashSet::new(),
            edges: HashSet::new(),
        }
    }

    pub(super) fn contains_vertex(&self, vertex: &NodeIndex) -> bool {
        self.vertices.contains(&(*vertex).into())
    }

    pub(super) fn contains_edge(&self, tail: NodeIndex, head: NodeIndex) -> bool {
        self.edges.contains(&(tail, head))
    }

    pub(super) fn vertices(&self) -> Iter<'_, NodeIndex> {
        self.vertices.iter()
    }

    /// Returns true if exactly one vertex is a member of the tree.
    pub(super) fn is_incident_edge(&self, tail: &NodeIndex, head: &NodeIndex) -> bool {
        self.vertices.contains(tail) ^ self.vertices.contains(head)
    }

    pub(super) fn make_edges_disjoint<T>(&self, graph: &mut StableDiGraph<Option<T>, usize>) {
        graph.retain_edges(|graph, edge| {
            let (tail, head) = graph.edge_endpoints(edge).unwrap();
            !self.contains_edge(tail, head)
        });
    }

    pub(super) fn build_tight_tree<T>(&mut self, graph: &StableDiGraph<Option<T>, usize>, ranks: &Ranks, vertex: NodeIndex, visited: &mut HashSet<(NodeIndex, NodeIndex)>) -> usize {
        // start from topmost nodes.
        // then for each topmost node add nodes to tree until done. Then continue with next node until no more nodes are found.
        let mut node_count = 1;
        if !self.vertices.contains(&vertex) {
            self.vertices.insert(vertex);
        }
        for connected_edge in graph.edges_directed(vertex, Incoming).chain(graph.edges_directed(vertex, Outgoing)) {
            let (tail, head) = (connected_edge.source(), connected_edge.target());
            // find out if the other vertex of the edge is the head or the tail
            let other = if connected_edge.source() == vertex { head } else { tail };

            if !visited.contains(&(tail, head)) {
                visited.insert((tail, head));
                if self.edges.contains(&(tail, head)) {
                    node_count += self.build_tight_tree(graph, ranks, other, visited);
                } else if !self.vertices.contains(&other) && ranks.slack(tail, head) == 0 {
                    self.vertices.insert(other);
                    self.edges.insert((tail, head));
                    node_count += self.build_tight_tree(graph, ranks, other, visited);
                }
            }
        }
        node_count
    }

    pub(super) fn into_tree_subgraph<T>(self) -> StableDiGraph<Option<T>, usize> {
        StableDiGraph::from_edges(self.edges.iter())
    }

    #[cfg(test)]
    pub(super) fn from_edges(edges: &[(usize, usize)]) -> Self {
        let mut tree = Self::new();
        for (tail, head) in edges {
            tree.vertices.insert(NodeIndex::new(*tail));
            tree.vertices.insert(NodeIndex::new(*head));
            tree.edges.insert((NodeIndex::new(*tail), NodeIndex::new(*head)));
        }
        tree
    }
}

#[derive(Debug)]
pub(super) struct TreeSubgraph<T> {
    pub(crate) graph: StableDiGraph<Option<T>, usize>,
}

impl<T> TreeSubgraph<T> {
    /// Adds an edge to the tree.
    /// Will panic if it doesn't contain the tail and head vertices.
    pub(super) fn node_indices(&self) -> NodeIndices<Option<T>> {
        self.graph.node_indices()
    }

    pub(super) fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    pub(super) fn vertice_count(&self) -> usize {
        self.graph.node_count()
    }

    pub(super) fn leaves(&self) -> VecDeque<NodeIndex> {
        self.graph.node_indices().filter(|v| self.graph.neighbors_undirected(*v).count() < 2).collect::<VecDeque<_>>()
    }

    pub(super) fn incoming(&self, vertex: NodeIndex) -> Neighbors<usize> {
        self.graph.neighbors_directed(vertex, Incoming)
    }

    pub(super) fn outgoing(&self, vertex: NodeIndex) -> Neighbors<usize> {
        self.graph.neighbors_directed(vertex, Outgoing)
    }

    pub(super) fn neighbors(&self, vertex: NodeIndex) -> Neighbors<usize> {
        self.graph.neighbors_undirected(vertex)
    }
}


#[cfg(test)]
mod tests {
    mod tight_tree_dfs {
        use std::collections::HashSet;

        use crate::graphs::p1_layering::{tree::TighTreeDFS, start_layering, tests::{create_test_graph, create_tight_tree_builder_non_tight_ranking}};


        #[test]
        fn test_dfs_start_from_root() {
            let mut dfs = TighTreeDFS::new();
            let tight_tree_builder = start_layering(create_test_graph::<i32>()).initial_ranking(1);
            let number_of_nodes = tight_tree_builder.graph.node_count();
            dfs.build_tight_tree(&tight_tree_builder.graph, &tight_tree_builder.ranks, 0.into(), &mut HashSet::new());
            assert_eq!(dfs.edges.len(), number_of_nodes - 1);
            assert_eq!(dfs.vertices.len(), number_of_nodes);
        }

        #[test]
        fn test_dfs_start_not_from_root() {
            let mut dfs = TighTreeDFS::new();
            let tight_tree_builder = start_layering(create_test_graph::<i32>()).initial_ranking(1);
            let number_of_nodes = tight_tree_builder.graph.node_count();
            dfs.build_tight_tree(&tight_tree_builder.graph, &tight_tree_builder.ranks, 4.into(), &mut HashSet::new());
            assert_eq!(dfs.edges.len(), number_of_nodes - 1);
            assert_eq!(dfs.vertices.len(), number_of_nodes);
        }

        #[test]
        fn test_dfs_non_tight_ranking() {
            let mut dfs = TighTreeDFS::new();
            let mut tight_tree_builder = create_tight_tree_builder_non_tight_ranking::<i32>();
            let number_of_nodes = tight_tree_builder.graph.node_count();
            dfs.build_tight_tree(&tight_tree_builder.graph, &tight_tree_builder.ranks, 0.into(), &mut HashSet::new());
            assert_eq!(dfs.edges.len(), number_of_nodes - 2);
            assert_eq!(dfs.vertices.len(), number_of_nodes - 1);

            // run steps of algorithm manually
            let (tail, head) = tight_tree_builder.find_non_tight_edge(&dfs);
            assert_eq!(head.index(), 8);
            assert_eq!(tail.index(), 7);
            let delta = tight_tree_builder.ranks.slack(tail, head);

            tight_tree_builder.ranks.tighten_edge(&dfs, delta);
            // build again
            dfs.build_tight_tree(&tight_tree_builder.graph, &tight_tree_builder.ranks, 0.into(), &mut HashSet::new());
            assert_eq!(dfs.edges.len(), number_of_nodes - 1);
            assert_eq!(dfs.vertices.len(), number_of_nodes);
        }
    }
}