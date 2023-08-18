use std::collections::{ HashSet, hash_set::Iter };

use petgraph::{stable_graph::{NodeIndex, StableDiGraph}, Direction::{Incoming, Outgoing}, visit::EdgeRef};

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

    // fn into_tree_subgraph(self) -> TreeSubgraph {
        // let graph = StableDiGraph::from_edges(self.edges.iter());;
    // }
}

#[derive(Debug)]
pub(super) struct TreeSubgraph {
    vertices: HashSet<Vertex>,
    edges: HashSet<(NodeIndex, NodeIndex)>,
}

impl TreeSubgraph {
    pub(super) fn new() -> Self {
        Self { 
            vertices: HashSet::new(), 
            edges: HashSet::new() 
        }
    }

    pub(super) fn add_vertex(&mut self, node: NodeIndex) {
        self.vertices.insert(Vertex::new(node));
    }

    /// Adds an edge to the tree.
    /// Will panic if it doesn't contain the tail and head vertices.
    pub(super) fn add_edge(&mut self, tail: NodeIndex, head: NodeIndex) {
        assert!(self.vertices.contains(&tail.into()));
        assert!(self.vertices.contains(&head.into()));

        // update tail
        let mut v_tail = self.vertices.take(&tail.into()).unwrap();
        v_tail.outgoing.insert(head);
        self.vertices.insert(v_tail);

        // update head
        let mut v_head = self.vertices.take(&head.into()).unwrap();
        v_head.incoming.insert(tail);
        self.vertices.insert(v_head);

        self.edges.insert((tail, head));
    }

    pub(super) fn contains_vertex(&self, vertex: &NodeIndex) -> bool {
        self.vertices.contains(&(*vertex).into())
    }

    pub(super) fn contains_edge(&self, tail: NodeIndex, head: NodeIndex) -> bool {
        self.edges.contains(&(tail, head))
    }

    /// Returns true if exactly one vertex is a member of the tree.
    pub(super) fn is_incident_edge(&self, tail: &NodeIndex, head: &NodeIndex) -> bool {
        self.contains_vertex(tail) ^ self.contains_vertex(head)
    }

    pub(super) fn is_leave(&self, vertex: &NodeIndex) -> bool {
        self.vertices.get(&(*vertex).into()).unwrap().is_leave()
    }

    pub(super) fn vertices(&self) -> Iter<'_, Vertex> {
        self.vertices.iter()
    }

    pub(super) fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub(super) fn vertice_count(&self) -> usize {
        self.vertices.len()
    }

    pub(super) fn neighbor_count(&self, vertex: NodeIndex) -> usize {
        self.vertices.get(&vertex.into()).unwrap().neighbor_count()
    }

    pub(super) fn leaves(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        self.vertices.iter().filter(|v| v.is_leave()).map(|v| v.id()) 
    }

    pub(super) fn incoming(&self, vertex: NodeIndex) -> Neighbors<'_> {
        match self.vertices.get(&vertex.into()) {
            Some(v) => Neighbors { items: Some(v.incoming.iter()) },
            None => Neighbors { items: None },
        }
    }

    pub(super) fn outgoing(&self, vertex: NodeIndex) -> Neighbors<'_> {
        match self.vertices.get(&vertex.into()) {
            Some(v) => Neighbors { items: Some(v.outgoing.iter()) },
            None => Neighbors { items: None },
        }
    }

    pub(super) fn connected_edges(&self, vertex: NodeIndex) -> ConnectedEdges<'_> {
        match self.vertices.get(&vertex.into()) {
            Some(v) => ConnectedEdges { vertex, incoming: Some(v.incoming.iter()), outgoing: Some(v.outgoing.iter()) },
            None => ConnectedEdges { vertex, incoming: None, outgoing: None },
        }
    }

    pub(crate) fn from_edges(edges: &[(usize, usize)]) -> Self {
        let mut tree = Self::new();
        for (tail, head) in edges {
            tree.add_vertex(NodeIndex::new(*tail));
            tree.add_vertex(NodeIndex::new(*head));
            tree.add_edge(NodeIndex::new(*tail), NodeIndex::new(*head));
        }
        tree
    }
}

pub(super) struct Neighbors<'tree> {
    items: Option<Iter<'tree, NodeIndex>>
}

impl<'tree> Iterator for Neighbors<'tree> {
    type Item = NodeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.items {
            Some(i) => i.next().copied(),
            None => None,
        }
    }
}

pub(super) struct ConnectedEdges<'tree> {
    vertex: NodeIndex,
    incoming: Option<Iter<'tree, NodeIndex>>,
    outgoing: Option<Iter<'tree, NodeIndex>>,
}

impl<'tree> Iterator for ConnectedEdges<'tree> {
    type Item = (NodeIndex, NodeIndex);

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.incoming {
            Some(iter) => iter.next()
                              .map(|inc| (*inc, self.vertex))
                              .or_else(|| self.outgoing.as_mut() // if incoming is some, outgoing will also be some
                                                       .unwrap()
                                                       .next()
                                                       .map(|out| (self.vertex, *out))
            ),
            None => None
        }
    }
}



#[derive(Eq, Debug)]
pub(crate) struct Vertex {
    id: NodeIndex,
    incoming: HashSet<NodeIndex>,
    outgoing: HashSet<NodeIndex>, 
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Vertex {
    fn new(id: NodeIndex) -> Self {
        Self {
            id,
            incoming: HashSet::new(),
            outgoing: HashSet::new(),
        }
    }

    pub(super) fn id(&self) -> NodeIndex {
        self.id
    }

    #[inline(always)]
    pub(super) fn neighbor_count(&self) -> usize {
        self.incoming.len() + self.outgoing.len()
    }

    fn is_leave(&self) -> bool {
        self.neighbor_count() < 2
    }
}

impl std::hash::Hash for Vertex {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl From<NodeIndex> for Vertex {
    fn from(value: NodeIndex) -> Self {
        Self::new(value)
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
    mod tree {
        use petgraph::adj::NodeIndex;

        use crate::graphs::p1_layering::tree::TreeSubgraph;

        #[test]
        fn test_leaves() {
            let tree = TreeSubgraph::from_edges(&[(0, 1), (0, 5), (5, 6), (4, 6), (1, 2), (2, 3), (3, 7)]);
            let leaves = tree.leaves().collect::<Vec<_>>();
            assert_eq!(leaves.len(), 2);
            assert!(leaves.contains(&NodeIndex::from(4)));
            assert!(leaves.contains(&NodeIndex::from(7)));
        }
    }
    mod vertex {
        use std::collections::HashSet;

        use petgraph::stable_graph::NodeIndex;

        use crate::graphs::p1_layering::tree::Vertex;

        #[test]
        fn test_vertex_id_equal_same_neighbors() {
            let v1 = Vertex::new(0.into());
            let v2 = Vertex::new(0.into());
            assert_eq!(v1, v2);
        }
        
        #[test]
        fn test_vertex_id_equal_not_same_neighbors() {
            let v1 = Vertex::new(0.into());
            let mut v2 = Vertex::new(0.into());
            v2.incoming.insert(1.into());
            assert_eq!(v1, v2);
        }

        #[test]
        fn test_vertex_store_hashset_with_neighbor() {
            let mut set = HashSet::new();
            let mut v = Vertex::new(0.into());
            v.outgoing.insert(1.into());
            set.insert(v);
            let v = set.take(&NodeIndex::new(0).into());
            assert!(v.is_some());
            assert!(v.unwrap().outgoing.contains(&1.into()));
        }

        #[test]
        fn test_vertex_store_multiple_times() {
            let mut set = HashSet::new();
            let v1 = Vertex::new(0.into());
            let v2= Vertex::new(0.into());

            assert!(set.insert(v1));
            assert!(!set.insert(v2));
        }
    }
}