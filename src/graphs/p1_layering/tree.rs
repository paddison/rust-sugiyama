use std::collections::{ HashSet, hash_set::Iter };

use petgraph::stable_graph::NodeIndex;

pub(super) struct Tree {
    vertices: HashSet<Vertex>,
    edges: HashSet<(NodeIndex, NodeIndex)>,
}

impl Tree {
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
        v_tail.neighbors.insert(head);
        self.vertices.insert(v_tail);

        // update head
        let mut v_head = self.vertices.take(&head.into()).unwrap();
        v_head.neighbors.insert(tail);
        self.vertices.insert(v_head);

        self.edges.insert((tail, head));
    }

    pub(super) fn contains_vertex(&self, node: &NodeIndex) -> bool {
        self.vertices.contains(&(*node).into())
    }

    pub(super) fn contains_edge(&self, tail: NodeIndex, head: NodeIndex) -> bool {
        self.edges.contains(&(tail, head))
    }

    /// Returns true if exactly one vertex is a member of the tree.
    pub(super) fn is_incident_edge(&self, tail: &NodeIndex, head: &NodeIndex) -> bool {
        self.contains_vertex(tail) ^ self.contains_vertex(head)
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

    pub(super) fn leaves(&self) -> impl Iterator<Item = NodeIndex> + '_ {
        self.vertices.iter().filter(|v| v.is_leave()).map(|v| v.id()) 
    }
}

#[derive(Eq, Debug)]
pub(crate) struct Vertex {
    id: NodeIndex,
    neighbors: HashSet<NodeIndex>, 
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
            neighbors: HashSet::new(),
        }
    }

    pub(super) fn id(&self) -> NodeIndex {
        self.id
    }

    fn is_leave(&self) -> bool {
        self.neighbors.len() < 2
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
            v2.neighbors.insert(1.into());
            assert_eq!(v1, v2);
        }

        #[test]
        fn test_vertex_store_hashset_with_neighbor() {
            let mut set = HashSet::new();
            let mut v = Vertex::new(0.into());
            v.neighbors.insert(1.into());
            set.insert(v);
            let v = set.take(&NodeIndex::new(0).into());
            assert!(v.is_some());
            assert!(v.unwrap().neighbors.contains(&1.into()));
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