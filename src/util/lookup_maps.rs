use std::{collections::HashMap, ops::{Index, IndexMut}, marker::PhantomData};

use petgraph::graph::{NodeIndex};
use petgraph::stable_graph::NodeIndices;

/// Wrapper around a hashmap that provides unsafe access to nodes
/// Also supports access via index
#[derive(Debug)]
pub(crate) struct NodeLookupMap<V> {
    _inner: HashMap<NodeIndex, V>,
}

impl<V: Copy> NodeLookupMap<V> {
    pub(crate) fn new_with_value(nodes: &[NodeIndex], value: V) -> Self {
        let mut _inner = HashMap::new();
        
        for n in nodes {
            _inner.insert(*n, value);
        }

        Self { _inner }
    }
}

impl NodeLookupMap<NodeIndex> {
    pub(crate) fn new_with_indices(nodes: &[NodeIndex]) -> Self {
        let mut _inner = HashMap::new();

        for n in nodes {
            _inner.insert(*n, *n);
        }

        Self { _inner }
    }
}

impl<V> Index<NodeIndex> for NodeLookupMap<V> {
    type Output = V;

    fn index(&self, index: NodeIndex) -> &Self::Output {
        self._inner.get(&index).unwrap()
    }
}

impl<V> IndexMut<NodeIndex> for NodeLookupMap<V> {
    fn index_mut(&mut self, index: NodeIndex) -> &mut Self::Output {
        self._inner.get_mut(&index).unwrap()
    }
}

impl<V> Index<&NodeIndex> for NodeLookupMap<V> {
    type Output = V;

    fn index(&self, index: &NodeIndex) -> &Self::Output {
        self._inner.get(index).unwrap()
    }
}

impl<V> IndexMut<&NodeIndex> for NodeLookupMap<V> {
    fn index_mut(&mut self, index: &NodeIndex) -> &mut Self::Output {
        self._inner.get_mut(index).unwrap()
    }
}