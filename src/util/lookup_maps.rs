use std::{
    collections::{HashMap, hash_map::Iter}, 
    ops::{Index, IndexMut}
};

use petgraph::graph::NodeIndex;


/// Wrapper around a hashmap that provides unsafe access to nodes
/// Also supports access via index
/// A NodeLookup has the invariant that it must at all times contain all the nodes of a graph
#[derive(Debug)]
pub(crate) struct NodeLookupMap<V> {
    _inner: HashMap<NodeIndex, V>,
}

impl<V: Copy> NodeLookupMap<V> {
    pub(crate) fn new_with_value(nodes: &[NodeIndex], value: V) -> Self {
        let mut _inner = HashMap::with_capacity(nodes.len());
        
        for n in nodes {
            _inner.insert(*n, value);
        }

        Self { _inner }
    }

}

impl NodeLookupMap<NodeIndex> {
    pub(crate) fn new_with_indices(nodes: &[NodeIndex]) -> Self {
        let mut _inner = HashMap::with_capacity(nodes.len());

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

impl<'a, V> IntoIterator for &'a NodeLookupMap<V> {
    type Item = (&'a NodeIndex, &'a V);

    type IntoIter = Iter<'a, NodeIndex, V>;

    fn into_iter(self) -> Self::IntoIter {
        self._inner.iter() 
    }
}
