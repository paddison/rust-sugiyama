use std::{ops::Index, collections::HashMap};

use petgraph::stable_graph::NodeIndex;

#[derive(Clone)]
/// Has to guarantee that each identifier in levels has an entry in position
pub(crate) struct Layers {
    _inner: Vec<Vec<NodeIndex>>,
    positions: HashMap<NodeIndex, (usize, usize)>, // level, position
}

impl Layers {
    pub fn new_from_layers(layers_raw: Vec<Vec<NodeIndex>>) -> Self {
        let mut positions = HashMap::new();

        for (level_index, level) in layers_raw.iter().enumerate() {
            for (pos, vertex) in level.iter().enumerate() {
                positions.insert(*vertex, (level_index, pos));
            }
        }

        let _inner = Self { _inner: layers_raw, positions };
        assert!(_inner.is_valid());
        _inner
    }

    pub fn height(&self) -> usize {
        self._inner.len()
    }

    pub fn get_position(&self, id: NodeIndex) -> usize {
        self.positions.get(&id).unwrap().1
    }

    pub fn get_level(&self, id: NodeIndex) -> usize {
        self.positions.get(&id).unwrap().0
    }

    pub fn get_previous(&self, id: NodeIndex) -> Option<NodeIndex> {
        let pos = self.get_position(id);
        match pos {
            0 => None,
            pos => self._inner[self.get_level(id)].get(pos - 1).cloned()
        }
    }

    pub(crate) fn levels(&self) -> &[Vec<NodeIndex>] {
        &self._inner
    }

    fn is_valid(&self) -> bool {
        for level in &self._inner {
            for id in level {
                if self.positions.get(id).is_none() {
                    return false
                }
            }
        }

        return true
    }
}

impl Index<usize> for Layers {
    type Output = [NodeIndex];

    fn index(&self, index: usize) -> &Self::Output {
        &self._inner[index]
    }
}
