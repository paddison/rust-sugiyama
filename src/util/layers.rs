use std::{ops::Index, collections::HashMap};

use petgraph::stable_graph::NodeIndex;

use crate::graphs::calculate_coordinates::{VDir, HDir};

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

    pub(super) fn traverse(dir: VDir, length: usize) -> impl Iterator<Item = usize> {
        let (mut start, step) = match dir {
            VDir::Down => (usize::MAX, 1), // up corresponds to left to right
            VDir::Up => (length, usize::MAX),
        };
        std::iter::repeat_with(move || {
                start = start.wrapping_add(step);
                start
            }).take(length)
    }
    
    pub(crate) fn traverse_vertically(&self, dir: VDir) -> impl Iterator<Item = usize> {
        Self::traverse(dir, self.height())
    }

    pub(crate) fn traverse_horizontally(&self, dir: HDir, layer: usize) -> impl Iterator<Item = usize> {
        Self::traverse(match dir {
            HDir::Left => VDir::Up,
            HDir::Right => VDir::Down,
        }, self._inner[layer].len())
    }
}

impl Index<usize> for Layers {
    type Output = [NodeIndex];

    fn index(&self, index: usize) -> &Self::Output {
        &self._inner[index]
    }
}

#[cfg(test)]
mod test {
    use crate::graphs::calculate_coordinates::{VDir, HDir};

    use super::Layers;

    fn create_test_layers() -> Layers {
        let layers = vec![
            vec![1.into()],
            vec![2.into(), 3.into()],
            vec![3.into(), 4.into(), 5.into()],
        ];
        Layers::new_from_layers(layers)
    }

    #[test]
    fn test_traverse_down() {
        let mut l = Layers::traverse(VDir::Down, 4);

        assert_eq!(l.next(), Some(0));
        assert_eq!(l.next(), Some(1));
        assert_eq!(l.next(), Some(2));
        assert_eq!(l.next(), Some(3));
        assert_eq!(l.next(), None);
    }

    #[test]
    fn test_traverse_up() {
        let mut l = Layers::traverse(VDir::Up, 4);

        assert_eq!(l.next(), Some(3));
        assert_eq!(l.next(), Some(2));
        assert_eq!(l.next(), Some(1));
        assert_eq!(l.next(), Some(0));
        assert_eq!(l.next(), None);
    }

    #[test]
    fn test_traverse_vertically_down() {
        let layers = create_test_layers();
        let mut iter = layers.traverse_vertically(VDir::Down);

        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_traverse_vertically_up() {
        let layers = create_test_layers();
        let mut iter = layers.traverse_vertically(VDir::Up);

        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_traverse_horizontally_right() {
        let layers = create_test_layers();
        let mut iter = layers.traverse_horizontally(HDir::Right, 2);

        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_traverse_horizontally_left() {
        let layers = create_test_layers();
        let mut iter = layers.traverse_horizontally(HDir::Left, 1);

        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), None);
    }
}