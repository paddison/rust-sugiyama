use std::{
    ops::Index, 
    collections::{HashMap, HashSet}
};

use petgraph::stable_graph::{NodeIndex, StableDiGraph};

use crate::graphs::{p3_calculate_coordinates::{VDir, HDir}, p2_reduce_crossings::{Vertex, Edge}};

#[derive(Clone)]
/// Has to guarantee that each identifier in levels has an entry in position
pub(crate) struct Layers {
    _inner: Vec<Vec<NodeIndex>>,
    positions: HashMap<NodeIndex, (usize, usize)>, // level, position
    upper_neighbours: HashMap<NodeIndex, Vec<NodeIndex>>,
    lower_neighbours: HashMap<NodeIndex, Vec<NodeIndex>>,
}

impl Layers {
    #[allow(dead_code)] 
    pub(crate) fn new_empty() -> Self {
        Self {
            _inner: vec![],
            positions: HashMap::new(),
            upper_neighbours: HashMap::new(),
            lower_neighbours: HashMap::new(),
        }
    }
    #[allow(dead_code)]
    pub fn new<T>(inner: Vec<Vec<NodeIndex>>, g: &StableDiGraph<Option<T>, usize>) -> Self {

        let mut positions = HashMap::new();
        let mut upper_neighbours = HashMap::new();
        let mut lower_neighbours = HashMap::new();

        for (level_index, level) in inner.iter().enumerate() {
            for (pos, vertex) in level.iter().enumerate() {
                positions.insert(*vertex, (level_index, pos));
            }
        }

        for l in &inner {
            for v in l {
                let v_level = positions.get(v).unwrap().0;
                let v_upper_neighbors = g.neighbors_directed(*v, petgraph::Direction::Incoming).collect::<HashSet<_>>();
                let v_lower_neighbors = g.neighbors_directed(*v, petgraph::Direction::Outgoing).collect::<HashSet<_>>();
                let v_direct_upper_neighbors = Self::initialize_neighbors(&inner, v_level.wrapping_sub(1), v_upper_neighbors);
                let v_direct_lower_neighbors = Self::initialize_neighbors(&inner, v_level + 1, v_lower_neighbors);
                
                upper_neighbours.insert(*v, v_direct_upper_neighbors);
                lower_neighbours.insert(*v, v_direct_lower_neighbors);
            }
        }

        let layers = Self { _inner: inner, positions, upper_neighbours, lower_neighbours };
        assert!(layers.is_valid());
        layers
    }

    fn initialize_neighbors(layers_raw: &Vec<Vec<NodeIndex>>, level_index: usize, neighbors: HashSet<NodeIndex>) -> Vec<NodeIndex> {
        layers_raw.get(level_index).unwrap_or(&Vec::new()).iter().filter(|n| neighbors.contains(n)).cloned().collect()
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

    pub fn get_adjacent(&self, id: NodeIndex, dir: HDir) -> Option<NodeIndex> {
        match dir {
            HDir::Left => self.get_next(id),
            HDir::Right => self.get_previous(id),
        }
    }

    pub(crate) fn get_upper_neighbours(&self, dest: NodeIndex) -> &[NodeIndex]  {
        self.upper_neighbours.get(&dest).unwrap()
    }

    pub(crate) fn get_lower_neighbours(&self, source: NodeIndex) -> &[NodeIndex] {
        self.lower_neighbours.get(&source).unwrap()
    }

    fn get_previous(&self, id: NodeIndex) -> Option<NodeIndex> {
        let pos = self.get_position(id);
        match pos {
            0 => None,
            pos => self._inner[self.get_level(id)].get(pos - 1).cloned()
        }
    }

    fn get_next(&self, id: NodeIndex) -> Option<NodeIndex> {
        let pos = self.get_position(id);
        self._inner[self.get_level(id)].get(pos + 1).cloned()
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

    // todo: Refactor this into trait
    pub(crate) fn iterate(dir: IterDir, length: usize) -> impl Iterator<Item = usize> {
        let (mut start, step) = match dir {
            IterDir::Forward => (usize::MAX, 1), // up corresponds to left to right
            IterDir::Backward => (length, usize::MAX),
        };
        std::iter::repeat_with(move || {
                start = start.wrapping_add(step);
                start
            }).take(length)
    }
    
    pub(crate) fn iterate_vertically(&self, dir: VDir) -> impl Iterator<Item = usize> {
        Self::iterate(dir.into(), self.height())
    }

    pub(crate) fn iterate_horizontally(&self, dir: HDir, level: usize) -> impl Iterator<Item = usize> {
        Self::iterate(dir.into(), self._inner[level].len())
    }
}

impl Index<usize> for Layers {
    type Output = [NodeIndex];

    fn index(&self, index: usize) -> &Self::Output {
        &self._inner[index]
    }
}

#[derive(Clone, Copy)]
pub(crate) enum IterDir {
    Forward,
    Backward,
}

impl From<VDir> for IterDir {
    fn from(val: VDir) -> Self {
        match val {
            VDir::Up => Self::Backward,
            VDir::Down => Self::Forward,
        }
    }
}

impl From<HDir> for IterDir {
    fn from(val: HDir) -> Self {
        match val {
            HDir::Left => Self::Backward,
            HDir::Right => Self::Forward,
        }
    }
}

#[cfg(test)]
mod test {
    use petgraph::stable_graph::StableDiGraph;

    use crate::{graphs::p3_calculate_coordinates::{VDir, HDir}, util::layers::IterDir};

    use super::Layers;

    fn create_test_layers() -> Layers {
        let layers = vec![
            vec![1.into()],
            vec![2.into(), 3.into()],
            vec![3.into(), 4.into(), 5.into()],
        ];
        let g = StableDiGraph::new();
        Layers::new::<usize>(layers, &g)
    }

    #[test]
    fn test_traverse_forward() {
        let mut l = Layers::iterate(IterDir::Forward, 4);

        assert_eq!(l.next(), Some(0));
        assert_eq!(l.next(), Some(1));
        assert_eq!(l.next(), Some(2));
        assert_eq!(l.next(), Some(3));
        assert_eq!(l.next(), None);
    }

    #[test]
    fn test_traverse_backward() {
        let mut l = Layers::iterate(IterDir::Backward, 4);

        assert_eq!(l.next(), Some(3));
        assert_eq!(l.next(), Some(2));
        assert_eq!(l.next(), Some(1));
        assert_eq!(l.next(), Some(0));
        assert_eq!(l.next(), None);
    }

    #[test]
    fn test_traverse_vertically_down() {
        let layers = create_test_layers();
        let mut iter = layers.iterate_vertically(VDir::Down);

        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_traverse_vertically_up() {
        let layers = create_test_layers();
        let mut iter = layers.iterate_vertically(VDir::Up);

        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_traverse_horizontally_right() {
        let layers = create_test_layers();
        let mut iter = layers.iterate_horizontally(HDir::Right, 2);

        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_traverse_horizontally_left() {
        let layers = create_test_layers();
        let mut iter = layers.iterate_horizontally(HDir::Left, 1);

        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), None);
    }
}