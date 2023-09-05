// TODO: maybe just rotate the whole graph?
use std::collections::{HashMap, HashSet};

use petgraph::stable_graph::{StableDiGraph, NodeIndex};

use crate::impl_layer_graph;
use crate::util::{
    layers::Layers,
    lookup_maps::NodeLookupMap,
    traits::LayerGraph
};

/// Reprents a Layered Graph, in which the number of crossings of edges between
/// Vertices has been minimized. This implies that the order of vertices will not change
/// in the following steps of the algorithm.
/// 
/// It's then used to mark all type 1 conflicts (a crossing between an inner segment and a non-inner segment)
pub struct MinimalCrossings<T: Default> {
    pub layers: Layers,
    graph: StableDiGraph<Option<T>, usize>
}

impl_layer_graph!(MinimalCrossings<T>);

impl<T: Default> MinimalCrossings<T> {

    #[allow(dead_code)]
    pub(crate) fn new(layers: Layers, graph: StableDiGraph<Option<T>, usize>) -> Self {
        Self { layers, graph }
    }

    fn is_dummy(&self, id: NodeIndex) -> bool {
        self.graph.node_weight(id).unwrap().is_none()
    }
    
    fn is_incident_to_inner_segment(&self, id: NodeIndex) -> bool {
        self.is_dummy(id) &&
        self.get_upper_neighbours(id).into_iter().any(|n| self.is_dummy(*n))
    }

    /// Assumes id is incident to inner segment 
    fn get_inner_segment_upper_neighbor(&self, id: NodeIndex) -> Option<NodeIndex> {
        if self.is_incident_to_inner_segment(id) {
            self.get_upper_neighbours(id).get(0).cloned()
        } else {
            None
        }
    }

    pub(crate) fn mark_type_1_conflicts(self) -> MarkedTypeOneConflicts<T>{
        let mut type_1_conflicts = HashSet::new();

        for (level, next_level) in self.layers
                                       .levels()[..self.layers.height() - 1]
                                       .iter()
                                       .zip(self.layers.levels()[1..].iter()) 
        {
            let mut left_dummy_index = 0;
            let mut l = 0;
            for (l_1, dummy_candidate) in next_level.iter().enumerate() {
                let right_dummy_index = match self.get_inner_segment_upper_neighbor(*dummy_candidate) {
                    Some(id) => self.layers.get_position(id),
                    None => if l_1 == next_level.len()  - 1 { level.len() } else { continue }
                };
                while l < l_1 {
                    let vertex = next_level[l];
                    for upper_neighbor in self.get_upper_neighbours(vertex) {
                        let vertex_index = self.layers.get_position(*upper_neighbor);
                        if vertex_index < left_dummy_index || vertex_index > right_dummy_index {
                            type_1_conflicts.insert((*upper_neighbor, vertex));
                        }
                    }
                    l = l + 1;
                }
                left_dummy_index = right_dummy_index;
            }
        }

        MarkedTypeOneConflicts { layers: self.layers, graph: self.graph, type_1_conflicts }
    }
}

/// Represents a Layered Graph, in which all type-1 conflicts have been marked.
/// 
/// It is used to align the graph in so called blocks, which are used in the next step 
/// to determine the x-coordinate of a vertex.
#[derive(Clone)]
pub(crate) struct MarkedTypeOneConflicts<T: Default> {
    layers: Layers,
    graph: StableDiGraph<Option<T>, usize>,
    type_1_conflicts: HashSet<(NodeIndex, NodeIndex)>,
}

impl_layer_graph!(MarkedTypeOneConflicts<T>);

impl<T: Default> MarkedTypeOneConflicts<T> {
    pub(crate) fn create_vertical_alignments(self, vertical_direction: VDir, horizontal_direction: HDir) -> Blocks<T> {
        let indices = self.graph.node_indices().collect::<Vec<_>>();
        let mut root = NodeLookupMap::new_with_indices(&indices);
        let mut align = NodeLookupMap::new_with_indices(&indices);

        for i in self.layers.iterate_vertically(vertical_direction) {
            let mut r = None;

            for k in self.layers.iterate_horizontally(horizontal_direction, i) {
                let v = self.layers[i][k];
                let neighbours = match vertical_direction {
                    VDir::Down => self.get_upper_neighbours(v),
                    VDir::Up => self.get_lower_neighbours(v),
                };
                if neighbours.len() == 0 {
                    continue;
                }

                let d = (neighbours.len() as f64 + 1.) / 2. - 1.; // need to subtract one because indices are zero based
                let lower_upper_median = match horizontal_direction {
                    HDir::Left => [d.ceil() as usize, d.floor() as usize],
                    HDir::Right => [d.floor() as usize, d.ceil() as usize],
                };

                for m in lower_upper_median  {
                    if align[v] == v {
                        let median_neighbour = neighbours[m];
                        if !self.is_marked(&(median_neighbour, v)) && self.has_no_type_0_conflict(self.get_position(median_neighbour), r, horizontal_direction) {
                            align[median_neighbour] = v;
                            root[v] = root[median_neighbour];
                            align[v] = root[v];
                            r = Some(self.get_position(median_neighbour));
                        }
                    }
                }
            }
        }

        Blocks{ layers: self.layers, graph: self.graph, root, align }
    }

    #[inline(always)]
    fn is_marked(&self, edge: &(NodeIndex, NodeIndex)) -> bool {
        println!("{:?}, {:?}", edge, self.type_1_conflicts.contains(edge));
        // TODO: This is a bug, edge needs to be reversed depending on direction
        self.type_1_conflicts.contains(edge)
    }

    #[inline(always)]
    fn has_no_type_0_conflict(&self, neighbour_pos: usize, r: Option<usize>, direction: HDir) -> bool {
        match r {
            None => true,
            Some(r) => {
                match direction {
                    HDir::Left => r > neighbour_pos,
                    HDir::Right => r < neighbour_pos,
                }
            }
        }
    }
}

/// Represents a layered graph whose vertices have been aligned in blocks.
/// A root is the highest node in a block, depending on the direction.
/// 
/// It is used to determine classes of a block, calculate the x-coordinates of a block
/// in regard to its class and shift classes together as close as possible.
pub(crate) struct Blocks<T: Default> {
    layers: Layers,
    graph: StableDiGraph<Option<T>, usize>,
    root: NodeLookupMap<NodeIndex>,
    align: NodeLookupMap<NodeIndex>,
}

impl_layer_graph!(Blocks<T>);

impl<T: Default> Blocks<T> {
    pub(crate) fn do_horizontal_compaction(&self, v_dir: VDir, h_dir: HDir, vertex_spacing: usize) -> HashMap<NodeIndex, isize> {
        let mut x_coordinates = HashMap::new();
        let indices = self.graph.node_indices().collect::<Vec<_>>();
        let mut sink = NodeLookupMap::new_with_indices(&indices);
        let initial_shift = match h_dir {
            HDir::Left => isize::MIN,
            HDir::Right => isize::MAX,
        };
        let mut shift = NodeLookupMap::new_with_value(&indices, initial_shift);

        // place blocks
        for id in self.graph.node_indices() {
            if self.root[id] == id {
                self.place_block(id, &mut x_coordinates, &mut sink, &mut shift, h_dir, vertex_spacing as isize);
            }
        }

        // calculate class shifts 
        for i in self.layers.iterate_vertically(v_dir) {
            let v_idx = match h_dir {
                HDir::Left => self.layers[i].len() - 1,
                HDir::Right => 0,
            };
            let mut v = self.layers[i][v_idx];
            if sink[v] == v {
                if shift[sink[v]] == initial_shift { 
                    shift[sink[v]] = 0 
                }
                let mut j = i; // level index
                let mut k = v_idx; // vertex in level index
                loop {
                    v = self.layers[j][k];

                    // traverse one block
                    while self.align[v] != self.root[v] {
                        v = self.align[v];
                        j = if v_dir == VDir::Up { j - 1 } else { j + 1 };

                        if let Some(u) = self.layers.get_adjacent(v, h_dir) {
                            let distance_v_u = *x_coordinates.get(&v).unwrap() - (*x_coordinates.get(&u).unwrap() + vertex_spacing as isize);
                            shift[sink[u]] = match h_dir {
                                HDir::Left => shift[sink[u]].max(shift[sink[v]] - distance_v_u),
                                HDir::Right => shift[sink[u]].min(shift[sink[v]] + distance_v_u),
                            };
                        }
                    }
                    let limit;
                    (k, limit) = match h_dir {
                        HDir::Left => (self.get_position(v).wrapping_sub(1), usize::MAX) ,
                        HDir::Right => (self.get_position(v) + 1, self.layers[j].len()) ,
                    };

                    if k == limit || sink[v] != sink[self.layers[j][k]] {
                        break;
                    }
                }
            }   
        }

        // calculate absolute x-coordinates
        for v in self.graph.node_indices() {
            match h_dir {
                HDir::Left => x_coordinates.insert(v, *x_coordinates.get(&v).unwrap() - shift[sink[v]]),
                HDir::Right => x_coordinates.insert(v, *x_coordinates.get(&v).unwrap() + shift[sink[v]]),
            };
        }

        x_coordinates
    }

    fn place_block(
        &self,
        root: NodeIndex, 
        x_coordinates: &mut HashMap<NodeIndex, isize>, 
        sink: &mut NodeLookupMap<NodeIndex>, 
        shift: &mut NodeLookupMap<isize>, 
        h_dir: HDir,
        vertex_spacing: isize
    ) {
        if x_coordinates.get(&root).is_some() {
            return;
        }
        x_coordinates.insert(root, 0);
        let mut w = root;
        loop {
            if let Some(neighbour) = self.layers.get_adjacent(w, h_dir) {
                let u = self.root[neighbour];
                self.place_block(u, x_coordinates, sink, shift, h_dir, vertex_spacing);
                // initialize sink of current node to have the same sink as the root
                if sink[root] == root { 
                    sink[root] = sink[u]; 
                }
                if sink[root] == sink[u] {
                    match h_dir {
                        HDir::Left => x_coordinates.insert(root, *x_coordinates.get(&root).unwrap().min(&(*x_coordinates.get(&u).unwrap() - vertex_spacing))),
                        HDir::Right => x_coordinates.insert(root, *x_coordinates.get(&root).unwrap().max(&(x_coordinates.get(&u).unwrap() + vertex_spacing))),
                    };
                }
            }
            w = self.align[w];
            if w == root {
                break
            }
        }
        // align all other vertices in this block to the x-coordinate of the root
        while self.align[w] != root {
            w = self.align[w];
            x_coordinates.insert(w, *x_coordinates.get(&root).unwrap());
            sink[w] = sink[root]
        }
    }
}

/// Represents the horizontal direction in which the algorithm is run
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum HDir {
    Left,
    Right,
}

/// Represents the vertical direction in which the algorithm is run
#[derive(Clone, Copy, PartialEq)]
pub(crate) enum VDir {
    Up,
    Down,
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashSet;

    use petgraph::stable_graph::{NodeIndex, StableDiGraph};

    use crate::{util::{layers::Layers, traits::LayerGraph}, graphs::p3_calculate_coordinates::{HDir, VDir}};

    use super::MinimalCrossings;

    pub(crate) fn create_test_layout() -> MinimalCrossings<usize> {
        let edges: [(usize, usize); 30] = [(0, 2), (0, 6), (0, 18), (1, 16), (1, 17), 
                     (3, 8), (16, 8), (4, 8), (17, 19), (18, 20), (5, 8), (5, 9), (6, 8), (6, 21),
                     (7, 10), (7, 11), (7, 12), (19, 23), (20, 24), (21, 12), (9, 22), (9, 25),
                     (10, 13), (10, 14), (11, 14), (22, 13), (23, 15), (24, 15), (12, 15), (25, 15)];

        let layers_raw: Vec<Vec<NodeIndex>> = [
            vec![0, 1],
            vec![2, 3, 16, 4, 17, 18, 5, 6],
            vec![7, 8, 19, 20, 21, 9],
            vec![10, 11, 22, 23, 24, 12, 25],
            vec![13, 14, 15],
        ].into_iter().map(|row| row.into_iter().map(|id| id.into()).collect())
        .collect();
        

        let mut graph = StableDiGraph::new();

        for _ in 0..16 {
            graph.add_node(Some(usize::default()));
        }
        for _ in 0..10 {
            graph.add_node(None);
        }

        for (a, b) in edges {
            graph.add_edge(NodeIndex::new(a), NodeIndex::new(b), usize::default());
        }
        let layers = Layers::new(layers_raw, &graph);
        MinimalCrossings { layers, graph }
    }

    #[test]
    fn test_is_dummy() {
        let g = create_test_layout();
        assert!(g.is_dummy(23.into()));
    }

    #[test]
    fn test_is_incident_to_inner_segment_true() {
        let g = create_test_layout();
        assert!(g.is_incident_to_inner_segment(19.into()));
    }

    #[test]
    fn test_is_not_incident_to_inner_segment_but_is_dummy() {
        let g = create_test_layout();
        assert!(!g.is_incident_to_inner_segment(22.into()));
    }

    #[test]
    fn test_get_upper_neighbours() {
        let g = create_test_layout();
        let actual = g.get_upper_neighbours(8.into()).into_iter().map(|n| n.index()).collect::<HashSet<usize>>();
        let expected = HashSet::from([3, 16, 4, 5, 6]);

        assert_eq!(actual.len(), 5);
        for n in expected {
            assert!(actual.contains(&n));
        }
    }

    #[test]
    fn test_get_inner_segment_upper_neighbour() {
        let g = create_test_layout();
        assert!(g.is_incident_to_inner_segment(23.into()));
    }

    #[test]
    fn test_mark_type_1_conflicts() {
        let g = create_test_layout();
        let marked_segments = g.mark_type_1_conflicts();
        println!("{:?}", marked_segments.type_1_conflicts);
        assert_eq!(marked_segments.type_1_conflicts.len(), 4);
    }

    #[test]
    fn test_create_vertical_alignments_down_right() {
        let g = create_test_layout();
        let marked_segments = g.mark_type_1_conflicts();
        let blocks = marked_segments.create_vertical_alignments(VDir::Down, HDir::Right);
        let root = &blocks.root;
        let align = &blocks.align;
        // verify roots
        assert_eq!(root[&0.into()], 0.into());
        assert_eq!(root[&1.into()], 1.into());
        assert_eq!(root[&2.into()], 0.into());
        assert_eq!(root[&3.into()], 3.into());
        assert_eq!(root[&4.into()], 4.into());
        assert_eq!(root[&5.into()], 5.into());
        assert_eq!(root[&6.into()], 6.into());
        assert_eq!(root[&7.into()], 7.into());
        assert_eq!(root[&8.into()], 4.into());
        assert_eq!(root[&9.into()], 9.into());
        assert_eq!(root[&10.into()], 7.into());
        assert_eq!(root[&11.into()], 11.into());
        assert_eq!(root[&12.into()], 6.into());
        assert_eq!(root[&13.into()], 7.into());
        assert_eq!(root[&14.into()], 11.into());
        assert_eq!(root[&15.into()], 18.into());
        assert_eq!(root[&16.into()], 1.into());
        assert_eq!(root[&17.into()], 17.into());
        assert_eq!(root[&18.into()], 18.into());
        assert_eq!(root[&19.into()], 17.into());
        assert_eq!(root[&20.into()], 18.into());
        assert_eq!(root[&21.into()], 6.into());
        assert_eq!(root[&22.into()], 22.into());
        assert_eq!(root[&23.into()], 17.into());
        assert_eq!(root[&24.into()], 18.into());
        assert_eq!(root[&25.into()], 9.into());
        
        // verify alignments
        assert_eq!(align[&0.into()], 2.into());
        assert_eq!(align[&1.into()], 16.into());
        assert_eq!(align[&2.into()], 0.into());
        assert_eq!(align[&3.into()], 3.into());
        assert_eq!(align[&4.into()], 8.into());
        assert_eq!(align[&5.into()], 5.into());
        assert_eq!(align[&6.into()], 21.into());
        assert_eq!(align[&7.into()], 10.into());
        assert_eq!(align[&8.into()], 4.into());
        assert_eq!(align[&9.into()], 25.into());
        assert_eq!(align[&10.into()], 13.into());
        assert_eq!(align[&11.into()], 14.into());
        assert_eq!(align[&12.into()], 6.into());
        assert_eq!(align[&13.into()], 7.into());
        assert_eq!(align[&14.into()], 11.into());
        assert_eq!(align[&15.into()], 18.into());
        assert_eq!(align[&16.into()], 1.into());
        assert_eq!(align[&17.into()], 19.into());
        assert_eq!(align[&18.into()], 20.into());
        assert_eq!(align[&19.into()], 23.into());
        assert_eq!(align[&20.into()], 24.into());
        assert_eq!(align[&21.into()], 12.into());
        assert_eq!(align[&22.into()], 22.into());
        assert_eq!(align[&23.into()], 17.into());
        assert_eq!(align[&24.into()], 15.into());
        assert_eq!(align[&25.into()], 9.into());
    }

    #[test]
    fn test_create_vertical_alignments_down_left() {
        let g = create_test_layout().mark_type_1_conflicts().create_vertical_alignments(VDir::Down, HDir::Left);
        
        // block root 0
        for n in [0, 6] { assert_eq!(g.root[NodeIndex::from(n)], 0.into()); }
        // block root 1
        for n in [1] { assert_eq!(g.root[NodeIndex::from(n)], 1.into()); }
        // block root 2
        for n in [2] { assert_eq!(g.root[NodeIndex::from(n)], 2.into()); }
        // block root 3
        for n in [3] { assert_eq!(g.root[NodeIndex::from(n)], 3.into()); }
        // block root 16
        for n in [16] { assert_eq!(g.root[NodeIndex::from(n)], 16.into()); }
        // block root 4
        for n in [4, 8] { assert_eq!(g.root[NodeIndex::from(n)], 4.into()); }
        // block root 17
        for n in [17, 19, 23] { assert_eq!(g.root[NodeIndex::from(n)], 17.into()); }
        // block root 18
        for n in [18, 20, 24] { assert_eq!(g.root[NodeIndex::from(n)], 18.into()); }
        // block root 5
        for n in [5, 9, 25] { assert_eq!(g.root[NodeIndex::from(n)], 5.into()); }
        // block root 7
        for n in [7, 11, 14] { assert_eq!(g.root[NodeIndex::from(n)], 7.into()); }
        // block root 21
        for n in [21, 12, 15] { assert_eq!(g.root[NodeIndex::from(n)], 21.into()); }
        // block root 10
        for n in [10, 13] { assert_eq!(g.root[NodeIndex::from(n)], 10.into()); }
        // block root 22
        for n in [22] { assert_eq!(g.root[NodeIndex::from(n)], 22.into()); }
    }

    #[test]
    fn test_create_vertical_alignments_up_right() {
        let g = create_test_layout().mark_type_1_conflicts().create_vertical_alignments(VDir::Up, HDir::Right);

        for n in [13, 10] { assert_eq!(g.root[NodeIndex::from(n)], 13.into()) }
        for n in [14, 11, 7] { assert_eq!(g.root[NodeIndex::from(n)], 14.into()) }
        for n in [15, 23, 19, 17] { assert_eq!(g.root[NodeIndex::from(n)], 15.into()) }
        for n in [22] { assert_eq!(g.root[NodeIndex::from(n)], 22.into()) }
        for n in [24, 20, 18, 0] { assert_eq!(g.root[NodeIndex::from(n)], 24.into()) }
        for n in [12, 21] { assert_eq!(g.root[NodeIndex::from(n)], 12.into()) }
        for n in [25, 9, 5] { assert_eq!(g.root[NodeIndex::from(n)], 25.into()) }
        for n in [8, 3] { assert_eq!(g.root[NodeIndex::from(n)], 8.into()) }
        for n in [2] { assert_eq!(g.root[NodeIndex::from(n)], 2.into()) }
        for n in [16] { assert_eq!(g.root[NodeIndex::from(n)], 16.into()) }
        for n in [4] { assert_eq!(g.root[NodeIndex::from(n)], 4.into()) }
        for n in [6] { assert_eq!(g.root[NodeIndex::from(n)], 6.into()) }
        for n in [1] { assert_eq!(g.root[NodeIndex::from(n)], 1.into()) }
    }

    #[test]
    fn test_create_vertical_alignments_up_left() {
        let g = create_test_layout().mark_type_1_conflicts().create_vertical_alignments(VDir::Up, HDir::Left);

        for n in [0] { assert_eq!(g.root[NodeIndex::from(n)], 0.into()) }
        for n in [2] { assert_eq!(g.root[NodeIndex::from(n)], 2.into()) }
        for n in [3] { assert_eq!(g.root[NodeIndex::from(n)], 3.into()) }
        for n in [16] { assert_eq!(g.root[NodeIndex::from(n)], 16.into()) }
        for n in [4] { assert_eq!(g.root[NodeIndex::from(n)], 4.into()) }
        for n in [17, 1] { assert_eq!(g.root[NodeIndex::from(n)], 17.into()) }
        for n in [18] { assert_eq!(g.root[NodeIndex::from(n)], 18.into()) }
        for n in [8, 5] { assert_eq!(g.root[NodeIndex::from(n)], 8.into()) }
        for n in [10] { assert_eq!(g.root[NodeIndex::from(n)], 10.into()) }
        for n in [11, 7] { assert_eq!(g.root[NodeIndex::from(n)], 11.into()) }
        for n in [23, 19] { assert_eq!(g.root[NodeIndex::from(n)], 23.into()) }
        for n in [24, 20] { assert_eq!(g.root[NodeIndex::from(n)], 24.into()) }
        for n in [12, 21, 6] { assert_eq!(g.root[NodeIndex::from(n)], 12.into()) }
        for n in [13, 22] { assert_eq!(g.root[NodeIndex::from(n)], 13.into()) }
        for n in [14] { assert_eq!(g.root[NodeIndex::from(n)], 14.into()) }
        for n in [15, 25, 9] { assert_eq!(g.root[NodeIndex::from(n)], 15.into()) }
    }

    #[test]
    fn test_alignment_directions() {
        let g = create_test_layout();
        let blocks = g.mark_type_1_conflicts().create_vertical_alignments(VDir::Down, HDir::Left);
        dbg!(blocks.align);
        let mut v = blocks.root.into_iter().collect::<Vec<_>>();
        v.sort_by(|r, l| r.0.cmp(l.0)); 
        println!("{:?}", v);
        // println!("{:?}", blocks.align);
    }

    #[test]
    fn test_do_horizontal_compaction() {
        let g = create_test_layout();
        let coords = g.mark_type_1_conflicts()
                      .create_vertical_alignments(VDir::Up, HDir::Left)
                      .do_horizontal_compaction(VDir::Up, HDir::Left, 20);

        println!("{:?}", coords);
    }
}
