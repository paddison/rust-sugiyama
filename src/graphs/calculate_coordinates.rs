use std::collections::{HashMap, HashSet};

use petgraph::stable_graph::{StableDiGraph, NodeIndex};

use crate::impl_layer_graph;
use crate::util::{
    layers::Layers,
    lookup_maps::NodeLookupMap,
    traits::LayerGraph
};

pub(crate) struct Blocks<T: Default> {
    layers: Layers,
    graph: StableDiGraph<Option<T>, usize>,
    root: NodeLookupMap<NodeIndex>,
    align: NodeLookupMap<NodeIndex>,
}

impl_layer_graph!(Blocks<T>);

impl<T: Default> Blocks<T> {
    pub(crate) fn do_horizontal_compaction(&self) -> HashMap<NodeIndex, isize> {
        let mut x_coordinates = HashMap::new();
        let indices = self.graph.node_indices().collect::<Vec<_>>();
        let mut sink = NodeLookupMap::new_with_indices(&indices);
        let mut shift = NodeLookupMap::new_with_value(&indices, isize::MAX);

        for id in self.graph.node_indices() {
            if self.root[id] == id {
                self.place_block(id, &mut x_coordinates, &mut sink, &mut shift);
            }
        }

        for i in 0..self.layers.height() {
            let mut v = self.layers[i][0];
            if sink[v] == v {
                if shift[sink[v]] == isize::MAX { shift[sink[v]] = 0 }
                let mut j = i;
                let mut k = 0;
                loop {
                    v = self.layers[j][k];
                    while self.align[v] != self.root[v] {
                        v = self.align[v];
                        j += 1;
                        if let Some(u) = self.layers.get_previous(v) {
                            shift[sink[u]] = std::cmp::min(
                                shift[sink[u]], 
                                shift[sink[v]] + *x_coordinates.get(&v).unwrap() - (*x_coordinates.get(&u).unwrap() + 20)
                            );
                        }
                    }
                    k = self.get_position(v) + 1;
                    if k >= self.layers[j].len() || sink[v] != sink[self.layers[j][k]] {
                        break;
                    }
                }
            }
        }

        for v in self.graph.node_indices() {
            x_coordinates.insert(v, *x_coordinates.get(&v).unwrap() + shift[sink[v]]);
        }

        x_coordinates
    }

    fn place_block(
        &self,
        id: NodeIndex, 
        x_coordinates: &mut HashMap<NodeIndex, isize>, 
        sink: &mut NodeLookupMap<NodeIndex>, 
        shift: &mut NodeLookupMap<isize>, 
    ) {
        if x_coordinates.get(&id).is_some() {
            return;
        }
        x_coordinates.insert(id, 0);
        let mut w = id;
        loop {
            if let Some(pred) = self.layers.get_previous(w) {
                let u = self.root[pred];
                self.place_block(u, x_coordinates, sink, shift);
                if sink[id] == id { 
                    sink[id] = sink[u]; 
                }
                if sink[id] == sink[u] {
                    x_coordinates.insert(id, std::cmp::max(*x_coordinates.get(&id).unwrap(), *x_coordinates.get(&u).unwrap() + 20));
                }
            }
            w = self.align[w];
            if w == id {
                break
            }
        }
        while self.align[w] != id {
            w = self.align[w];
            x_coordinates.insert(w, *x_coordinates.get(&id).unwrap());
            sink[w] = sink[id]
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum HorizontalDirection {
    Left,
    Right,
}

#[derive(Clone, Copy)]
pub(crate) enum VerticalDirection {
    Up,
    Down,
}

#[derive(Clone)]
pub(crate) struct MarkedInnerSegments<T: Default> {
    layers: Layers,
    graph: StableDiGraph<Option<T>, usize>,
    marked_segments: HashSet<(NodeIndex, NodeIndex)>,
}

impl_layer_graph!(MarkedInnerSegments<T>);

impl<T: Default> MarkedInnerSegments<T> {
    pub(crate) fn create_vertical_alignments(self, vertical_direction: VerticalDirection, horizontal_direction: HorizontalDirection) -> Blocks<T> {
        let indices = self.graph.node_indices().collect::<Vec<_>>();
        let mut root = NodeLookupMap::new_with_indices(&indices);
        let mut align = NodeLookupMap::new_with_indices(&indices);

        for level in self.layers.levels() {
            let mut r = None;
            for vertex in level {
                let neighbours = match vertical_direction {
                    VerticalDirection::Up => self.get_upper_neighbours(*vertex),
                    VerticalDirection::Down => self.get_lower_neighbours(*vertex),
                };
                if neighbours.len() == 0 {
                    continue;
                }
                let d = (neighbours.len() as f64 + 1.) / 2. - 1.; // need to subtract one because indices are zero based
                let iter = d.floor() as usize..(d.ceil() as usize) + 1; 
                let median_indices = match horizontal_direction {
                    HorizontalDirection::Left => iter.collect::<Vec<_>>(),
                    HorizontalDirection::Right => iter.rev().collect::<Vec<_>>(),
                };
                for m in median_indices {
                    if align[vertex] == *vertex {
                        let median_neighbour = neighbours[m];
                        if !self.is_marked(&(median_neighbour, *vertex)) && self.has_no_type_0_conflict(self.get_position(median_neighbour), r, horizontal_direction) {
                            match vertical_direction {
                                VerticalDirection::Up => {
                                    align[median_neighbour] = *vertex;
                                    root[vertex] = root[median_neighbour];
                                    align[vertex] = root[vertex];
                                },
                                VerticalDirection::Down => {
                                    align[*vertex] = median_neighbour;
                                    root[median_neighbour] = root[vertex];
                                    align[median_neighbour] = root[median_neighbour];
                                },
                            }
                            // align[median_neighbour] = *vertex;
                            // root[vertex] = root[median_neighbour];
                            // align[vertex] = root[vertex];
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
        self.marked_segments.contains(edge)
    }

    #[inline(always)]
    fn has_no_type_0_conflict(&self, neighbour_pos: usize, r: Option<usize>, direction: HorizontalDirection) -> bool {
        match r {
            None => true,
            Some(v) => {
                match direction {
                    HorizontalDirection::Left => v < neighbour_pos,
                    HorizontalDirection::Right => v > neighbour_pos,
                }
            }
        }
    }
}

pub(crate) struct MinimalCrossings<T: Default> {
    layers: Layers,
    graph: StableDiGraph<Option<T>, usize>
}

impl_layer_graph!(MinimalCrossings<T>);

impl<T: Default> MinimalCrossings<T> {
    pub(crate) fn new(layers: Layers, graph: StableDiGraph<Option<T>, usize>) -> Self {
        Self { layers, graph }
    }

    fn is_dummy(&self, id: NodeIndex) -> bool {
        self.graph.node_weight(id).unwrap().is_none()
    }
    
    fn is_incident_to_inner_segment(&self, id: NodeIndex) -> bool {
        self.is_dummy(id) &&
        self.get_upper_neighbours(id).into_iter().any(|n| self.is_dummy(n))
    }

    /// Assumes id is incident to inner segment 
    fn get_inner_segment_upper_neighbor(&self, id: NodeIndex) -> Option<NodeIndex> {
        if self.is_incident_to_inner_segment(id) {
            self.get_upper_neighbours(id).get(0).cloned()
        } else {
            None
        }
    }

    pub(crate) fn mark_type_1_conflicts(self) -> MarkedInnerSegments<T>{
        let mut marked_segments = HashSet::new();

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
                        let vertex_index = self.layers.get_position(upper_neighbor);
                        if vertex_index < left_dummy_index || vertex_index > right_dummy_index {
                            marked_segments.insert((upper_neighbor, vertex));
                        }
                    }
                    l = l + 1;
                }
                left_dummy_index = right_dummy_index;
            }
        }

        MarkedInnerSegments { layers: self.layers, graph: self.graph, marked_segments }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashSet;

    use petgraph::stable_graph::{NodeIndex, StableDiGraph};

    use crate::{util::{layers::Layers, traits::LayerGraph}, graphs::calculate_coordinates::{HorizontalDirection, VerticalDirection}};

    use super::MinimalCrossings;

    pub(crate) fn create_test_layout() -> MinimalCrossings<usize> {
        let edges: [(usize, usize); 29] = [(0, 2), (0, 6), (1, 16), (1, 17), 
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
        
        let layers = Layers::new_from_layers(layers_raw);

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
        println!("{:?}", marked_segments.marked_segments);
        assert_eq!(marked_segments.marked_segments.len(), 4);
    }

    #[test]
    fn test_create_vertical_alignments() {
        let g = create_test_layout();
        let marked_segments = g.mark_type_1_conflicts();
        let blocks = marked_segments.create_vertical_alignments(VerticalDirection::Up, HorizontalDirection::Left);
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
    fn test_alignment_directions() {
        let g = create_test_layout();
        let blocks = g.mark_type_1_conflicts().create_vertical_alignments(VerticalDirection::Up, HorizontalDirection::Right);
        dbg!(blocks.align);
        // println!("{:?}", blocks.align);
    }

    #[test]
    fn test_do_horizontal_compaction() {
        let g = create_test_layout();
        let coords = g.mark_type_1_conflicts()
                      .create_vertical_alignments(VerticalDirection::Down, HorizontalDirection::Right)
                      .do_horizontal_compaction();

        println!("{:?}", coords);
    }
}
