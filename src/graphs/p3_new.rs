use std::collections::{HashMap, HashSet};
use std::ops::{Deref, DerefMut};

use petgraph::Direction::Incoming;
use petgraph::stable_graph::{StableDiGraph, NodeIndex, EdgeIndex};
use petgraph::visit::{EdgeRef, IntoEdgeReferences};

use crate::graphs::p3_calculate_coordinates::{ VDir, HDir };
use crate::util:: layers::Layers;

/// Reprents a Layered Graph, in which the number of crossings of edges between
/// Vertices has been minimized. This implies that the order of vertices will not change
/// in the following steps of the algorithm.
/// 
/// It's then used to mark all type 1 conflicts (a crossing between an inner segment and a non-inner segment)
#[derive(Clone, Copy)]
pub struct Vertex {
    rank: usize,
    pos: usize,
    is_dummy: bool,
    root: NodeIndex,
    align: NodeIndex,
    shift: isize,
    sink: NodeIndex,
}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            rank: usize::default(),
            pos: usize::default(),
            is_dummy: false,
            root: 0.into(),
            align: 0.into(),
            shift: isize::MAX,
            sink: 0.into(),
        }
    }
}

impl Vertex {
    pub fn new(id: NodeIndex, rank: usize, pos: usize, is_dummy: bool) -> Self {
        Self {
            rank,
            pos,
            is_dummy,      
            root: id,
            align: id,
            shift: isize::MAX,
            sink: id,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Edge {
    has_type_1_conflict: bool
}

impl Default for Edge {
    fn default() -> Self {
        Self {
            has_type_1_conflict: false,
        }
    }
}


pub struct MinimalCrossings {
    pub layers: Layers,
    graph: StableDiGraph<Vertex, Edge>
}

impl Deref for MinimalCrossings {
    type Target = StableDiGraph<Vertex, Edge>;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl DerefMut for MinimalCrossings {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}

impl MinimalCrossings {

    #[allow(dead_code)]
    pub(crate) fn new(layers: Layers, graph: StableDiGraph<Vertex, Edge>) -> Self {
        Self { layers, graph }
    }

    fn is_incident_to_inner_segment(&self, id: NodeIndex) -> bool {
        self[id].is_dummy &&
        self.graph.neighbors_directed(id, Incoming).into_iter().any(|n| self[n].is_dummy)
    }

    /// Assumes id is incident to inner segment 
    fn get_inner_segment_upper_neighbor(&self, id: NodeIndex) -> Option<NodeIndex> {
        if self.is_incident_to_inner_segment(id) {
            self.neighbors_directed(id, Incoming).next()
        } else {
            None
        }
    }

    pub(crate) fn mark_type_1_conflicts(mut self) -> MarkedTypeOneConflicts{
        for (level, next_level) in self.layers
                                       .levels()[..self.layers.height() - 1]
                                       .iter()
                                       .zip(self.layers.levels()[1..].iter()) 
        {
            let mut left_dummy_index = 0;
            let mut l = 0;
            for (l_1, dummy_candidate) in next_level.iter().enumerate() {
                let right_dummy_index = match self.get_inner_segment_upper_neighbor(*dummy_candidate) {
                    Some(id) => self[id].pos,
                    None => if l_1 == next_level.len()  - 1 { 
                        level.len() 
                    } else { 
                        continue;
                    }
                };
                while l < l_1 {
                    let vertex = next_level[l];
                    let mut upper_neighbors = self.neighbors_directed(vertex, Incoming).collect::<Vec<_>>();
                    upper_neighbors.sort_by(|a, b| self[*a].pos.cmp(&self[*b].pos));
                    for upper_neighbor in upper_neighbors {
                        let vertex_index = self[upper_neighbor].pos;
                        if vertex_index < left_dummy_index || vertex_index > right_dummy_index {
                            let edge = self.find_edge(upper_neighbor, vertex).unwrap();
                            self.graph[edge].has_type_1_conflict = true;
                        }
                    }
                    l = l + 1;
                }
                left_dummy_index = right_dummy_index;
            }
        }
        MarkedTypeOneConflicts { layers: self.layers, graph: self.graph }
    }
}



/// Represents a Layered Graph, in which all type-1 conflicts have been marked.
/// 
/// It is used to align the graph in so called blocks, which are used in the next step 
/// to determine the x-coordinate of a vertex.
#[derive(Clone)]
pub(crate) struct MarkedTypeOneConflicts {
    layers: Layers,
    graph: StableDiGraph<Vertex, Edge>,
}

impl Deref for MarkedTypeOneConflicts {
    type Target = StableDiGraph<Vertex, Edge>;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl DerefMut for MarkedTypeOneConflicts {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}


impl MarkedTypeOneConflicts {

    pub(crate) fn create_vertical_alignments(mut self, vertical_direction: VDir, horizontal_direction: HDir) -> Blocks {
        // rotate the graph
        println!("{:?}", self.edge_endpoints(10.into()).unwrap());
        if let VDir::Up = vertical_direction {
            self.reverse();
            self.layers._inner.reverse();
        }
        println!("{:?}", self.edge_endpoints(10.into()).unwrap());

        if let HDir::Left = horizontal_direction {
            for row in self.layers._inner.iter_mut() {
                row.reverse();
            }
        }

        for (rank, row) in self.layers._inner.iter().enumerate() {
            for (pos, v) in row.iter().enumerate() {
                let weight: &mut Vertex = &mut self.graph[*v]; 
                weight.rank = rank;
                weight.pos = pos;
            }
        }

        for i in 0..self.layers.height() {
            let mut r = -1;

            for k in 0..self.layers[i].len() {
                let v = self.layers[i][k];
                let mut edges = self.edges_directed(v, Incoming).map(|e| (e.id(), e.source())).collect::<Vec<_>>();
                if edges.len() == 0 {
                    continue;
                }
                edges.sort_by(|e1, e2| self[e1.1].pos.cmp(&self[e2.1].pos));

                let d = (edges.len() as f64 + 1.) / 2. - 1.; // need to subtract one because indices are zero based
                let lower_upper_median = [d.floor() as usize, d.ceil() as usize];

                for m in lower_upper_median  {
                    if self[v].align == v {
                        let edge_id = edges[m].0;
                        let median_neighbor = edges[m].1;
                        if !self[edge_id].has_type_1_conflict && r < self[median_neighbor].pos as isize {
                            self[median_neighbor].align = v;
                            self[v].root = self[median_neighbor].root;
                            self[v].align = self[v].root;
                            r = self[median_neighbor].pos as isize;
                        }
                    }
                }
            }
        }

        Blocks{ layers: self.layers, graph: self.graph }
    }
}

/// Represents a layered graph whose vertices have been aligned in blocks.
/// A root is the highest node in a block, depending on the direction.
/// 
/// It is used to determine classes of a block, calculate the x-coordinates of a block
/// in regard to its class and shift classes together as close as possible.
pub(crate) struct Blocks {
    layers: Layers,
    graph: StableDiGraph<Vertex, Edge>,
}

impl Deref for Blocks {
    type Target = StableDiGraph<Vertex, Edge>;

    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

impl DerefMut for Blocks {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.graph
    }
}

impl Blocks {
    pub(crate) fn do_horizontal_compaction(&mut self, vertex_spacing: usize, horizontal_direction: HDir) -> HashMap<NodeIndex, isize> {
        let mut x_coordinates = HashMap::new();
        // place blocks
        for id in self.graph.node_indices().collect::<Vec<_>>() {
            if self[id].root == id {
                self.place_block(id, &mut x_coordinates, vertex_spacing as isize);
            }
        }

        // calculate class shifts 
        for i in 0..self.layers.height() { 
            let mut v = self.layers[i][0];
            if self[v].sink == v {
                if self[self[v].sink].shift == isize::MAX {
                    let v_sink = self.graph[v].sink;
                    self.graph[v_sink].shift = 0;
                }
                let mut j = i; // level index
                let mut k = 0; // vertex in level index
                loop {
                    v = self.layers[j][k];

                    // traverse one block
                    while self[v].align != self[v].root {
                        v = self[v].align;
                        j += 1;

                        if self[v].pos > 1 {
                            let u = self.pred(v);
                            let distance_v_u = *x_coordinates.get(&v).unwrap() - (*x_coordinates.get(&u).unwrap() + vertex_spacing as isize);
                            let u_sink = self[u].sink;
                            self[u_sink].shift = self[self[u].sink].shift.min(self[self[v].sink].shift + distance_v_u);
                        }
                    }
                    k = self[v].pos + 1;

                    if k == self.layers[j].len() || self[v].sink != self[self.layers[j][k]].sink {
                        break;
                    }
                }
            }   
        }

        // calculate absolute x-coordinates
        for v in self.graph.node_indices() {
            x_coordinates.insert(v, *x_coordinates.get(&v).unwrap() + self[self[v].sink].shift);
        }
        // flip x_coordinates if we went from right to left
        if let HDir::Left = horizontal_direction {
            for v in self.node_indices() {
                x_coordinates.entry(v).and_modify(|x| *x = -*x);
            }
        }
        x_coordinates
    }

    fn place_block(
        &mut self,
        root: NodeIndex, 
        x_coordinates: &mut HashMap<NodeIndex, isize>, 
        vertex_spacing: isize
    ) {
        if x_coordinates.get(&root).is_some() {
            return;
        }
        x_coordinates.insert(root, 0);
        let mut w = root;
        loop {
            if self[w].pos > 1 {
                let u = self[self.pred(w)].root;
                self.place_block(u, x_coordinates, vertex_spacing);
                // initialize sink of current node to have the same sink as the root
                if self[root].sink == root { 
                    self[root].sink = self[u].sink; 
                }
                if self[root].sink == self[u].sink {
                    x_coordinates.insert(root, *x_coordinates.get(&root).unwrap().max(&(x_coordinates.get(&u).unwrap() + vertex_spacing)));

                }
            }
            w = self[w].align;
            if w == root {
                break
            }
        }
        // align all other vertices in this block to the x-coordinate of the root
        while self[w].align != root {
            w = self[w].align;
            x_coordinates.insert(w, *x_coordinates.get(&root).unwrap());
            self[w].sink = self[root].sink;
        }
    }

    fn pred(&self, vertex: NodeIndex) -> NodeIndex {
        self.layers._inner[self[vertex].rank][self[vertex].pos - 1]
    }
}
/*
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
*/

mod tests {
    use petgraph::{stable_graph::{StableDiGraph, NodeIndex}, visit::IntoEdgeReferences};

    use super::{Vertex, Edge, MinimalCrossings};
    use crate::util::layers::Layers;
    use petgraph::visit::EdgeRef;
    use crate::graphs::p3_calculate_coordinates::{HDir, VDir};

    pub(crate) fn create_test_layout() -> MinimalCrossings {

        let edges: [(u32, u32); 30] = [(0, 2), (0, 6), (0, 18), (1, 16), (1, 17), 
                     (3, 8), (16, 8), (4, 8), (17, 19), (18, 20), (5, 8), (5, 9), (6, 8), (6, 21),
                     (7, 10), (7, 11), (7, 12), (19, 23), (20, 24), (21, 12), (9, 22), (9, 25),
                     (10, 13), (10, 14), (11, 14), (22, 13), (23, 15), (24, 15), (12, 15), (25, 15)];

        let mut graph = StableDiGraph::<Vertex, Edge>::from_edges(&edges);
        let layers_raw: Vec<Vec<NodeIndex>> = [
            vec![0, 1],
            vec![2, 3, 16, 4, 17, 18, 5, 6],
            vec![7, 8, 19, 20, 21, 9],
            vec![10, 11, 22, 23, 24, 12, 25],
            vec![13, 14, 15],
        ].into_iter().map(|row| row.into_iter().map(|id| id.into()).collect())
        .collect();
        
        for (rank, row) in layers_raw.iter().enumerate() {
            for (pos, v) in row.iter().enumerate() {
                let weight = &mut graph[*v];
                if v.index() < 16 {
                    *weight = Vertex::new(*v, rank, pos, false);
                } else {
                    *weight = Vertex::new(*v, rank, pos, true);
                }
            }
        }


        let layers = Layers::new2(layers_raw, &graph);
        MinimalCrossings { layers, graph }
    }

    #[test]
    fn type_1() {
        let mc = create_test_layout();
        let c = mc.mark_type_1_conflicts();
        assert!(c.graph[c.find_edge(6.into(), 8.into()).unwrap()].has_type_1_conflict);
        assert!(c.graph[c.find_edge(7.into(), 12.into()).unwrap()].has_type_1_conflict);
        assert!(c.graph[c.find_edge(5.into(), 8.into()).unwrap()].has_type_1_conflict);
        assert!(c.graph[c.find_edge(9.into(), 22.into()).unwrap()].has_type_1_conflict);
    }

    #[test]
    fn alignment_down_right() {
        let mc = create_test_layout().mark_type_1_conflicts().create_vertical_alignments(VDir::Down, HDir::Right); 
        // verify roots
        assert_eq!(mc[NodeIndex::from(0)].root, 0.into());
        assert_eq!(mc[NodeIndex::from(1)].root, 1.into());
        assert_eq!(mc[NodeIndex::from(2)].root, 0.into());
        assert_eq!(mc[NodeIndex::from(3)].root, 3.into());
        assert_eq!(mc[NodeIndex::from(4)].root, 4.into());
        assert_eq!(mc[NodeIndex::from(5)].root, 5.into());
        assert_eq!(mc[NodeIndex::from(6)].root, 6.into());
        assert_eq!(mc[NodeIndex::from(7)].root, 7.into());
        assert_eq!(mc[NodeIndex::from(8)].root, 4.into());
        assert_eq!(mc[NodeIndex::from(9)].root, 9.into());
        assert_eq!(mc[NodeIndex::from(10)].root, 7.into());
        assert_eq!(mc[NodeIndex::from(11)].root, 11.into());
        assert_eq!(mc[NodeIndex::from(12)].root, 6.into());
        assert_eq!(mc[NodeIndex::from(13)].root, 7.into());
        assert_eq!(mc[NodeIndex::from(14)].root, 11.into());
        assert_eq!(mc[NodeIndex::from(15)].root, 18.into());
        assert_eq!(mc[NodeIndex::from(16)].root, 1.into());
        assert_eq!(mc[NodeIndex::from(17)].root, 17.into());
        assert_eq!(mc[NodeIndex::from(18)].root, 18.into());
        assert_eq!(mc[NodeIndex::from(19)].root, 17.into());
        assert_eq!(mc[NodeIndex::from(20)].root, 18.into());
        assert_eq!(mc[NodeIndex::from(21)].root, 6.into());
        assert_eq!(mc[NodeIndex::from(22)].root, 22.into());
        assert_eq!(mc[NodeIndex::from(23)].root, 17.into());
        assert_eq!(mc[NodeIndex::from(24)].root, 18.into());
        assert_eq!(mc[NodeIndex::from(25)].root, 9.into());
        
        // verify alignments
        assert_eq!(mc[NodeIndex::from(0)].align, 2.into());
        assert_eq!(mc[NodeIndex::from(1)].align, 16.into());
        assert_eq!(mc[NodeIndex::from(2)].align, 0.into());
        assert_eq!(mc[NodeIndex::from(3)].align, 3.into());
        assert_eq!(mc[NodeIndex::from(4)].align, 8.into());
        assert_eq!(mc[NodeIndex::from(5)].align, 5.into());
        assert_eq!(mc[NodeIndex::from(6)].align, 21.into());
        assert_eq!(mc[NodeIndex::from(7)].align, 10.into());
        assert_eq!(mc[NodeIndex::from(8)].align, 4.into());
        assert_eq!(mc[NodeIndex::from(9)].align, 25.into());
        assert_eq!(mc[NodeIndex::from(10)].align, 13.into());
        assert_eq!(mc[NodeIndex::from(11)].align, 14.into());
        assert_eq!(mc[NodeIndex::from(12)].align, 6.into());
        assert_eq!(mc[NodeIndex::from(13)].align, 7.into());
        assert_eq!(mc[NodeIndex::from(14)].align, 11.into());
        assert_eq!(mc[NodeIndex::from(15)].align, 18.into());
        assert_eq!(mc[NodeIndex::from(16)].align, 1.into());
        assert_eq!(mc[NodeIndex::from(17)].align, 19.into());
        assert_eq!(mc[NodeIndex::from(18)].align, 20.into());
        assert_eq!(mc[NodeIndex::from(19)].align, 23.into());
        assert_eq!(mc[NodeIndex::from(20)].align, 24.into());
        assert_eq!(mc[NodeIndex::from(21)].align, 12.into());
        assert_eq!(mc[NodeIndex::from(22)].align, 22.into());
        assert_eq!(mc[NodeIndex::from(23)].align, 17.into());
        assert_eq!(mc[NodeIndex::from(24)].align, 15.into());
        assert_eq!(mc[NodeIndex::from(25)].align, 9.into());
    }

    #[test]
    fn alignment_down_left() {
        let mc = create_test_layout().mark_type_1_conflicts().create_vertical_alignments(VDir::Down, HDir::Left); 

        // block root 0
        for n in [0, 6] { assert_eq!(mc[NodeIndex::from(n)].root, 0.into()); }
        // block root 1
        for n in [1] { assert_eq!(mc[NodeIndex::from(n)].root, 1.into()); }
        // block root 2
        for n in [2] { assert_eq!(mc[NodeIndex::from(n)].root, 2.into()); }
        // block root 3
        for n in [3] { assert_eq!(mc[NodeIndex::from(n)].root, 3.into()); }
        // block root 16
        for n in [16] { assert_eq!(mc[NodeIndex::from(n)].root, 16.into()); }
        // block root 4
        for n in [4, 8] { assert_eq!(mc[NodeIndex::from(n)].root, 4.into()); }
        // block root 17
        for n in [17, 19, 23] { assert_eq!(mc[NodeIndex::from(n)].root, 17.into()); }
        // block root 18
        for n in [18, 20, 24] { assert_eq!(mc[NodeIndex::from(n)].root, 18.into()); }
        // block root 5
        for n in [5, 9, 25] { assert_eq!(mc[NodeIndex::from(n)].root, 5.into()); }
        // block root 7
        for n in [7, 11, 14] { assert_eq!(mc[NodeIndex::from(n)].root, 7.into()); }
        // block root 21
        for n in [21, 12, 15] { assert_eq!(mc[NodeIndex::from(n)].root, 21.into()); }
        // block root 10
        for n in [10, 13] { assert_eq!(mc[NodeIndex::from(n)].root, 10.into()); }
        // block root 22
        for n in [22] { assert_eq!(mc[NodeIndex::from(n)].root, 22.into()); }
    }

    #[test]
    fn alignment_up_right() {
        let mc = create_test_layout().mark_type_1_conflicts().create_vertical_alignments(VDir::Up, HDir::Right); 

        for n in [13, 10] { assert_eq!(mc[NodeIndex::from(n)].root, 13.into()) }
        for n in [14, 11, 7] { assert_eq!(mc[NodeIndex::from(n)].root, 14.into()) }
        for n in [15, 23, 19, 17] { assert_eq!(mc[NodeIndex::from(n)].root, 15.into()) }
        for n in [22] { assert_eq!(mc[NodeIndex::from(n)].root, 22.into()) }
        for n in [24, 20, 18, 0] { assert_eq!(mc[NodeIndex::from(n)].root, 24.into()) }
        for n in [12, 21] { assert_eq!(mc[NodeIndex::from(n)].root, 12.into()) }
        for n in [25, 9, 5] { assert_eq!(mc[NodeIndex::from(n)].root, 25.into()) }
        for n in [8, 3] { assert_eq!(mc[NodeIndex::from(n)].root, 8.into()) }
        for n in [2] { assert_eq!(mc[NodeIndex::from(n)].root, 2.into()) }
        for n in [16] { assert_eq!(mc[NodeIndex::from(n)].root, 16.into()) }
        for n in [4] { assert_eq!(mc[NodeIndex::from(n)].root, 4.into()) }
        for n in [6] { assert_eq!(mc[NodeIndex::from(n)].root, 6.into()) }
        for n in [1] { assert_eq!(mc[NodeIndex::from(n)].root, 1.into()) }
    }

    #[test]
    fn alignment_up_left() {
        let mc = create_test_layout().mark_type_1_conflicts().create_vertical_alignments(VDir::Up, HDir::Left); 

        for n in [15, 25, 9] { assert_eq!(mc[NodeIndex::from(n)].root, 15.into()) }
        for n in [14] { assert_eq!(mc[NodeIndex::from(n)].root, 14.into()) }
        for n in [13, 22] { assert_eq!(mc[NodeIndex::from(n)].root, 13.into()) }
        for n in [12, 21, 6] { assert_eq!(mc[NodeIndex::from(n)].root, 12.into()) }
        for n in [24, 20, 18] { assert_eq!(mc[NodeIndex::from(n)].root, 24.into()) }
        for n in [23, 19, 17, 1] { assert_eq!(mc[NodeIndex::from(n)].root, 23.into()) }
        for n in [11, 7] { assert_eq!(mc[NodeIndex::from(n)].root, 11.into()) }
        for n in [10] { assert_eq!(mc[NodeIndex::from(n)].root, 10.into()) }
        for n in [4, 8] { assert_eq!(mc[NodeIndex::from(n)].root, 8.into()) }
        for n in [0] { assert_eq!(mc[NodeIndex::from(n)].root, 0.into()) }
        for n in [2] { assert_eq!(mc[NodeIndex::from(n)].root, 2.into()) }
        for n in [3] { assert_eq!(mc[NodeIndex::from(n)].root, 3.into()) }
        for n in [16] { assert_eq!(mc[NodeIndex::from(n)].root, 16.into()) }
    }
}
