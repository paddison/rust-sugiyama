use petgraph::stable_graph::{StableDiGraph, NodeIndex};

use crate::{Layouts, algorithm, Config};

pub trait IntoCoordinates {}

impl<V, E> IntoCoordinates for &StableDiGraph<V, E> {}
impl IntoCoordinates for &[(u32, u32)] {}

pub struct CoordinatesBuilder<Input: IntoCoordinates> {
    config: Config,
    _inner: Input,
}

impl<Input: IntoCoordinates> CoordinatesBuilder<Input> {
    pub fn minimum_length(mut self, v: u32) -> Self {
        self.config.minimum_length = v;
        self
    }

    pub fn vertex_spacing(mut self, v: usize) -> Self {
        self.config.vertex_spacing = v;
        self
    }

    pub fn root_vertices_on_first_level(mut self, v: bool) -> Self {
        self.config.root_vertices_on_first_level = v;
        self
    }
}


impl<'i, V, E> CoordinatesBuilder<&'i StableDiGraph<V, E>> {
    pub(super) fn build_layout_from_graph(graph: &'i StableDiGraph<V, E>) -> Self {
        Self {
            config: Config::default(),
            _inner: graph,
        }
    }

    pub fn build(self) -> Layouts<NodeIndex> {
        let Self { config , _inner: graph } = self;
        algorithm::build_layout_from_graph(graph, config).into_iter()
            .map(|(l, w, h)| 
                (
                    l.into_iter().map(|(id, coords)| (NodeIndex::from(id as u32), coords)).collect(),
                    w,
                    h
                )
            )
            .collect()
        }
}

impl<'i> CoordinatesBuilder<&'i [(u32, u32)]> {
    pub(super) fn build_layout_from_edges(edges: &'i[(u32, u32)]) -> Self {
        Self {
            config: Config::default(),
            _inner: edges,
        }
    }

    pub fn build(self) -> Layouts<usize> {
        let Self { config , _inner: edges } = self;
        algorithm::build_layout_from_edges(edges, config)
    }
}
