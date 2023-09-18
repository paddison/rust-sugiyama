use std::marker::PhantomData;

use petgraph::stable_graph::{StableDiGraph, NodeIndex};

use crate::{Layouts, algorithm::{self, Vertex, Edge}, Config};

pub trait IntoCoordinates {}

impl<V, E> IntoCoordinates for StableDiGraph<V, E> {}
impl IntoCoordinates for &[(u32, u32)] {}
impl IntoCoordinates for (&[u32], &[(u32, u32)]) {}

pub struct CoordinatesBuilder<Input: IntoCoordinates> {
    config: Config,
    _inner: StableDiGraph<Vertex, Edge>,
    pd: PhantomData<Input>
}

impl<Input: IntoCoordinates> CoordinatesBuilder<Input> {
    pub(super) fn new(graph: StableDiGraph<Vertex, Edge>) -> Self {
        Self {
            config: Config::default(),
            _inner: graph,
            pd: PhantomData,
        }
    }

    pub fn minimum_length(mut self, v: u32) -> Self {
        self.config.minimum_length = v;
        self
    }

    pub fn vertex_spacing(mut self, v: usize) -> Self {
        self.config.vertex_spacing = v;
        self
    }

    pub fn root_vertices_on_top(mut self, v: bool) -> Self {
        self.config.root_vertices_on_top = v;
        self
    }

    pub fn no_dummy_vertices(mut self, v: bool) -> Self {
        self.config.no_dummy_vertices = v;
        self
    }
}

impl<V, E> CoordinatesBuilder<StableDiGraph<V, E>> {
    pub fn build(self) -> Layouts<NodeIndex> {
        let Self { config , _inner: graph, .. } = self;
        algorithm::start(graph.map(|_, _| Vertex::default(), |_, _| Edge::default()), config).into_iter()
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

impl CoordinatesBuilder<&[(u32, u32)]> {

    pub fn build(self) -> Layouts<usize> {
        let Self { config , _inner: graph , ..} = self;
        algorithm::start(graph, config)
    }
}

impl CoordinatesBuilder<(&[u32], &[(u32, u32)])> {
    pub fn build(self) -> Layouts<usize> {
        let Self { config , _inner: graph , ..} = self;
        algorithm::start(graph, config)
    }
}