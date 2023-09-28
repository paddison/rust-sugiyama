use std::marker::PhantomData;

use log::trace;
use petgraph::stable_graph::{NodeIndex, StableDiGraph};

use crate::{
    algorithm::{self, Edge, Vertex},
    Config, CrossingMinimization, Layouts, RankingType,
};

pub trait IntoCoordinates {}

impl<V, E> IntoCoordinates for StableDiGraph<V, E> {}
impl IntoCoordinates for &[(u32, u32)] {}
impl IntoCoordinates for (&[u32], &[(u32, u32)]) {}

pub struct CoordinatesBuilder<Input: IntoCoordinates> {
    config: Config,
    _inner: StableDiGraph<Vertex, Edge>,
    pd: PhantomData<Input>,
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
        trace!(target: "initializing",
            "Setting minimum length to: {v}");
        self.config.minimum_length = v;
        self
    }

    pub fn vertex_spacing(mut self, v: usize) -> Self {
        trace!(target: "initializing",
            "Setting vertex spacing to: {v}");
        self.config.vertex_spacing = v;
        self
    }

    pub fn dummy_vertices(mut self, v: bool) -> Self {
        trace!(target: "initializing",
            "Has dummy vertices: {v}");
        self.config.dummy_vertices = v;
        self
    }

    pub fn layering_type(mut self, v: RankingType) -> Self {
        trace!(target: "initializing",
            "using layering type: {v:?}");
        self.config.layering_type = v;
        self
    }

    pub fn crossing_minimization(mut self, v: CrossingMinimization) -> Self {
        trace!(target: "initializing",
            "Heuristic for crossing minimization: {v:?}");
        self.config.c_minimization = v;
        self
    }

    pub fn transpose(mut self, v: bool) -> Self {
        trace!(target: "initializing",
            "Use transpose to further reduce crossings: {v}");
        self.config.transpose = v;
        self
    }

    pub fn dummy_size(mut self, v: f64) -> Self {
        trace!(target: "initializing",
            "Dummy size in regards to vertex size: {v}");
        self.config.dummy_size = v;
        self
    }
}

impl<V, E> CoordinatesBuilder<StableDiGraph<V, E>> {
    pub fn build(self) -> Layouts<NodeIndex> {
        let Self {
            config,
            _inner: graph,
            ..
        } = self;
        algorithm::start(
            graph.map(|_, _| Vertex::default(), |_, _| Edge::default()),
            config,
        )
        .into_iter()
        .map(|(l, w, h)| {
            (
                l.into_iter()
                    .map(|(id, coords)| (NodeIndex::from(id as u32), coords))
                    .collect(),
                w,
                h,
            )
        })
        .collect()
    }
}

impl CoordinatesBuilder<&[(u32, u32)]> {
    pub fn build(self) -> Layouts<usize> {
        let Self {
            config,
            _inner: graph,
            ..
        } = self;
        algorithm::start(graph, config)
    }
}

impl CoordinatesBuilder<(&[u32], &[(u32, u32)])> {
    pub fn build(self) -> Layouts<usize> {
        let Self {
            config,
            _inner: graph,
            ..
        } = self;
        algorithm::start(graph, config)
    }
}
