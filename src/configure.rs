use std::{env, marker::PhantomData};

use log::{error, trace};
use petgraph::stable_graph::{NodeIndex, StableDiGraph};

use crate::{
    algorithm::{self, Edge, Vertex},
    Layouts,
};

// Default values for configuration
pub static MINIMUM_LENGTH_DEFAULT: u32 = 1;
pub static VERTEX_SPACING_DEFAULT: usize = 10;
pub static DUMMY_VERTICES_DEFAULT: bool = true;
pub static RANKING_TYPE_DEFAULT: RankingType = RankingType::MinimizeEdgeLength;
pub static C_MINIMIZATION_DEFAULT: CrossingMinimization = CrossingMinimization::Barycenter;
pub static TRANSPOSE_DEFAULT: bool = true;
pub static DUMMY_SIZE_DEFAULT: f64 = 1.0;

static ENV_MINIMUM_LENGTH: &str = "RUST_GRAPH_MIN_LEN";
static ENV_VERTEX_SPACING: &str = "RUST_GRAPH_V_SPACING";
static ENV_DUMMY_VERTICES: &str = "RUST_GRAPH_DUMMIES";
static ENV_RANKING_TYPE: &str = "RUST_GRAPH_R_TYPE";
static ENV_CROSSING_MINIMIZATION: &str = "RUST_GRAPH_CROSS_MIN";
static ENV_TRANSPOSE: &str = "RUST_GRAPH_TRANSPOSE";
static ENV_DUMMY_SIZE: &str = "RUST_GRAPH_DUMMY_SIZE";

pub trait IntoCoordinates {}

impl<V, E> IntoCoordinates for StableDiGraph<V, E> {}
impl IntoCoordinates for &[(u32, u32)] {}
impl IntoCoordinates for (&[u32], &[(u32, u32)]) {}

macro_rules! read_env {
    ($field:expr, $cb:tt, $env:ident) => {
        #[allow(unused_parens)]
        match env::var($env).map($cb) {
            Ok(Ok(v)) => $field = v,
            Ok(Err(e)) => {
                error!(target: "initialization", "{e}");
            }
            _ => (),
        }
    };
}

/// Used to configure parameters of the graph layout.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Length between layers.
    pub minimum_length: u32,
    /// The minimum spacing between vertices on the same layer and between
    /// layers.
    pub vertex_spacing: usize,
    /// Whether to include dummy vertices when calculating the layout.
    pub dummy_vertices: bool,
    /// How much space a dummy should take up, as a multiplier of the
    /// [`Self::vertex_spacing`].
    pub dummy_size: f64,
    /// Defines how vertices are placed vertically.
    pub ranking_type: RankingType,
    /// Which heuristic to use when minimizing edge crossings.
    pub c_minimization: CrossingMinimization,
    /// Whether to attempt to further reduce crossings by swapping vertices in a
    /// layer. This may increase runtime significantly.
    pub transpose: bool,
}

impl Config {
    /// Create a new config by reading in environment variables.
    /// See [CoordinatesBuilder::configure_from_env] for a detailed description of environment variables.
    pub fn new_from_env() -> Self {
        let config = Self::default();
        config.read_env()
    }

    /// Updates the config by reading in environment variables.
    /// See [CoordinatesBuilder::configure_from_env] for a detailed description of environment variables.
    pub fn read_env(mut self) -> Self {
        let parse_bool = |x: String| match x.as_str() {
            "y" => Ok(true),
            "n" => Ok(false),
            v => Err(format!("Invalid argument for dummy vertex env: {v}")),
        };

        read_env!(
            self.minimum_length,
            (|x| x.parse::<u32>()),
            ENV_MINIMUM_LENGTH
        );

        read_env!(
            self.c_minimization,
            (TryFrom::try_from),
            ENV_CROSSING_MINIMIZATION
        );

        read_env!(self.ranking_type, (TryFrom::try_from), ENV_RANKING_TYPE);

        read_env!(
            self.vertex_spacing,
            (|x| x.parse::<usize>()),
            ENV_VERTEX_SPACING
        );

        read_env!(self.dummy_vertices, parse_bool, ENV_DUMMY_VERTICES);

        read_env!(self.dummy_size, (|x| x.parse::<f64>()), ENV_DUMMY_SIZE);

        read_env!(self.transpose, parse_bool, ENV_TRANSPOSE);

        self
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            minimum_length: MINIMUM_LENGTH_DEFAULT,
            vertex_spacing: VERTEX_SPACING_DEFAULT,
            dummy_vertices: DUMMY_VERTICES_DEFAULT,
            ranking_type: RANKING_TYPE_DEFAULT,
            c_minimization: C_MINIMIZATION_DEFAULT,
            transpose: TRANSPOSE_DEFAULT,
            dummy_size: DUMMY_SIZE_DEFAULT,
        }
    }
}

/// Defines the Ranking type, i.e. how vertices are placed on each layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RankingType {
    /// First moves vertices as far up as possible, and then as low as possible
    Original,
    /// Tries to minimize edge lengths across layers
    MinimizeEdgeLength,
    /// Move vertices as far up as possible
    Up,
    /// Move vertices as far down as possible
    Down,
}

impl TryFrom<String> for RankingType {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.as_str() {
            "original" => Ok(Self::Original),
            "minimize" => Ok(Self::MinimizeEdgeLength),
            "up" => Ok(Self::Up),
            "down" => Ok(Self::Down),
            s => Err(format!("invalid value for ranking type: {s}")),
        }
    }
}

impl From<RankingType> for &'static str {
    fn from(value: RankingType) -> Self {
        match value {
            RankingType::Up => "up",
            RankingType::Down => "down",
            RankingType::Original => "original",
            RankingType::MinimizeEdgeLength => "minimize",
        }
    }
}

/// Defines the heuristic used for crossing minimization.
/// During crossing minimization, the vertices of one layer are
/// ordered, so they're as close to neighboring vertices as possible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrossingMinimization {
    /// Calculates the average of the positions of adjacent neighbors
    Barycenter,
    /// Calculates the weighted median of the positions of adjacent neighbors
    Median,
}

impl TryFrom<String> for CrossingMinimization {
    type Error = String;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        match value.as_str() {
            "barycenter" => Ok(Self::Barycenter),
            "median" => Ok(Self::Median),
            s => Err(format!("invalid value for crossing minimization: {s}")),
        }
    }
}

impl From<CrossingMinimization> for &'static str {
    fn from(value: CrossingMinimization) -> Self {
        match value {
            CrossingMinimization::Median => "median",
            CrossingMinimization::Barycenter => "barycenter",
        }
    }
}

/// Can be used to configure the layout of the graph, via the builder pattern.
///
/// # Example
/// ```
/// use rust_sugiyama::from_edges;
/// let edges = [(0, 1), (0, 2), (2, 3)];
/// let coords = from_edges(&edges)
///     .vertex_spacing(20) // vertices are at least 20px apart
///     .dummy_vertices(false) // ignore dummy vertices when calculating layout
///     .transpose(false) // don't use tranpose function during crossing minimization
///     .build(); // build the layout
/// ```
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

    /// Set the minimimum length, see [Config] for description
    pub fn minimum_length(mut self, v: u32) -> Self {
        trace!(target: "initializing",
            "Setting minimum length to: {v}");
        self.config.minimum_length = v;
        self
    }

    /// Set the spacing between vertices, see [Config] for description
    pub fn vertex_spacing(mut self, v: usize) -> Self {
        trace!(target: "initializing",
            "Setting vertex spacing to: {v}");
        self.config.vertex_spacing = v;
        self
    }

    /// Activate/deactivate dummy vertices, see [Config] for description
    pub fn dummy_vertices(mut self, v: bool) -> Self {
        trace!(target: "initializing",
            "Has dummy vertices: {v}");
        self.config.dummy_vertices = v;
        self
    }

    /// Set the layering type, see [Config] for description
    pub fn layering_type(mut self, v: RankingType) -> Self {
        trace!(target: "initializing",
            "using layering type: {v:?}");
        self.config.ranking_type = v;
        self
    }

    /// Set the crossing minimization heuristic, see [Config] for description
    pub fn crossing_minimization(mut self, v: CrossingMinimization) -> Self {
        trace!(target: "initializing",
            "Heuristic for crossing minimization: {v:?}");
        self.config.c_minimization = v;
        self
    }

    /// Use transpose function during crossing minimization, see [Config]
    pub fn transpose(mut self, v: bool) -> Self {
        trace!(target: "initializing",
            "Use transpose to further reduce crossings: {v}");
        self.config.transpose = v;
        self
    }

    /// Set the size of the dummy vertices, see [Config]
    pub fn dummy_size(mut self, v: f64) -> Self {
        trace!(target: "initializing",
            "Dummy size in regards to vertex size: {v}");
        self.config.dummy_size = v;
        self
    }

    pub fn with_config(mut self, config: Config) -> Self {
        trace!(target: "initializing",
            "With config {:?}", config);
        self.config = config;
        self
    }

    /// Read in configuration values from environment variables.
    ///
    /// Envs that can be set include:
    ///
    /// | ENV | values | default | description |
    /// | --- | ------ | ------- | ----------- |
    /// | RUST_GRAPH_MIN_LEN    | integer, > 0         | 1          | minimum edge length between layers |
    /// | RUST_GRAPH_V_SPACING  | integer, > 0         | 10         | minimum spacing between vertices on the same layer |
    /// | RUST_GRAPH_DUMMIES    | y \| n               | y          | if dummy vertices are included in the final layout |
    /// | RUST_GRAPH_R_TYPE     | original \| minimize \| up \| down | minimize   | defines how vertices are places vertically |
    /// | RUST_GRAPH_CROSS_MIN  | barycenter \| median | barycenter | which heuristic to use for crossing reduction |
    /// | RUST_GRAPH_TRANSPOSE  | y \| n               | y          | if transpose function is used to further try to reduce crossings (may increase runtime significally for large graphs) |
    /// | RUST_GRAPH_DUMMY_SIZE | float, 1 >= v > 0    | 1.0        |size of dummy vertices in final layout, if dummy vertices are included. this will squish the graph horizontally |
    pub fn configure_from_env(mut self) -> Self {
        self.config = self.config.read_env();
        self
    }
}

impl<V, E> CoordinatesBuilder<StableDiGraph<V, E>> {
    /// Build the layout.
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
    /// Build the layout.
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
    /// Build the layout.
    pub fn build(self) -> Layouts<usize> {
        let Self {
            config,
            _inner: graph,
            ..
        } = self;
        algorithm::start(graph, config)
    }
}

#[test]
fn from_env_all_valid() {
    use super::from_edges;
    use std::env;
    let edges = [(1, 2), (2, 3)];
    env::set_var(ENV_MINIMUM_LENGTH, "5");
    env::set_var(ENV_DUMMY_VERTICES, "y");
    env::set_var(ENV_DUMMY_SIZE, "0.1");
    env::set_var(ENV_RANKING_TYPE, "up");
    env::set_var(ENV_CROSSING_MINIMIZATION, "median");
    env::set_var(ENV_TRANSPOSE, "n");
    env::set_var(ENV_VERTEX_SPACING, "20");
    let cfg = from_edges(&edges).configure_from_env();
    assert_eq!(cfg.config.minimum_length, 5);
    assert_eq!(cfg.config.dummy_vertices, true);
    assert_eq!(cfg.config.dummy_size, 0.1);
    assert_eq!(cfg.config.ranking_type, RankingType::Up);
    assert_eq!(cfg.config.c_minimization, CrossingMinimization::Median);
    assert_eq!(cfg.config.transpose, false);
    assert_eq!(cfg.config.vertex_spacing, 20);
}

#[test]
fn from_env_invalid_value() {
    use super::from_edges;
    use std::env;

    let edges = [(1, 2)];
    env::set_var(ENV_CROSSING_MINIMIZATION, "flubbeldiflap");
    env::set_var(ENV_VERTEX_SPACING, "1.5");
    let cfg = from_edges(&edges);
    let default = Config::default();
    assert_eq!(default.c_minimization, cfg.config.c_minimization);
    assert_eq!(default.vertex_spacing, cfg.config.vertex_spacing);
}

#[test]
fn run_algo_empty_graph() {
    use super::from_edges;
    let edges = [];
    let g = from_edges(&edges).build();
    assert!(g.is_empty());
}
