use std::env;

use log::error;
use petgraph::stable_graph::StableDiGraph;

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
///
/// Struct fields are:
/// 1. minimum_edge: length between layers
/// 2. vertex_spacing: minimum spacing between vertices on the same layer
/// 3. dummy_vertices: should dummie vertices be included when calculating the layout
/// 4. ranking_type: defines how vertices are places vertically, see [RankingType]
/// 5. c_minimization: which heuristic to use when minimizing edge crossings, see [CrossingMinimization]
/// 6. transpose: try to further reduce crossings, by swaping vertices in a layer, may increase runtime significantly
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub minimum_length: u32,
    pub vertex_spacing: usize,
    pub dummy_vertices: bool,
    pub dummy_size: f64,
    pub ranking_type: RankingType,
    pub c_minimization: CrossingMinimization,
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

#[test]
fn from_env_all_valid() {
    use std::env;
    env::set_var(ENV_MINIMUM_LENGTH, "5");
    env::set_var(ENV_DUMMY_VERTICES, "y");
    env::set_var(ENV_DUMMY_SIZE, "0.1");
    env::set_var(ENV_RANKING_TYPE, "up");
    env::set_var(ENV_CROSSING_MINIMIZATION, "median");
    env::set_var(ENV_TRANSPOSE, "n");
    env::set_var(ENV_VERTEX_SPACING, "20");
    let cfg = Config::new_from_env();
    assert_eq!(cfg.minimum_length, 5);
    assert_eq!(cfg.dummy_vertices, true);
    assert_eq!(cfg.dummy_size, 0.1);
    assert_eq!(cfg.ranking_type, RankingType::Up);
    assert_eq!(cfg.c_minimization, CrossingMinimization::Median);
    assert_eq!(cfg.transpose, false);
    assert_eq!(cfg.vertex_spacing, 20);
}

#[test]
fn from_env_invalid_value() {
    use std::env;

    env::set_var(ENV_CROSSING_MINIMIZATION, "flubbeldiflap");
    env::set_var(ENV_VERTEX_SPACING, "1.5");
    let cfg = Config::new_from_env();
    let default = Config::default();
    assert_eq!(default.c_minimization, cfg.c_minimization);
    assert_eq!(default.vertex_spacing, cfg.vertex_spacing);
}
