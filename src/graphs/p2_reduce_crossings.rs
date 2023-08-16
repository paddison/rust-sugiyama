use petgraph::stable_graph::StableDiGraph;

use crate::{util::layers::Layers, impl_layer_graph};
use crate::util::traits::LayerGraph;

// create from LayeredGraph
pub struct ProperLayeredGraph<T: Default> {
    layers: Layers,
    graph: StableDiGraph<Option<T>, usize>
}

impl_layer_graph!(ProperLayeredGraph<T>);