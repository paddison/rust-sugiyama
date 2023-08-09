use petgraph::stable_graph::{StableDiGraph, NodeIndex};

use super::layers::Layers;

pub(crate) trait LayerGraph<'a ,T: Default + 'a>{
    fn get_actual_graph(&self) -> &StableDiGraph<Option<T>, usize>;
    fn get_layers(&self) -> &Layers;

    fn get_upper_neighbours(&'a self, dest: NodeIndex) -> &[NodeIndex]  {
        self.get_layers().get_upper_neighbours(dest)
    }

    fn get_lower_neighbours(&'a self, source: NodeIndex) -> &[NodeIndex] {
        self.get_layers().get_lower_neighbours(source)
    }

    fn get_position(&self, vertex: NodeIndex) -> usize {
        let layers = self.get_layers();
        layers.get_position(vertex)
    }
}

#[macro_export]
macro_rules! impl_layer_graph{
    ($t:ty) => {
        impl<'a, T: Default + 'a> LayerGraph<'a, T> for $t {
            fn get_actual_graph(&self) -> &StableDiGraph<Option<T>, usize> {
                &self.graph
            }

            fn get_layers(&self) -> &Layers {
                &self.layers
            }
        } 
    };
}