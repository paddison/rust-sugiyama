use petgraph::stable_graph::{StableDiGraph, NodeIndex};

use super::layers::Layers;

pub(crate) trait LayerGraph<'a ,T: Default + 'a>{
    fn get_actual_graph(&self) -> &StableDiGraph<Option<T>, usize>;
    fn get_layers(&self) -> &Layers;

    fn get_upper_neighbours(&'a self, dest: NodeIndex) -> Vec<NodeIndex>  {
        let graph = self.get_actual_graph();
        let layers = self.get_layers();
        let dest_level = layers.get_level(dest);
        if dest_level == 0 {
            Vec::new()  
        } else {
            layers[dest_level - 1].iter().filter(move |source| graph.contains_edge(**source, dest)).cloned().collect()
        }
    }

    fn get_lower_neighbours(&'a self, source: NodeIndex) -> Vec<NodeIndex> {
        let graph = self.get_actual_graph();
        let layers = self.get_layers();
        let source_level = layers.get_level(source);
        if source_level == layers.height() - 1 {
            Vec::new()  
        } else {
            layers[source_level + 1].iter().filter(move |dest| graph.contains_edge(source, **dest)).cloned().collect()
        }
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