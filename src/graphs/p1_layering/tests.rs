use petgraph::stable_graph::{StableDiGraph, NodeIndex};

use super::{InitialRanks, UnlayeredGraph, Vertex, Edge};

static EXAMPLE_GRAPH: [(u32, u32); 7] = [(0, 1), (1, 2), (2, 3), (3, 7), (4, 6), (5, 6), (6, 7)];
static LOW_LIM_GRAPH: [(u32, u32); 8] = [(0, 1), (1, 2), (1, 3), (0, 4), (4, 5), (5, 6), (4, 7), (4, 8)];
static FEASIBLE_TREE_NEG_CUT_VALUE: [(u32, u32); 7] = [(0, 1), (1, 2), (2, 3), (3, 7), (4, 6), (5, 6), (6, 7)];
static FEASIBLE_TREE_POS_CUT_VALUE: [(u32, u32); 7] = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 6), (5, 6)];
trait GraphBuilder {
    type Target;

    fn from_edges(mut self, edges: &[(u32, u32)]) -> Self 
    where Self: Sized 
    {

        *self.inner() = Some(StableDiGraph::from_edges(edges));
        self
    }

    fn with_minimum_length(mut self, minimum_length: u32) -> Self 
    where Self: Sized 
    {
        *self.minimum_length() = minimum_length as i32;
        self
    }

    fn inner(&mut self) -> &mut Option<StableDiGraph<Vertex, Edge>>; 
    fn minimum_length(&mut self) -> &mut i32;
    fn build(self) -> Self::Target;
}

macro_rules! impl_graph_builder {
    ($t:ty, $target:ty) => {
        impl GraphBuilder for $t {
            type Target = $target;

            fn inner(&mut self) -> &mut Option<StableDiGraph<Vertex, Edge>> {
                &mut self._inner
            } 

            fn minimum_length(&mut self) -> &mut i32 {
                &mut self.minimum_length
            }

            fn build(self) -> Self::Target {
                let Self { _inner, minimum_length } = self;
                let graph = match _inner {
                    Some(g) => g,
                    None => StableDiGraph::new(),
                };
                Self::Target { graph, minimum_length }
            }
        }
    };
}


struct Builder;

impl Builder {
    fn build_unlayered() -> UnlayeredGraphBuilder {
        UnlayeredGraphBuilder { _inner: None, minimum_length: 1 }
    }
    
    fn build_initial_ranks() -> InitialRanksBuilder {
        InitialRanksBuilder { _inner: None, minimum_length: 1 }
    }

}



struct UnlayeredGraphBuilder {
    _inner: Option<StableDiGraph<Vertex, Edge>>,
    minimum_length: i32,
}

impl_graph_builder!(UnlayeredGraphBuilder, UnlayeredGraph);

impl UnlayeredGraphBuilder {
    fn build(self) -> UnlayeredGraph {
        let graph = match self._inner {
            Some(g) => g,
            None => StableDiGraph::new(),
        };
        UnlayeredGraph { graph, minimum_length: self.minimum_length }
    }
}
struct InitialRanksBuilder {
    _inner: Option<StableDiGraph<Vertex, Edge>>,
    minimum_length: i32,
}

impl_graph_builder!(InitialRanksBuilder, InitialRanks);

impl InitialRanksBuilder {
}

mod test_initial_ranks {
    use petgraph::Direction::{Incoming, Outgoing};

    use super::{Builder, GraphBuilder, EXAMPLE_GRAPH};

    #[test]
    fn test_initial_ranking_correct_order() {
        let initial_ranks = Builder::build_unlayered()
                             .from_edges(&EXAMPLE_GRAPH)
                             .build()
                             .init_rank();
        
        let g = &initial_ranks.graph;

        for v in g.node_indices() {
            // all incoming neighbors need to have lower ranks,
            for inc in g.neighbors_directed(v, Incoming) {
                assert!(g[v].rank > g[inc].rank);
            }
            // all outgoing higher
            for out in g.neighbors_directed(v, Outgoing) {
                assert!(g[v].rank < g[out].rank);
            }
        }
    }

    #[test]
    fn test_initial_ranking_at_least_minimum_length_2() {
        let actual = Builder::build_unlayered()
                             .from_edges(&EXAMPLE_GRAPH)
                             .with_minimum_length(2)
                             .build()
                             .init_rank();

        let g = &actual.graph;

        for v in g.node_indices() {
            for n in g.neighbors_undirected(v) {
                assert!(g[v].rank.abs_diff(g[n].rank) as i32 >= actual.minimum_length )
            }
        }
    }
} 