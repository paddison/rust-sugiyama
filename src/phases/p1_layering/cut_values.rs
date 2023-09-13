use std::collections::VecDeque;

use petgraph::{stable_graph::{StableDiGraph, EdgeIndex, NodeIndex}, Direction::{Incoming, Outgoing, self}, visit::EdgeRef};

use super::{Vertex, Edge, NeighborhoodInfo};


pub(super) fn init_cutvalues(graph: &mut StableDiGraph<Vertex, Edge>) {
    // TODO: check if it is faster to collect tree edges or to do unecessary iterations
    // let tree_edges = self.graph.edge_indices().filter(|e| self.graph[*e].is_tree_edge).collect::<HashSet<_>>();
    let queue = leaves(graph);
    // traverse tree inward via breadth first starting from leaves
    calculate_cut_values(graph, queue);
}

pub(super) fn update_cutvalues(graph: &mut StableDiGraph<Vertex, Edge>, removed_edge: EdgeIndex, swap_edge: EdgeIndex) -> NodeIndex {
    let least_common_ancestor = remove_outdated_cut_values(graph, swap_edge, removed_edge);
    let queue = VecDeque::from([graph.edge_endpoints(removed_edge).unwrap().0]);
    calculate_cut_values(graph, queue);
    least_common_ancestor
}

fn leaves(graph: &StableDiGraph<Vertex, Edge>) -> VecDeque<NodeIndex> {
    graph.node_indices()
                .filter(|v| 
                1 == graph.edges_directed(*v, Incoming)
                            .chain(graph.edges_directed(*v, Outgoing))
                            .filter(|e| e.weight().is_tree_edge)
                            .count())
                .collect::<VecDeque<_>>()
}

fn calculate_cut_values(graph: &mut StableDiGraph<Vertex, Edge>, mut queue: VecDeque<NodeIndex>) {
    while let Some(vertex) = queue.pop_front() {
        let incoming = get_neighborhood_info(graph, vertex, Incoming); 
        let outgoing = get_neighborhood_info(graph, vertex, Outgoing); 

        // if we can't calculate cut value yet, or the value is already known
        let (mut incoming, mut outgoing) = match (incoming, outgoing) {
            (Some(inc), Some(out)) => (inc, out),
            _ => continue,
        };

        let missing = match (incoming.missing, outgoing.missing) {
            (Some(u), None) => u,
            (None, Some(v)) => v,
            _ => continue,
        };

        let edge = match graph.find_edge(vertex, missing) {
            Some(e) => {
                // switch direction, if vertex is tail component of edge
                std::mem::swap(&mut incoming, &mut outgoing);
                e
            },
            None => graph.find_edge(missing, vertex).unwrap()

        };

        graph[edge].cut_value = Some(calculate_cut_value(graph[edge].weight, incoming, outgoing));
        // continue traversing tree in direction of edge whose vertex was missing before
        queue.push_back(missing);
    }
}

fn calculate_cut_value(edge_weight: i32, incoming: NeighborhoodInfo, outgoing: NeighborhoodInfo) -> i32 {
    edge_weight 
    + incoming.non_tree_edge_weight_sum - incoming.cut_value_sum + incoming.tree_edge_weight_sum
    - outgoing.non_tree_edge_weight_sum + outgoing.cut_value_sum - outgoing.tree_edge_weight_sum
}

fn get_neighborhood_info(graph: &StableDiGraph<Vertex, Edge>, vertex: NodeIndex, direction: Direction) -> Option<NeighborhoodInfo> {
    // return the sum of all cut values,
    // sum of weights of cut value edges
    // missing cut value (only if there is one)
    // sum of weights of edges who are not tree edges
    let mut cut_value_sum = 0;
    let mut tree_edge_weight_sum = 0;
    let mut non_tree_edge_weight_sum = 0;
    let mut missing = None;

    for edge in graph.edges_directed(vertex, direction) {
        let (tail, head) = (edge.source(), edge.target());
        let edge = *edge.weight();
        if !edge.is_tree_edge {
            non_tree_edge_weight_sum += edge.weight;
        } else if let Some(cut_value) = edge.cut_value {
            cut_value_sum += cut_value;
            tree_edge_weight_sum += edge.weight;
        } else if missing.is_none() {
            missing = Some(if tail == vertex { head } else { tail });
        } else {
            return None;
        }
    }
    Some(
        NeighborhoodInfo {
            cut_value_sum,
            tree_edge_weight_sum,
            non_tree_edge_weight_sum,
            missing,
        }
    )
}

pub(super) fn remove_outdated_cut_values(graph: &mut StableDiGraph<Vertex, Edge>, swap_edge: EdgeIndex, removed_edge: EdgeIndex) -> NodeIndex {
    graph[removed_edge].cut_value = None;
    let (mut w, mut x)  = graph.edge_endpoints(swap_edge).unwrap();
    if graph[w].lim > graph[x].lim {
        std::mem::swap(&mut w, &mut x)
    }
    // follow path back until least common ancestor is found
    // and remove cut_values on the way
    let least_common_ancestor = match graph[w].parent {
        None => w,
        Some(mut parent) => {
            let mut l = w;
            loop {
                let edge = graph.find_edge_undirected(l, parent).unwrap().0;
                graph[edge].cut_value = None;
                l = parent;
                if graph[l].low <= graph[w].lim && graph[x].lim <= graph[l].lim || graph[l].parent.is_none() {
                    break l;
                }
                parent = graph[l].parent.unwrap();
            }
        }
    };
    // record path from x to l
    // we don't need to care about the order in which the edges are added,
    // since we only need them to remove the outdated cutvalues.
    let mut l = x;
    while l != least_common_ancestor {
        let parent = graph[l].parent.unwrap();
        let edge = graph.find_edge_undirected(l, parent).unwrap().0;
        graph[edge].cut_value = None;
        l = parent;
    }

    least_common_ancestor
}

#[cfg(test)]
mod tests {

    use crate::phases::p1_layering::{tests::{EXAMPLE_GRAPH_FEASIBLE_TREE_NEG_CUT_VALUE, EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE, GraphBuilder, EXAMPLE_GRAPH}, cut_values::init_cutvalues};


    #[test]
    fn test_cut_values_one_negative() {
        let edges = EXAMPLE_GRAPH_FEASIBLE_TREE_NEG_CUT_VALUE; 
        let (mut graph, ..) = GraphBuilder::new(&EXAMPLE_GRAPH).with_tree_edges(&edges).build();

        init_cutvalues(&mut graph);

        assert_eq!(graph[graph.find_edge(edges[0].0.into(), edges[0].1.into()).unwrap()].cut_value, Some(3));
        assert_eq!(graph[graph.find_edge(edges[1].0.into(), edges[1].1.into()).unwrap()].cut_value, Some(3));
        assert_eq!(graph[graph.find_edge(edges[2].0.into(), edges[2].1.into()).unwrap()].cut_value, Some(3));
        assert_eq!(graph[graph.find_edge(edges[3].0.into(), edges[3].1.into()).unwrap()].cut_value, Some(3));
        assert_eq!(graph[graph.find_edge(edges[4].0.into(), edges[4].1.into()).unwrap()].cut_value, Some(0));
        assert_eq!(graph[graph.find_edge(edges[5].0.into(), edges[5].1.into()).unwrap()].cut_value, Some(0));
        assert_eq!(graph[graph.find_edge(edges[6].0.into(), edges[6].1.into()).unwrap()].cut_value, Some(-1));
    }

    #[test]
    fn test_cut_values_all_positive() {
        let edges = EXAMPLE_GRAPH_FEASIBLE_TREE_POS_CUT_VALUE;
        let (mut graph, ..) = GraphBuilder::new(&EXAMPLE_GRAPH).with_tree_edges(&edges).build();

        init_cutvalues(&mut graph);

        assert_eq!(graph[graph.find_edge(edges[0].0.into(), edges[0].1.into()).unwrap()].cut_value, Some(2));
        assert_eq!(graph[graph.find_edge(edges[1].0.into(), edges[1].1.into()).unwrap()].cut_value, Some(1));
        assert_eq!(graph[graph.find_edge(edges[2].0.into(), edges[2].1.into()).unwrap()].cut_value, Some(2));
        assert_eq!(graph[graph.find_edge(edges[3].0.into(), edges[3].1.into()).unwrap()].cut_value, Some(2));
        assert_eq!(graph[graph.find_edge(edges[4].0.into(), edges[4].1.into()).unwrap()].cut_value, Some(2));
        assert_eq!(graph[graph.find_edge(edges[5].0.into(), edges[5].1.into()).unwrap()].cut_value, Some(1));
        assert_eq!(graph[graph.find_edge(edges[6].0.into(), edges[6].1.into()).unwrap()].cut_value, Some(0));
    }
}