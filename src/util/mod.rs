use std::collections::{HashSet, VecDeque, HashMap};

use petgraph::{stable_graph::{StableDiGraph, NodeIndex}, Direction::{Incoming, Outgoing}};

/// Takes a graph and breaks it down into its weakly connected components.
/// A weakly connected component is a list of edges which are connected with each other.
pub fn into_weakly_connected_components<V, E>(mut graph: StableDiGraph<V, E>) -> Vec<StableDiGraph<V, E>> {
    // map graph weights to options so we can take them later while building the subgraphs
    let mut sub_graphs = Vec::new();

    // build each subgraph
    while graph.node_count() > 0 {
        // since node count is > 0, calling unwrap is safe.
        let vertex = graph.node_indices().next().unwrap();

        // if the graph is a single node graph
        if graph.neighbors_undirected(vertex).count() == 0 {
            let weight = graph.remove_node(vertex).unwrap();
            let mut g = StableDiGraph::new();
            g.add_node(weight);
            sub_graphs.push(g);
            continue;
        }

        let sub_graph_raw = determine_sub_graph(vertex, &mut graph);
        
        sub_graphs.push(build_sub_graph(sub_graph_raw, &mut graph));
    }

    return sub_graphs
}

fn build_sub_graph<V, E>(raw_parts: Vec<(NodeIndex, NodeIndex, E)>, graph: &mut StableDiGraph<V, E>) -> StableDiGraph<V, E> {
        // remove all edges from graph. store references to nodeindices with them.
        // build new graph:
        // start with first edge. create a set hashmap to lookup nodeindices 
        let mut indices = HashMap::<NodeIndex, NodeIndex>::new();
        let mut sub_graph = StableDiGraph::new();
        for (from, to, weight) in raw_parts {
            let sg_from = match indices.get(&from) {
                Some(id) => *id,
                None => {
                    let id = sub_graph.add_node(graph.remove_node(from).unwrap());
                    indices.insert(from, id);
                    id
                }
            };
            let sg_to = match indices.get(&to) {
                Some(id) => *id,
                None => {
                    let id = sub_graph.add_node(graph.remove_node(to).unwrap());
                    indices.insert(to, id);
                    id
                }
            };
            sub_graph.add_edge(sg_from, sg_to, weight);
        }
        sub_graph
}

fn determine_sub_graph<V, E>(vertex: NodeIndex, graph: &mut StableDiGraph<V, E>) -> Vec<(NodeIndex, NodeIndex, E)> {
        let mut visited = HashSet::<NodeIndex>::new();
        let mut edges = Vec::<(NodeIndex, NodeIndex, E)>::new();

        // traverse via bfs, remove edges as we go
        let mut queue = VecDeque::from([vertex]);
        while let Some(vertex) = queue.pop_front() {
            visited.insert(vertex);
            let mut incoming_walker = graph.neighbors_directed(vertex, Incoming).detach();
            let mut outgoing_walker = graph.neighbors_directed(vertex, Outgoing).detach();
            while let Some((e, n)) = incoming_walker.next(&graph) {
                if !visited.contains(&n) {
                    let edge_weight = graph.remove_edge(e).unwrap();
                    edges.push((n, vertex, edge_weight));
                    queue.push_back(n);
                }
            }
            while let Some((e, n)) = outgoing_walker.next(&graph) {
                if !visited.contains(&n) {
                    let edge_weight = graph.remove_edge(e).unwrap();
                    edges.push((vertex, n, edge_weight));
                    queue.push_back(n);
                }
            }
        }

        edges
}

#[test]
fn into_weakly_connected_components_two_components() {
    let g = StableDiGraph::<usize, usize>::from_edges([(0, 1), (1, 2), (3, 2), (4, 5), (4, 6)]);
    let sgs = into_weakly_connected_components(g);
    assert_eq!(sgs.len(), 2);
    assert!(sgs[0].contains_edge(0.into(), 1.into()));
    assert!(sgs[0].contains_edge(1.into(), 2.into()));
    assert!(sgs[0].contains_edge(3.into(), 2.into()));
    assert!(sgs[1].contains_edge(0.into(), 1.into()));
    assert!(sgs[1].contains_edge(0.into(), 2.into()));
}

// todo: Refactor this into trait
pub(super) fn iterate(dir: IterDir, length: usize) -> impl Iterator<Item = usize> {
    let (mut start, step) = match dir {
        IterDir::Forward => (usize::MAX, 1), // up corresponds to left to right
        IterDir::Backward => (length, usize::MAX),
    };
    std::iter::repeat_with(move || {
            start = start.wrapping_add(step);
            start
        }).take(length)
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum IterDir {
    Forward,
    Backward,
}

// TODO: implement this with binary and see if it is faster
pub(super) fn radix_sort(mut input: Vec<usize>, key_length: usize) -> Vec<usize> {
    let mut output = vec![0; input.len()];

    let mut key = 1;
    for _ in 0..key_length {
        counting_sort(&mut input, key, &mut output);
        key *= 10;
    }
    input
}

#[inline(always)]
fn counting_sort(input: &mut [usize], key: usize, output: &mut [usize]) {
    let mut count = [0; 10]; 
    // insert initial counts
    for i in input.iter().map(|n| self::key(key, *n)) {
        count[i] += 1;
    }

    // built accumulative sum
    for i in 1..10 {
        count[i] += count[i - 1];
    }

    for i in (0..input.len()).rev() {
        let k = self::key(key, input[i]);
        count[k] = count[k] - 1;
        output[count[k]] = input[i];
    }
    for i in 0..input.len() {
        input[i] = output[i];
    }
}

#[inline(always)]
fn key(key: usize, n: usize) -> usize {
    (n / key) % 10
}

#[test]
fn test_counting_sort_first_digit() {
    let mut input = [10, 0, 1, 5, 4, 22, 12];
    let mut output = [0; 7];
    counting_sort(&mut input, 1, &mut output);
    assert_eq!(output, [10, 0, 1, 22, 12, 4, 5])
}

#[test]
fn test_counting_sort_second_digit() {
    let mut input = [10, 0, 1, 5, 4, 22, 12];
    let mut output = [0; 7];
    counting_sort(&mut input, 10, &mut output);
    assert_eq!(output, [0, 1, 5, 4, 10, 12, 22]);
}

#[test]
fn test_radix_sort() {
    let input = [10, 0, 1, 5, 4, 22, 12];
    let output = radix_sort(input.to_vec(), 2);
    assert_eq!(output, [0, 1, 4, 5, 10, 12, 22]);
}
