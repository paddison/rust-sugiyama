use std::collections::{HashMap, HashSet, VecDeque};

use petgraph::{
    stable_graph::{NodeIndex, StableDiGraph, EdgeIndex},
    Direction::{Incoming, Outgoing},
};

/// Takes a graph and breaks it down into its weakly connected components.
/// A weakly connected component is a list of edges which are connected with each other.
pub fn into_weakly_connected_components<V: Copy, E: Copy>(
    mut graph: StableDiGraph<V, E>,
) -> Vec<StableDiGraph<V, E>> {
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

        let (vertices, edges) = find_component(vertex, &mut graph);
        let component = graph.filter_map(|n, w| vertices.get(&n).map(|_| *w), |e, w| edges.get(&e).map(|_| *w));
        graph.retain_nodes(|_, v| !vertices.contains(&v));
        sub_graphs.push(component);
    }

    return sub_graphs;
}

fn find_component<V, E>(vertex: NodeIndex, graph: &StableDiGraph<V, E>, ) -> (HashSet<NodeIndex>, HashSet<EdgeIndex>) {
    // traverse via dfs
    let mut edges = HashSet::new();
    let mut queue = VecDeque::from([vertex]);
    let mut vertices = HashSet::from([vertex]);

    while let Some(vertex) = queue.pop_front() {
        let mut neighbors = graph.neighbors_undirected(vertex).detach();
        while let Some((e, n)) = neighbors.next(graph) {
            if edges.contains(&e) {
                continue;
            }
            vertices.insert(n);
            edges.insert(e);
            queue.push_front(n);
        }

    }
    (vertices, edges)
}

#[test]
fn into_weakly_connected_components_two_components() {
    let g = StableDiGraph::<usize, usize>::from_edges([(0, 1), (1, 2), (3, 2), (4, 5), (4, 6)]);
    let sgs = into_weakly_connected_components(g);
    assert_eq!(sgs.len(), 2);
    assert!(sgs[0].contains_edge(0.into(), 1.into()));
    assert!(sgs[0].contains_edge(1.into(), 2.into()));
    assert!(sgs[0].contains_edge(3.into(), 2.into()));
    assert!(sgs[1].contains_edge(4.into(), 5.into()));
    assert!(sgs[1].contains_edge(4.into(), 6.into()));
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
    })
    .take(length)
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
