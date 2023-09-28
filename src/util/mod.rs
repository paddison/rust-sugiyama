use std::collections::HashMap;

use petgraph::stable_graph::{NodeIndex, StableDiGraph};

/// Takes a graph and breaks it down into its weakly connected components.
/// A weakly connected component is a list of edges which are connected with each other.
pub fn weakly_connected_components<V: Copy, E: Copy>(
    graph: StableDiGraph<V, E>,
) -> Vec<StableDiGraph<V, E>> {
    // map graph weights to options so we can take them later while building the subgraphs
    let mut sub_graphs = Vec::new();

    let (components, n_components) = find_components(&graph);

    // build each subgraph
    for component in 0..n_components {
        let sub_graph = graph.filter_map(
            |v, w| match components.get(&v) {
                Some(c) if *c == component => Some(*w),
                _ => None,
            },
            |e, w| match graph.edge_endpoints(e) {
                Some((t, _)) if components.get(&t) == Some(&component) => Some(*w),
                _ => None,
            },
        );
        sub_graphs.push(sub_graph);
    }

    sub_graphs
}

fn find_components<V, E>(graph: &StableDiGraph<V, E>) -> (HashMap<NodeIndex, usize>, usize) {
    // traverse via dfs
    let mut components = HashMap::new();
    let mut n_components = 0;

    for v in graph.node_indices() {
        if !components.contains_key(&v) {
            find_component_dfs(v, graph, &mut components, n_components);
            n_components += 1;
        }
    }

    (components, n_components)
}

fn find_component_dfs<V, E>(
    vertex: NodeIndex,
    graph: &StableDiGraph<V, E>,
    components: &mut HashMap<NodeIndex, usize>,
    n_components: usize,
) {
    let mut queue = vec![vertex];
    while let Some(v) = queue.pop() {
        if components.contains_key(&v) {
            continue;
        }
        components.insert(v, n_components);
        for n in graph.neighbors_undirected(v) {
            queue.push(n);
        }
    }
}

#[test]
fn into_weakly_connected_components_two_components() {
    let g = StableDiGraph::<usize, usize>::from_edges([(0, 1), (1, 2), (3, 2), (4, 5), (4, 6)]);
    let sgs = weakly_connected_components(g);
    assert_eq!(sgs.len(), 2);
    assert!(sgs[0].contains_edge(0.into(), 1.into()));
    assert!(sgs[0].contains_edge(1.into(), 2.into()));
    assert!(sgs[0].contains_edge(3.into(), 2.into()));
    assert!(sgs[1].contains_edge(4.into(), 5.into()));
    assert!(sgs[1].contains_edge(4.into(), 6.into()));
}

// TODO: refactor into trait
// disable warnings, since we might still need this someday
#[allow(dead_code)]
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

#[allow(dead_code)]
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
        count[k] -= 1;
        output[count[k]] = input[i];
    }
    input.copy_from_slice(&output[..input.len()]);
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
