use std::collections::HashSet;

use log::{debug, info};
use petgraph::stable_graph::{NodeIndex, StableDiGraph};

pub fn weakly_connected_components<V: Copy, E: Copy>(
    graph: StableDiGraph<V, E>,
) -> Vec<StableDiGraph<V, E>> {
    info!(target: "connected_components", "Splitting graph into its connected components");
    let mut components = Vec::new();
    let mut visited = HashSet::new();

    for node in graph.node_indices() {
        if visited.contains(&node) {
            continue;
        }

        let component_nodes = component_dfs(node, &graph);
        let component = graph.filter_map(
            |n, w| {
                if component_nodes.contains(&n) {
                    Some(*w)
                } else {
                    None
                }
            },
            |_, w| Some(*w),
        );

        component_nodes.into_iter().for_each(|n| {
            visited.insert(n);
        });
        components.push(component);
    }
    debug!(target: "connected_components", "Found {} components", components.len());

    components
}

fn component_dfs<V: Copy, E: Copy>(
    start: NodeIndex,
    graph: &StableDiGraph<V, E>,
) -> HashSet<NodeIndex> {
    let mut queue = vec![start];
    let mut visited = HashSet::new();

    visited.insert(start);

    while let Some(cur) = queue.pop() {
        for neighbor in graph.neighbors_undirected(cur) {
            if visited.contains(&neighbor) {
                continue;
            }
            visited.insert(neighbor);
            queue.push(neighbor);
        }
    }

    visited
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
