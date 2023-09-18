use configure::CoordinatesBuilder;
use petgraph::stable_graph::StableDiGraph;

mod algorithm;
mod util;
mod configure;

type Layout = (Vec<(usize, (isize, isize))>, usize, usize);
type Layouts<T> = Vec<(Vec<(T, (isize, isize))>, usize, usize)>;

#[derive(Clone, Copy)]
pub struct Config {
    minimum_length: u32,
    vertex_spacing: usize,
    root_vertices_on_first_level: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            minimum_length: 1,
            vertex_spacing: 10,
            root_vertices_on_first_level: false,
        }
    }
}

pub fn from_edges(edges: &[(u32, u32)]) -> CoordinatesBuilder<&[(u32, u32)]> {
    CoordinatesBuilder::build_layout_from_edges(edges)
}

pub fn from_graph<V, E>(graph: &StableDiGraph<V, E>) -> CoordinatesBuilder<&StableDiGraph<V, E>> {
    CoordinatesBuilder::build_layout_from_graph(graph)
}

#[cfg(test)]
mod benchmark {
    use super::from_edges;

    #[test]
    fn r_100() {
        let edges = graph_generator::RandomLayout::new(100).build_edges().into_iter().map(|(r, l)| (r as u32, l as u32)).collect::<Vec<(u32, u32)>>();
        let start = std::time::Instant::now();
        let _ = from_edges(&edges).build();
        println!("Random 100 edges: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn r_1000() {
        let edges = graph_generator::RandomLayout::new(1000).build_edges().into_iter().map(|(r, l)| (r as u32, l as u32)).collect::<Vec<(u32, u32)>>();
        let start = std::time::Instant::now();
        let _ = from_edges(&edges).build();
        println!("Random 1000 edges: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn r_2000() {
        let edges = graph_generator::RandomLayout::new(2000).build_edges();
        let start = std::time::Instant::now();
        let _ = from_edges(&edges).build();
        println!("Random 2000 edges: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn r_4000() {
        let edges = graph_generator::RandomLayout::new(4000).build_edges();
        let start = std::time::Instant::now();
        let _ = from_edges(&edges).build();
        println!("Random 4000 edges: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn l_1000_2() {
        let n = 1000;
        let e = 2;
        let edges = graph_generator::GraphLayout::new_from_num_nodes(n, e).build_edges();
        let start = std::time::Instant::now();
        let _ = from_edges(&edges).build();
        println!("{n} nodes, {e} edges per node: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn l_2000_2() {
        let n = 2000;
        let e = 2;
        let edges = graph_generator::GraphLayout::new_from_num_nodes(n, e).build_edges();
        let start = std::time::Instant::now();
        let _ = from_edges(&edges).build();
        println!("{n} nodes, {e} edges per node: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn l_4000_2() {
        let n = 4000;
        let e = 2;
        let edges = graph_generator::GraphLayout::new_from_num_nodes(n, e).build_edges();
        let start = std::time::Instant::now();
        let _ = from_edges(&edges).build();
        println!("{n} nodes, {e} edges per node: {}ms", start.elapsed().as_millis());
    }

    #[test]
    fn l_8000_2() {
        let n = 8000;
        let e = 2;
        let edges = graph_generator::GraphLayout::new_from_num_nodes(n, e).build_edges();
        let start = std::time::Instant::now();
        let _ = from_edges(&edges).build();
        println!("{n} nodes, {e} edges per node: {}ms", start.elapsed().as_millis());
    }
}

#[cfg(test)]
mod check_visuals {
    use super::from_edges;
    
    #[test]
    fn verify_looks_good() {
        // NOTE: This test might fail eventually, since the order of lements in a row canot be guaranteed;
        let edges = [
                (0, 1), 
                (1, 2), 
                (2, 3), (2, 4), 
                (3, 5), (3, 6), (3, 7), (3, 8), (4, 5), (4, 6), (4, 7), (4, 8),
                (5, 9), (6, 9), (7, 9), (8, 9)
        ];
        let (layout, width, height) = &mut from_edges(&edges).build()[0]; 
        layout.sort_by(|a, b| a.0.cmp(&b.0));

        assert_eq!(*width, 4);
        assert_eq!(*height,6);
        assert_eq!(layout[0], (0, (15, 0)));
        assert_eq!(layout[1], (1, (15, -10)));
        assert_eq!(layout[2], (2, (15, -20)));
        assert_eq!(layout[3], (3, (10, -30)));
        assert_eq!(layout[4], (4, (20, -30)));
        assert_eq!(layout[5], (5, (0, -40)));
        assert_eq!(layout[6], (6, (10, -40)));
        assert_eq!(layout[7], (7, (20, -40)));
        assert_eq!(layout[8], (8, (30, -40)));
        assert_eq!(layout[9], (9, (15, -50)));
        println!("{:?}", layout);
    }

    #[test]
    fn check_coords() {
        let edges = [(1, 0), (2, 1), (3, 0), (4, 0)];
        let layout = from_edges(&edges).build();
        println!("{:?}", layout);
    }

    #[test]
    fn check_coords_2() {
        let edges = [(0, 1), (0, 2), (0, 3), (1, 4), (4, 5), (5, 6), (2, 6), (3, 6), (3, 7), (3, 8), (3, 9)];
        let layout = from_edges(&edges).build();
        println!("{:?}", layout);
    }

}