use graph_coordinates::algorithm::{calculate_coordinates, g_levels};
fn main() {
    let g = g_levels(13);
    calculate_coordinates(g);
}