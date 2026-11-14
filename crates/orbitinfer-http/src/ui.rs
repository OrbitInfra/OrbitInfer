// Adapted from Resin a1989594 for OrbitInfer; changes: docs/source-import.md.
pub const INDEX_HTML: &str = include_str!("index.html");

pub fn index_html(routed_dp_size: usize) -> String {
    INDEX_HTML.replace(
        "name=\"orbitinfer-dp-size\" content=\"1\"",
        &format!(
            "name=\"orbitinfer-dp-size\" content=\"{}\"",
            routed_dp_size.max(1)
        ),
    )
}
