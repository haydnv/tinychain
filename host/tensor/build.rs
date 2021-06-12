extern crate pkg_config;

fn main() {
    pkg_config::Config::new()
        .atleast_version("3.8")
        .probe("arrayfire")
        .unwrap();
}
