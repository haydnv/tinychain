pub use generic;
pub use value;

pub use auth;
pub use error;
pub use kernel::*;

#[allow(dead_code)]
mod route;

pub mod chain;
pub mod cluster;
pub mod fs;
pub mod gateway;
pub mod http;
pub mod kernel;
pub mod object;
pub mod scalar;
pub mod state;
pub mod txn;

pub mod testutils {
    use std::path::{Path, PathBuf};

    use error::TCResult;

    use super::fs;

    pub async fn setup<P: AsRef<Path>>(data_dir: &[P]) -> TCResult<fs::Dir> {
        let data_dir: PathBuf = data_dir.iter().collect();
        if data_dir.exists() {
            tokio::fs::remove_dir(&data_dir).await.unwrap();
        }

        tokio::fs::create_dir(&data_dir).await.unwrap();

        let cache = fs::Cache::new(0);
        fs::load(cache, data_dir).await
    }
}
