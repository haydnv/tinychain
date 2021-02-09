use tinychain::testutils;

const DATA_DIR: [&str; 3] = ["/tmp", "tctest", "data"];

#[tokio::test]
async fn test() {
    let _data_dir = testutils::setup(&DATA_DIR).await.unwrap();
    assert!(true);
}
