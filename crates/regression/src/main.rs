struct TestServer {
    process: std::process::Child,
    port: u16,
}

impl TestServer {
    pub fn new() -> std::io::Result<Self> {
        let port = get_free_port();
        // this is assuming that the tests are being run from the workspace level
        let process = std::process::Command::new("./target/debug/dewey_server")
            .args(["-p", &port.to_string()])
            .spawn()?;

        Ok(Self { process, port })
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.process.kill().expect("Failed to kill server");
        self.process.wait().expect("Failed to wait for server");
    }
}

fn get_free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")
        .expect("Failed to bind address while trying to get free port");
    listener
        .local_addr()
        .expect("Failed to get listener local_addr")
        .port()
}

fn query_test(port: u32) {
    let client = dewey_lib::DeweyClient {
        address: String::from("127.0.0.1"),
        port,
    };

    // Wait for server to be ready
    let mut retries = 0;
    let max_retries = 5;

    while retries < max_retries {
        match client.query(String::from("testing"), 10, Vec::new()) {
            Ok(_) => break,
            Err(e) => {
                println!("Attempt {} failed with error: {:?}", retries + 1, e);
            }
        }

        retries += 1;
    }

    assert!(retries < 5);

    let response = client.query(String::from("testing"), 10, Vec::new());
    assert!(response.is_ok());
}

fn main() {
    let server = TestServer::new().unwrap();
    std::thread::sleep(std::time::Duration::from_millis(500));
    println!("server initialized");

    query_test(server.port as u32);
}
