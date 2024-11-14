use std::io::BufRead;

use dewey_lib::logger::Logger;
use dewey_lib::lprint;

struct TestServer {
    process: std::process::Child,
    port: u16,
}

impl TestServer {
    pub fn new() -> std::io::Result<Self> {
        let port = get_free_port();
        // this is assuming that the tests are being run from the workspace level
        let mut process = std::process::Command::new("./target/debug/dewey_server")
            .args(["-p", &port.to_string()])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .stdin(std::process::Stdio::null())
            .spawn()?;

        let start = std::time::Instant::now();
        let mut reader = std::io::BufReader::new(process.stdout.take().unwrap());
        let mut line = String::new();
        loop {
            if (std::time::Instant::now() - start).as_secs() > 5 {
                panic!("Error: timed out waiting for `dewey_server` regression testing stdout",);
            }

            match reader.read_line(&mut line) {
                Ok(_) => {
                    if line.contains("Compiled for regression testing: ") {
                        let split = line.split(": ").collect::<Vec<_>>();
                        if split.last().unwrap().trim() != "true" {
                            lprint!(error, "Regression check: {:?}", split);
                            panic!("Error: `dewey_server` isn't compiled for regression testing");
                        }

                        break;
                    } else {
                        if line.len() > 0 {
                            lprint!(info, "server process output: {}", line);
                        }
                    }
                }
                Err(e) => panic!("error reading server output: {}", e),
            }
        }

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

    let mut retries = 0;
    let max_retries = 5;

    while retries < max_retries {
        match client.query(String::from("testing"), 10, Vec::new()) {
            Ok(_) => break,
            Err(e) => {
                lprint!(info, "Attempt {} failed with error: {:?}", retries + 1, e);
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        }

        retries += 1;
    }

    assert!(retries < max_retries);

    let response = client.query(String::from("testing"), 10, Vec::new());
    assert!(response.is_ok());

    let response = response.unwrap();
    assert!(response.results.len() > 0);
}

macro_rules! test {
    ($func:ident($($arg:expr),*)) => {{
        print!("Test {}...\r", stringify!($func));
        $func($($arg),*);
        println!("Test {}... passed", stringify!($func));
    }}
}

// hideous code
// think something is funky with ownerhsip and when different things get dropped
// gotta be careful with the process handle + whatnot
fn main() {
    let _cleanup = dewey_lib::test_common::Cleanup;
    dewey_lib::test_common::setup().unwrap();

    let mut cli_process = std::process::Command::new("./target/debug/dewey")
        .args(["-rsebf"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .stdin(std::process::Stdio::null())
        .spawn()
        .unwrap();

    let start = std::time::Instant::now();
    let mut reader = std::io::BufReader::new(cli_process.stdout.take().unwrap());
    let mut line = String::new();
    loop {
        if (std::time::Instant::now() - start).as_secs() > 5 {
            panic!("Error: timed out waiting for `dewey` regression testing stdout",);
        }

        match reader.read_line(&mut line) {
            Ok(_) => {
                if line.contains("Compiled for regression testing: ") {
                    let split = line.split(": ").collect::<Vec<_>>();
                    if split.last().unwrap().trim() != "true" {
                        panic!("Error: `dewey` isn't compiled for regression testing");
                    }

                    break;
                } else {
                    if line.len() > 0 {
                        lprint!(info, "cli process output: {}", line);
                    }
                }
            }
            Err(e) => panic!("error reading server output: {}", e),
        }
    }

    cli_process.wait().unwrap();

    let server = TestServer::new().unwrap();
    test!(query_test(server.port as u32))
}
