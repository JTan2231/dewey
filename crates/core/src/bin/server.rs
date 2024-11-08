use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::thread;

use dewey_lib::config;
use dewey_lib::logger::Logger;
use dewey_lib::message::DeweyRequest;
use dewey_lib::serialization::Serialize;
use dewey_lib::{error, info};

struct Flags {
    address: String,
    port: usize,
}

fn parse_flags() -> Flags {
    let args: Vec<String> = std::env::args().collect();
    let mut flags = Flags {
        address: String::from("127.0.0.1"),
        port: 5050,
    };

    if args.len() < 1 {
        std::process::exit(1);
    }

    for (i, arg) in args.iter().skip(1).enumerate() {
        if arg.starts_with("-") && !arg.starts_with("--") {
            for c in arg.chars().skip(1) {
                match c {
                    'a' => {
                        flags.address = args[i + 2].clone();
                    }
                    'p' => {
                        flags.port = args[i + 2].parse().unwrap();
                    }
                    _ => panic!("error: unknown flag: {}", c),
                }
            }
        }
    }

    flags
}

pub fn main() -> std::io::Result<()> {
    config::setup();
    let flags = parse_flags();

    let listener = TcpListener::bind(format!("{}:{}", flags.address, flags.port)).unwrap();
    info!("Server listening on {}:{}", flags.address, flags.port);
    println!("Server listening on {}:{}", flags.address, flags.port);

    let state = Arc::new(Mutex::new(dewey_lib::ServerState::new()?));

    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                let state = Arc::clone(&state);
                thread::spawn(move || {
                    let mut state = state.lock().unwrap();

                    let mut size_buffer = [0u8; 4];
                    stream.read_exact(&mut size_buffer).unwrap();
                    let message_size = u32::from_be_bytes(size_buffer) as usize;

                    let mut buffer = vec![0u8; message_size];
                    stream.read_exact(&mut buffer).unwrap();

                    let request: DeweyRequest =
                        match serde_json::from_str(&String::from_utf8_lossy(&buffer)) {
                            Ok(r) => r,
                            Err(e) => {
                                error!("Error parsing request: {}", e);
                                return;
                            }
                        };

                    let response = match request.message_type.as_str() {
                        "query" => match state.query(request.payload) {
                            Ok(r) => r,
                            Err(e) => {
                                let r = format!("Error handling client: {}", e);
                                error!("{}", r);
                                r
                            }
                        },
                        "edit" => match state.reindex(request.payload) {
                            Ok(r) => r,
                            Err(e) => {
                                let r = format!("Error handling client: {}", e);
                                error!("{}", r);
                                r
                            }
                        },
                        _ => format!("Invalid message_type: {}", request.message_type),
                    };

                    let mut bytes = Vec::new();
                    bytes.extend((response.len() as u32).to_be_bytes());
                    bytes.extend_from_slice(response.as_bytes());

                    match stream.write(&response.to_bytes()) {
                        Ok(bytes_written) => {
                            stream.flush().unwrap();
                            info!("wrote {} bytes to stream", bytes_written);
                        }
                        Err(e) => {
                            error!("Failed to write response: {}", e);
                        }
                    };
                });
            }
            Err(e) => {
                info!("Error: {}", e);
            }
        }
    }

    Ok(())
}
