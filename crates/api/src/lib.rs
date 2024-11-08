use std::io::{Read, Write};

use dewey_lib::error;
use dewey_lib::logger::Logger;

pub use dewey_lib::message;

pub struct DeweyClient {
    pub address: String,
    pub port: u32,
}

impl DeweyClient {
    pub fn new(address: String, port: u32) -> Self {
        Self { address, port }
    }

    fn send(
        &self,
        message: message::DeweyRequest,
    ) -> Result<message::DeweyResponse, std::io::Error> {
        let mut stream = std::net::TcpStream::connect(format!("{}:{}", self.address, self.port))?;

        let message_bytes = serde_json::to_string(&message)?.into_bytes();
        stream.write(&message_bytes)?;
        stream.flush()?;

        let mut length_bytes = [0u8; 4];
        stream.read_exact(&mut length_bytes)?;
        let length = u32::from_be_bytes(length_bytes) as usize;

        let mut buffer = vec![0u8; length];
        stream.read_exact(&mut buffer)?;
        let buffer = String::from_utf8_lossy(&buffer);

        match serde_json::from_str(&buffer) {
            Ok(resp) => Ok(resp),
            Err(e) => {
                error!("Failed to parse response: {}", e);
                error!("buffer: {:?}", buffer);
                return Err(e.into());
            }
        }
    }

    pub fn query(
        &self,
        request: String,
        k: usize,
        filters: Vec<String>,
    ) -> Result<message::DeweyResponse, std::io::Error> {
        let message = message::DeweyRequest {
            message_type: "query".to_string(),
            payload: message::RequestPayload::Query {
                query: request,
                k,
                filters,
            },
        };

        self.send(message)
    }

    pub fn reindex(&self, filepath: String) -> Result<message::DeweyResponse, std::io::Error> {
        let message = message::DeweyRequest {
            message_type: "edit".to_string(),
            payload: message::RequestPayload::Edit { filepath },
        };

        self.send(message)
    }
}
