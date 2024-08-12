use std::env;
use std::io::{BufRead, Read, Write};
use std::net::TcpStream;

use crate::logger::Logger;
use crate::{error, info};

#[derive(Debug)]
struct RequestParams {
    host: String,
    path: String,
    port: u16,
    model: String,
    authorization_token: String,
}

#[derive(Debug)]
pub struct EmbeddingSource {
    pub filepath: String,
    pub subset: Option<(usize, usize)>,
}

#[derive(Debug)]
pub struct Embedding {
    pub data: [f32; 1536],
    pub source_file: EmbeddingSource,
}

fn read_source(source: &EmbeddingSource) -> Result<String, std::io::Error> {
    let mut file = match std::fs::File::open(&source.filepath) {
        Ok(file) => file,
        Err(e) => {
            error!("Failed to open file: {:?}", e);
            return Err(e);
        }
    };

    let mut contents = String::new();
    match file.read_to_string(&mut contents) {
        Ok(_) => (),
        Err(e) => {
            error!("Failed to read from file: {:?}", e);
            return Err(e);
        }
    }

    // TODO: we can easily get away without loading the entire file into memory
    let contents = match source.subset {
        Some((start, end)) => contents[start..std::cmp::min(end, contents.len())].to_string(),
        _ => contents,
    };

    Ok(contents)
}

pub fn embed(source: EmbeddingSource) -> Result<Embedding, std::io::Error> {
    let params = RequestParams {
        host: "api.openai.com".to_string(),
        path: "/v1/embeddings".to_string(),
        port: 443,
        model: "text-embedding-3-small".to_string(),
        authorization_token: env::var("OPENAI_API_KEY")
            .expect("OPENAI_API_KEY environment variable not set"),
    };

    let contents = read_source(&source)?;

    let stream = TcpStream::connect((params.host.clone(), params.port))?;

    let connector = native_tls::TlsConnector::new().expect("Failed to create TLS connector");
    let mut stream = connector
        .connect(&params.host, stream)
        .expect("Failed to establish TLS connection");

    info!("Connected to OpenAI API");

    let body = serde_json::json!({
        "model": params.model,
        "input": contents
    });
    let json = serde_json::json!(body);
    let json_string = serde_json::to_string(&json)?;

    let auth_string = "Authorization: Bearer ".to_string() + &params.authorization_token;

    let request = format!(
        "POST {} HTTP/1.1\r\n\
        Host: {}\r\n\
        Content-Type: application/json\r\n\
        Content-Length: {}\r\n\
        Accept: */*\r\n\
        {}\r\n\r\n\
        {}",
        params.path,
        params.host,
        json_string.len(),
        auth_string,
        json_string.trim()
    );

    info!("Request: {}", request);

    stream.write_all(request.as_bytes())?;
    stream.flush()?;

    info!("Sent request to OpenAI API");

    let mut reader = std::io::BufReader::new(&mut stream);

    let mut headers = String::new();
    let mut content_length = None;
    loop {
        let mut buffer = String::new();

        reader.read_line(&mut buffer)?;
        reader.read_line(&mut buffer)?;

        info!("Buffer: {}", buffer);

        if buffer.is_empty() || buffer.contains("\r\n\r\n") {
            break;
        }

        if buffer.to_lowercase().contains("content-length:") {
            content_length = Some(
                buffer
                    .split_whitespace()
                    .last()
                    .unwrap()
                    .parse::<usize>()
                    .unwrap(),
            );
        }

        headers.push_str(&buffer);
    }

    info!("headers: {:?}", headers);

    let mut body = String::new();
    match content_length {
        Some(content_length) => {
            reader
                .take(content_length as u64)
                .read_to_string(&mut body)?;
        }
        _ => {
            panic!("Content-Length header not found");
        }
    }

    let response_json = serde_json::from_str(&body);

    if response_json.is_err() {
        error!("Failed to parse JSON: {}", body);
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Failed to parse JSON",
        ));
    }

    info!("Parsed JSON response");

    let response_json: serde_json::Value = response_json.unwrap();
    let mut embedding = Embedding {
        data: [0.0; 1536],
        source_file: source,
    };

    // TODO: batching
    for (i, value) in response_json["data"][0]["embedding"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        embedding.data[i] = value.as_f64().unwrap() as f32;
    }

    Ok(embedding)
}
