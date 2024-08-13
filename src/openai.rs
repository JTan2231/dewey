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

#[derive(Debug, Clone)]
pub struct EmbeddingSource {
    pub filepath: String,
    pub subset: Option<(u64, u64)>,
}

impl EmbeddingSource {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let u64_size = std::mem::size_of::<u64>();

        let filepath_len = u64::from_be_bytes(bytes[0..u64_size].try_into().unwrap()) as usize;
        let filepath =
            String::from_utf8(bytes[u64_size..u64_size + filepath_len].to_vec()).unwrap();
        let start = u64::from_be_bytes(
            bytes[u64_size + filepath_len..u64_size + filepath_len + u64_size]
                .try_into()
                .unwrap(),
        );
        let end = u64::from_be_bytes(
            bytes[u64_size + filepath_len + u64_size..u64_size + filepath_len + 2 * u64_size]
                .try_into()
                .unwrap(),
        );

        let subset = match (start, end) {
            (0, 0) => None,
            _ => Some((start, end)),
        };

        Self { filepath, subset }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        let filepath_len = self.filepath.len() as u64;

        bytes.extend_from_slice(&filepath_len.to_be_bytes());
        bytes.extend_from_slice(&self.filepath.as_bytes());
        bytes.extend_from_slice(&self.subset.unwrap_or((0, 0)).0.to_be_bytes());
        bytes.extend_from_slice(&self.subset.unwrap_or((0, 0)).1.to_be_bytes());

        bytes
    }
}

#[derive(Debug)]
pub struct Embedding {
    pub source_file: EmbeddingSource,
    pub data: [f32; 1536],
}

impl Embedding {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let u64_size = std::mem::size_of::<u64>();
        let f32_size = std::mem::size_of::<f32>();

        let source_size = u64::from_be_bytes(bytes[0..u64_size].try_into().unwrap()) as usize;
        let source = EmbeddingSource::from_bytes(&bytes[u64_size..u64_size + source_size]);

        let mut data = [0.0; 1536];
        for (i, value) in bytes[u64_size + source_size..]
            .chunks(f32_size)
            .map(|chunk| f32::from_be_bytes(chunk.try_into().unwrap()))
            .enumerate()
        {
            data[i] = value;
        }

        Self {
            source_file: source,
            data,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        Vec::new()
    }
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
        Some((start, end)) => contents[start as usize..end as usize].to_string(),
        _ => contents,
    };

    Ok(contents)
}
pub fn embed(sources: &Vec<EmbeddingSource>) -> Result<Vec<Embedding>, std::io::Error> {
    let params = RequestParams {
        host: "api.openai.com".to_string(),
        path: "/v1/embeddings".to_string(),
        port: 443,
        model: "text-embedding-3-small".to_string(),
        authorization_token: env::var("OPENAI_API_KEY")
            .expect("OPENAI_API_KEY environment variable not set"),
    };

    // API requests need batched up to keep from exceeding token limits
    // TODO: a proper tokenizer
    const TOKEN_LIMIT: usize = 8192;

    let mut batches: Vec<Vec<(EmbeddingSource, String)>> = vec![Vec::new()];
    let mut split = batches.last_mut().unwrap();
    let mut split_len = 0;
    for source in sources {
        let source_contents = read_source(&source)?;
        let mut contents_split = Vec::new();
        let mut offset = 0;
        while offset < source_contents.len() {
            let end = std::cmp::min(offset + TOKEN_LIMIT, source_contents.len());
            contents_split.push(source_contents[offset..end].to_string());
            offset = end;
        }

        for contents in contents_split {
            if contents.len() + split_len >= TOKEN_LIMIT {
                batches.push(Vec::new());

                split = batches.last_mut().unwrap();
                split_len = 0;
            }

            split_len += contents.len();
            split.push((source.clone(), contents));
        }
    }

    let mut sizes = Vec::new();
    for batch in batches.iter() {
        sizes.push(
            batch
                .iter()
                .map(|pair| pair.1.len())
                .collect::<Vec<usize>>()
                .iter()
                .sum::<usize>(),
        );
    }

    let mut embeddings = Vec::new();
    for batch in batches {
        let stream = TcpStream::connect((params.host.clone(), params.port))?;

        let connector = native_tls::TlsConnector::new().expect("Failed to create TLS connector");
        let mut stream = connector
            .connect(&params.host, stream)
            .expect("Failed to establish TLS connection");

        info!("Connected to OpenAI API");

        let body = serde_json::json!({
            "model": params.model,
            "input": batch.iter().map(|pair| pair.1.clone()).collect::<Vec<String>>(),
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
        let data = response_json["data"].as_array().unwrap();

        for (i, datum) in data.iter().enumerate() {
            let mut embedding = Embedding {
                data: [0.0; 1536],
                source_file: batch[i].0.clone(),
            };

            for (i, value) in datum["embedding"].as_array().unwrap().iter().enumerate() {
                embedding.data[i] = value.as_f64().unwrap() as f32;
            }

            embeddings.push(embedding);
        }
    }

    Ok(embeddings)
}
