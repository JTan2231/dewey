#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct DeweyRequest {
    pub message_type: String,
    pub payload: RequestPayload,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum RequestPayload {
    Query {
        k: usize,
        query: String,
        filters: Vec<String>,
    },
    Edit {
        filepath: String,
    },
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct DeweyResponseItem {
    pub filepath: String,
    pub subset: (u64, u64),
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct DeweyResponse {
    pub results: Vec<DeweyResponseItem>,
}
