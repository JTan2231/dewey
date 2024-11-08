#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct QueryRequest {
    pub k: usize,
    pub query: String,
    pub filters: Vec<String>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct EditRequest {
    pub filepath: String,
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
