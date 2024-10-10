#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct DeweyRequest {
    pub query: String,
    pub filters: Vec<String>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct DeweyResponse {
    pub body: String,
}
