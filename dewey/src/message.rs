use crate::serialization::Serialize;

use serialize_macros::Serialize;

// TODO: message type enum for the serialization macros

#[derive(Serialize)]
pub struct Message {
    pub message_type: String,
    pub body: String,
}
