// TODO: better error handling: error-stack?
pub mod error;
pub mod inference_request;
pub mod inference_response;
pub mod message;
// TODO: add prometheus metrics publishing
pub mod metrics;
// TODO: learn from cpp code, try to avoid copy when creating rpc server responses
pub mod response_allocator;
pub mod server;
// TODO: add more server options support
pub mod server_options;
pub mod utils;
// TODO: s3 download / model registry support
