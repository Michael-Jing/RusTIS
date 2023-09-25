#![feature(vec_into_raw_parts)]
use anyhow::Result;
use log::{error, info, warn};
// use infer_proto::infer_proto::{ServerLiveRequest, ServerLiveResponse};
use infer_proto::infer_proto::grpc_inference_service_server::{
    GrpcInferenceService, GrpcInferenceServiceServer,
};
use infer_proto::infer_proto::{
    ModelInferRequest, ModelInferResponse, ModelMetadataRequest, ModelMetadataResponse,
    ModelReadyRequest, ModelReadyResponse, ServerLiveRequest, ServerLiveResponse,
    ServerMetadataRequest, ServerMetadataResponse, ServerReadyRequest, ServerReadyResponse,
};
use tonic::{transport::Server, Request, Response, Status};

#[derive(Default)]
pub struct RPCServer {
    infer_server: triton_sys::wrapper::server::Server,
}

#[tonic::async_trait]
impl GrpcInferenceService for RPCServer {
    async fn model_infer(
        &self,
        request: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        let reply = self
            .infer_server
            .infer(Box::new(request.into_inner()), 5_000_000)
            .await;
        match reply {
            Ok(r) => return Ok(Response::new(r)),
            Err(e) => {
                error!("error in rpc: {:?}", e);
                return Err(Status::internal(format!("error: {:?}", e)));
            }
        }
    }

    async fn server_live(
        &self,
        _request: Request<ServerLiveRequest>,
    ) -> Result<Response<ServerLiveResponse>, Status> {
        let reply = self.infer_server.is_live();
        match reply {
            Ok(r) => return Ok(Response::new(ServerLiveResponse { live: r })),
            Err(e) => {
                error!("error in rpc: {:?}", e);
                return Err(Status::internal(format!("error: {:?}", e)));
            }
        }
    }

    async fn server_ready(
        &self,
        _request: Request<ServerReadyRequest>,
    ) -> Result<Response<ServerReadyResponse>, Status> {
        let reply = self.infer_server.is_ready();
        match reply {
            Ok(r) => return Ok(Response::new(ServerReadyResponse { ready: r })),
            Err(e) => return Err(Status::internal(format!("error: {:?}", e))),
        }
    }

    async fn model_ready(
        &self,
        request: Request<ModelReadyRequest>,
    ) -> Result<Response<ModelReadyResponse>, Status> {
        let request = request.into_inner();
        let model_version = match request.version.parse::<i64>() {
            Ok(v) => v,
            Err(e) => return Err(Status::internal(format!("error: {:?}", e))),
        };

        let reply = self
            .infer_server
            .is_model_ready(request.name.as_ref(), model_version);
        match reply {
            Ok(r) => return Ok(Response::new(ModelReadyResponse { ready: r })),
            Err(e) => return Err(Status::internal(format!("error: {:?}", e))),
        }
    }

    async fn server_metadata(
        &self,
        _request: Request<ServerMetadataRequest>,
    ) -> Result<Response<ServerMetadataResponse>, Status> {
        let reply = self.infer_server.metadata();
        match reply {
            Ok(r) => {
                let obj: ServerMetadataResponse = match serde_json::from_str(&r) {
                    Ok(o) => o,
                    Err(e) => {
                        return Err(Status::internal(format!("deserialize json error: {:?}", e)))
                    }
                };
                return Ok(Response::new(obj));
            }
            Err(e) => return Err(Status::internal(format!("error: {:?}", e))),
        }
    }

    async fn model_metadata(
        &self,
        request: Request<ModelMetadataRequest>,
    ) -> Result<Response<ModelMetadataResponse>, Status> {
        let request = request.into_inner();
        let model_name = request.name;
        let model_version = match request.version.parse::<i64>() {
            Ok(v) => v,
            Err(e) => return Err(Status::internal(format!("error: {:?}", e))),
        };
        let reply = self.infer_server.model_metadata(&model_name, model_version);
        match reply {
            Ok(r) => {
                let obj: ModelMetadataResponse = match serde_json::from_str(&r) {
                    Ok(o) => o,
                    Err(e) => {
                        return Err(Status::internal(format!("deserialize json error: {:?}", e)))
                    }
                };
                return Ok(Response::new(obj));
            }
            Err(e) => return Err(Status::internal(format!("error: {:?}", e))),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if let Err(e) = initlog() {
        warn!("init logger failed with error: {}", e);
    }
    let addr = "127.0.0.1:50051".parse().unwrap();
    let rpc_server = RPCServer::default();

    info!("RPCServer listening on {}", addr);

    Server::builder()
        .add_service(GrpcInferenceServiceServer::new(rpc_server))
        .serve(addr)
        .await?;

    Ok(())
}

fn initlog() -> Result<()> {
    fern::Dispatch::new()
        // Perform allocation-free log formatting
        .format(|out, message, record| {
            out.finish(format_args!(
                "[{} {} {}] {}",
                humantime::format_rfc3339(std::time::SystemTime::now()),
                record.level(),
                record.target(),
                message
            ))
        })
        // Add blanket level filter -
        .level(log::LevelFilter::Debug)
        // - and per-module overrides
        // .level_for("hyper", log::LevelFilter::Info)
        // Output to stdout, files, and other Dispatch configurations
        .chain(std::io::stdout())
        .chain(fern::log_file("output.log")?)
        // Apply globally
        .apply()?;
    Ok(())
}
