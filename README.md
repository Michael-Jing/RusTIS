RusTIS
===

RusTIS is a rut wrapper around the Triton Inference Server

# How to run
1. open the code in a docker container that with image nvcr.io/nvidia/tritonserver
2. put Triton Inference Server compatible models into the models folder
3. `cargo run` and the rpc server will listen on :50051
4. now you can use Triton Inference Server grpc client to send request to the "127.0.0.1:50051" 