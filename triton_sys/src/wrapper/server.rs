use std::{
    collections::HashMap,
    ffi::{c_void, CString},
    ptr::{self, null_mut},
};

use log::{error, info};

use crate::sys::*;
use anyhow::Result;
use infer_proto::infer_proto;
use tokio::sync::mpsc;

use super::{
    error::{InferError, TritonError},
    inference_request::{InferRequest, RequestDataHolder},
    inference_response::InferResponse,
    message::TritonMessage,
    response_allocator::ResponseAllocator,
    server_options::ServerOptions,
    utils,
};

pub struct Server {
    pub _server: *mut TRITONSERVER_Server,
    pub response_allocator: ResponseAllocator,
}

impl Server {
    pub fn raw_pointer(self) -> *mut TRITONSERVER_Server {
        self._server
    }
    pub fn new(options: ServerOptions) -> Result<Self, TritonError> {
        let mut server = ptr::null_mut() as *mut TRITONSERVER_Server;
        let err = unsafe {
            TRITONSERVER_ServerNew(
                &mut server as *mut *mut TRITONSERVER_Server,
                options._options,
            )
        };
        if !err.is_null() {
            return Err(TritonError { _err: err });
        }
        let response_allocator = ResponseAllocator::new().unwrap();
        Ok(Server {
            _server: server,
            response_allocator,
        })
    }

    pub fn is_live(&self) -> Result<bool, TritonError> {
        let mut live = false;
        let err = unsafe { TRITONSERVER_ServerIsLive(self._server, &mut live as *mut bool) };
        if !err.is_null() {
            return Err(TritonError { _err: err });
        }
        Ok(live)
    }

    pub fn is_ready(&self) -> Result<bool, TritonError> {
        let mut ready = false;
        let err = unsafe { TRITONSERVER_ServerIsReady(self._server, &mut ready as *mut bool) };
        if !err.is_null() {
            return Err(TritonError { _err: err });
        }
        Ok(ready)
    }

    pub fn load_model(&self, model_name: &str) -> Result<(), TritonError> {
        let c_model_name = std::ffi::CString::new(model_name).unwrap();
        let err = unsafe { TRITONSERVER_ServerLoadModel(self._server, c_model_name.as_ptr()) };
        if !err.is_null() {
            return Err(TritonError { _err: err });
        }
        Ok(())
    }

    pub fn is_model_ready(
        &self,
        model_name: &str,
        model_version: i64,
    ) -> Result<bool, TritonError> {
        let c_model_name = CString::new(model_name).unwrap();
        let mut ready = false;
        let err = unsafe {
            TRITONSERVER_ServerModelIsReady(
                self._server,
                c_model_name.as_ptr(),
                model_version,
                &mut ready,
            )
        };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        Ok(ready)
    }

    pub fn metadata(&self) -> Result<String, TritonError> {
        let mut server_metadata: *mut TRITONSERVER_Message = null_mut();
        let err = unsafe { TRITONSERVER_ServerMetadata(self._server, &mut server_metadata) };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        let json_message = TritonMessage::from_ptr(server_metadata).to_serialized_json()?;
        Ok(json_message)
    }

    pub fn model_metadata(
        &self,
        model_name: &str,
        model_version: i64,
    ) -> Result<String, TritonError> {
        let mut model_metadata: *mut TRITONSERVER_Message = null_mut();
        let c_model_name = CString::new(model_name).unwrap();
        let err = unsafe {
            TRITONSERVER_ServerModelMetadata(
                self._server,
                c_model_name.as_ptr(),
                model_version,
                &mut model_metadata,
            )
        };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        let json_message = TritonMessage::from_ptr(model_metadata).to_serialized_json()?;
        Ok(json_message)
    }
    // TODO: add InferenceTrace
    pub fn infer_async(&self, request: &mut InferRequest) -> Result<(), TritonError> {
        let err = unsafe {
            TRITONSERVER_ServerInferAsync(self._server, request.raw_pointer(), ptr::null_mut())
        };
        if !err.is_null() {
            return Err(TritonError { _err: err });
        }
        Ok(())
    }

    pub fn parse_pb_request(
        &self,
        pb_request: Box<infer_proto::ModelInferRequest>,
    ) -> Result<InferRequest> {
        let raw_input_length = pb_request.raw_input_contents.len();
        let input_length = pb_request.inputs.len();
        if raw_input_length != 0 && raw_input_length != input_length {
            return Err(InferError::new(
                "raw_input_contents and inputs must be of same length".to_string(),
            )
            .into());
        }
        let model_version = pb_request.model_version.parse::<i64>().unwrap_or(-1);
        let mut infer_request = match InferRequest::new(
            self,
            &pb_request.model_name,
            model_version,
            raw_input_length > 0,
        ) {
            Ok(r) => r,
            Err(e) => return Err(InferError::new(e.msg()).into()),
        };
        if let Err(e) = self.copy_request_data(&mut infer_request, pb_request) {
            infer_request.delete();
            return Err(e);
        };
        Ok(infer_request)
    }

    #[allow(non_upper_case_globals)]
    fn copy_request_data(
        &self,
        infer_request: &mut InferRequest,
        pb_request: Box<infer_proto::ModelInferRequest>,
    ) -> Result<()> {
        match infer_request.set_id(&pb_request.id) {
            Ok(_) => (),
            Err(e) => return Err(InferError::new(e.msg()).into()),
        };
        let raw_input_length = pb_request.raw_input_contents.len();
        let mut bytes_slice = vec![];
        for (i, input_tensor) in pb_request.inputs.iter().enumerate() {
            let data_type = utils::string_to_data_type(&input_tensor.datatype);
            if data_type == TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INVALID {
                return Err(InferError::new("Invalid datatype".to_owned()).into());
            }
            // check shape
            for s in &input_tensor.shape {
                if *s < 0 {
                    return Err(InferError::new("Invalid shape".to_string()).into());
                }
            }
            match infer_request.add_input(&input_tensor.name, data_type, &input_tensor.shape) {
                Ok(_) => (),
                Err(e) => return Err(InferError::new(e.msg()).into()),
            };
            if raw_input_length != 0 {
                let length = pb_request.raw_input_contents[i].len();
                    infer_request
                        .append_input_data_generic1(
                            &input_tensor.name,
                            pb_request.raw_input_contents[i].as_ref(),
                            length,
                            TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU,
                            0,
                        )
                        .map_err(|e| InferError::new(e.msg()))?;
            } else {
                match data_type {
                    TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BOOL => {
                        if let Some(contents) = &input_tensor.contents {
                            let data = contents.bool_contents.as_ref();
                                infer_request
                                    .append_input_data_generic1(
                                        &input_tensor.name,
                                        data,
                                        contents.bool_contents.len()
                                            * utils::data_type_size(data_type),
                                        TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU,
                                        0,
                                    )
                                    .map_err(|e| InferError::new(e.msg()))?;
                            
                        }
                    }
                    TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT8 => {
                        if let Some(contents) = &input_tensor.contents {
                            // write data back to input tensor's memory location
                            let raw_ptr = contents.uint_contents.as_ptr() as *mut u8;
                            for (i, &d) in contents.uint_contents.iter().enumerate() {
                                unsafe {
                                    raw_ptr.add(i).write(d as u8);
                                }
                            }
                                infer_request
                                    .append_input_data_generic2(
                                        &input_tensor.name,
                                        contents.uint_contents.as_ref(),
                                        data_type,
                                        TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU,
                                        0,
                                    )
                                    .map_err(|e| InferError::new(e.msg()))?;
                        }
                    }
                    TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT16 => {
                        if let Some(contents) = &input_tensor.contents {
                            let raw_ptr = contents.uint_contents.as_ptr() as *mut u16;
                            for (i, &d) in contents.uint_contents.iter().enumerate() {
                                unsafe {
                                    *raw_ptr.add(i) = d as u16;
                                }
                            }
                                infer_request
                                    .append_input_data_generic2(
                                        &input_tensor.name,
                                        
                                        contents.uint_contents.as_ref(),
                                            data_type,
                                        TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU,
                                        0,
                                    )
                                    .map_err(|e| InferError::new(e.msg()))?;
                        }
                    }
                    TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT32 => {
                        if let Some(contents) = &input_tensor.contents {
                                infer_request
                                    .append_input_data_generic2(
                                        &input_tensor.name,
                                        contents.uint_contents.as_ref(),
                                            data_type,
                                        TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU,
                                        0,
                                    )
                                    .map_err(|e| InferError::new(e.msg()))?;
                        }
                    }
                    TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT64 => {
                        if let Some(contents) = &input_tensor.contents {
                                infer_request
                                    .append_input_data_generic2(
                                        &input_tensor.name,
                                        contents.uint64_contents.as_ref(),
                                        data_type,
                                        TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU,
                                        0,
                                    )
                                    .map_err(|e| InferError::new(e.msg()))?;
                        }
                    }
                    TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT8 => {
                        if let Some(contents) = &input_tensor.contents {
                            let raw_ptr = contents.int_contents.as_ptr() as *mut i8;
                            for (i, &d) in contents.int_contents.iter().enumerate() {
                                unsafe {
                                    *raw_ptr.add(i) = d as i8;
                                }
                            }
                                infer_request
                                    .append_input_data_generic2(
                                        &input_tensor.name,

                                        contents.int_contents.as_ref(),
                                        data_type,
                                        TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU,
                                        0,
                                    )
                                    .map_err(|e| InferError::new(e.msg()))?;
                            }
                    }
                    TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT16 => {
                        if let Some(contents) = &input_tensor.contents {
                            let raw_ptr = contents.int_contents.as_ptr() as *mut i16;
                            for (i, &d) in contents.int_contents.iter().enumerate() {
                                unsafe {
                                    *raw_ptr.add(i) = d as i16;
                                }
                            }
                                infer_request
                                    .append_input_data_generic2(
                                        &input_tensor.name,
                                        contents.int_contents.as_ref(),
                                        data_type,
                                        TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU,
                                        0,
                                    )
                                    .map_err(|e| InferError::new(e.msg()))?;
                        }
                    }
                    TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT32 => {
                        if let Some(contents) = &input_tensor.contents {
                                infer_request
                                    .append_input_data_generic2(
                                        &input_tensor.name,
                                        contents.int_contents.as_ref(),
                                        data_type,
                                        TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU,
                                        0,
                                    )
                                    .map_err(|e| InferError::new(e.msg()))?;
                        }
                    }
                    TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT64 => {
                        if let Some(contents) = &input_tensor.contents {
                                infer_request
                                    .append_input_data_generic2(
                                        &input_tensor.name,
                                        contents.int64_contents.as_ref(),
                                        data_type,
                                        TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU,
                                        0,
                                    )
                                    .map_err(|e| InferError::new(e.msg()))?;
                        }
                    }
                    TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP32 => {
                        if let Some(contents) = &input_tensor.contents {
                                infer_request
                                    .append_input_data_generic2(
                                        &input_tensor.name,
                                        contents.fp32_contents.as_ref(),
                                        data_type,
                                        TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU,
                                        0,
                                    )
                                    .map_err(|e| InferError::new(e.msg()))?;
                        }
                    }
                    TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP64 => {
                        if let Some(contents) = &input_tensor.contents {
                                infer_request
                                    .append_input_data_generic2(
                                        &input_tensor.name,
                                        contents.fp64_contents.as_ref(),
                                        data_type,
                                        TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU,
                                        0,
                                    )
                                    .map_err(|e| InferError::new(e.msg()))?;
                        }
                    }
                    TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BYTES => {
                        if let Some(contents) = &input_tensor.contents {
                            let mut total_bytes = contents.bytes_contents.len() * 4;
                            for b in &contents.bytes_contents {
                                total_bytes += b.len();
                            }
                            let mut data = Vec::<u8>::with_capacity(total_bytes);
                            let mut k = 0;
                            for b in &contents.bytes_contents {
                                data.extend_from_slice(&(b.len() as u32).to_le_bytes());
                                k += 4;
                                data.extend_from_slice(b);
                                k += b.len();
                            }
                                infer_request
                                    .append_input_data_generic1(
                                        &input_tensor.name,
                                        data.as_ref(),
                                        k,
                                        TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU,
                                        0,
                                    )
                                    .map_err(|e| InferError::new(e.msg()))?;
                            bytes_slice.push(data);
                        }
                    }

                    _ => (),
                }
            }
        }
        for output_tensor in pb_request.outputs.iter() {
            infer_request
                .add_requested_output(&output_tensor.name)
                .map_err(|e| InferError::new(e.msg()))?;
        }

        let request_data_holder = Box::new(RequestDataHolder::new(pb_request, bytes_slice));

        infer_request
            .set_release_callback(request_data_holder)
            .map_err(|e| InferError::new(e.msg()))?;

        let (sender, receiver) = mpsc::channel(1);
        infer_request
            .set_response_callback(&self.response_allocator, sender)
            .map_err(|e| InferError::new(e.msg()))?;

        infer_request.set_response_receiver(receiver);

        Ok(())
    }

    #[allow(non_upper_case_globals)]
    pub fn create_pb_response(
        &self,
        response: InferResponse,
        raw_output: bool,
    ) -> Result<infer_proto::ModelInferResponse> {
        let (model, version) = match response.model() {
            Ok((m, v)) => (m, v),
            Err(e) => return Err(InferError::new(e.msg()).into()),
        };

        let request_id = match response.id() {
            Ok(id) => id,
            Err(e) => return Err(InferError::new(e.msg()).into()),
        };

        let output_count = match response.output_count() {
            Ok(c) => c,
            Err(e) => return Err(InferError::new(e.msg()).into()),
        };

        let mut pb_response = infer_proto::ModelInferResponse {
            model_name: model,
            model_version: version.to_string(),
            id: request_id,
            outputs: Vec::with_capacity(output_count as usize),
            parameters: HashMap::new(),
            raw_output_contents: Vec::with_capacity(output_count as usize),
        };

        for i in 0..output_count {
            let output_info = match response.output(i) {
                Ok(o) => o,
                Err(e) => return Err(InferError::new(e.msg()).into()),
            };

            let mut out_tensor = infer_proto::model_infer_response::InferOutputTensor {
                name: output_info.name().to_owned(),
                datatype: utils::data_type_to_string(output_info.datatype()).to_owned(),
                shape: output_info.shape().clone().to_vec(),
                ..Default::default()
            };

            if raw_output {
                unsafe {
                    let c_data = Vec::from_raw_parts(
                        output_info.base() as *mut u8,
                        output_info.byte_size(),
                        output_info.byte_size(),
                    );
                    let data = c_data.clone();
                    let _ = c_data.into_raw_parts();

                    pb_response.raw_output_contents.push(data);
                }
            } else {
                match output_info.datatype() {
                    TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BOOL => {
                        let c_data = unsafe {
                            Vec::from_raw_parts(
                                output_info.base() as *mut bool,
                                output_info.len(),
                                output_info.capacity(),
                            )
                        };
                        let data = c_data.clone();
                        let _ = c_data.into_raw_parts();
                        out_tensor.contents = Some(infer_proto::InferTensorContents {
                            bool_contents: data,
                            ..Default::default()
                        });
                    }
                    TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT8 => {
                        let c_data = unsafe {
                            Vec::from_raw_parts(
                                output_info.base() as *mut u8,
                                output_info.len(),
                                output_info.capacity(),
                            )
                        };
                        let data = c_data.iter().map(|&x| x as u32).collect::<Vec<u32>>();
                        let _ = c_data.into_raw_parts();
                        out_tensor.contents = Some(infer_proto::InferTensorContents {
                            uint_contents: data,
                            ..Default::default()
                        });
                    }
                    TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT16 => {
                        let c_data = unsafe {
                            Vec::from_raw_parts(
                                output_info.base() as *mut u16,
                                output_info.len(),
                                output_info.capacity(),
                            )
                        };
                        let data: Vec<u32> = c_data.iter().map(|&x| x as u32).collect();
                        let _ = c_data.into_raw_parts();
                        out_tensor.contents = Some(infer_proto::InferTensorContents {
                            uint_contents: data,
                            ..Default::default()
                        });
                    }
                    TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT32 => {
                        let c_data = unsafe {
                            Vec::from_raw_parts(
                                output_info.base() as *mut u32,
                                output_info.len(),
                                output_info.capacity(),
                            )
                        };
                        let data = c_data.clone();
                        let _ = c_data.into_raw_parts();
                        out_tensor.contents = Some(infer_proto::InferTensorContents {
                            uint_contents: data,
                            ..Default::default()
                        });
                    }
                    TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT64 => {
                        let c_data = unsafe {
                            Vec::from_raw_parts(
                                output_info.base() as *mut u64,
                                output_info.len(),
                                output_info.capacity(),
                            )
                        };
                        let data = c_data.clone();
                        let _ = c_data.into_raw_parts();
                        out_tensor.contents = Some(infer_proto::InferTensorContents {
                            uint64_contents: data,
                            ..Default::default()
                        });
                    }
                    TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT8 => {
                        let c_data = unsafe {
                            Vec::from_raw_parts(
                                output_info.base() as *mut i8,
                                output_info.len(),
                                output_info.capacity(),
                            )
                        };
                        let data: Vec<i32> = c_data.iter().map(|&x| x as i32).collect();
                        let _ = c_data.into_raw_parts();
                        out_tensor.contents = Some(infer_proto::InferTensorContents {
                            int_contents: data,
                            ..Default::default()
                        });
                    }
                    TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT16 => {
                        let c_data = unsafe {
                            Vec::from_raw_parts(
                                output_info.base() as *mut i16,
                                output_info.len(),
                                output_info.capacity(),
                            )
                        };
                        let data: Vec<i32> = c_data.iter().map(|&x| x as i32).collect();
                        let _ = c_data.into_raw_parts();
                        out_tensor.contents = Some(infer_proto::InferTensorContents {
                            int_contents: data,
                            ..Default::default()
                        });
                    }
                    TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT32 => {
                        let c_data = unsafe {
                            Vec::from_raw_parts(
                                output_info.base() as *mut i32,
                                output_info.len(),
                                output_info.capacity(),
                            )
                        };
                        let data = c_data.clone();
                        let _ = c_data.into_raw_parts();
                        out_tensor.contents = Some(infer_proto::InferTensorContents {
                            int_contents: data,
                            ..Default::default()
                        });
                    }
                    TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT64 => {
                        let c_data = unsafe {
                            Vec::from_raw_parts(
                                output_info.base() as *mut i64,
                                output_info.len(),
                                output_info.capacity(),
                            )
                        };
                        let data = c_data.clone();
                        let _ = c_data.into_raw_parts();
                        out_tensor.contents = Some(infer_proto::InferTensorContents {
                            int64_contents: data,
                            ..Default::default()
                        });
                    }
                    TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP32 => {
                        let c_data = unsafe {
                            Vec::from_raw_parts(
                                output_info.base() as *mut f32,
                                output_info.len(),
                                output_info.len(),
                            )
                        };
                        let data = c_data.clone();
                        let _ = c_data.into_raw_parts();
                        out_tensor.contents = Some(infer_proto::InferTensorContents {
                            fp32_contents: data,
                            ..Default::default()
                        });
                    }
                    TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP64 => {
                        let c_data = unsafe {
                            Vec::from_raw_parts(
                                output_info.base() as *mut f64,
                                output_info.len(),
                                output_info.capacity(),
                            )
                        };
                        let data = c_data.clone();
                        let _ = c_data.into_raw_parts();
                        out_tensor.contents = Some(infer_proto::InferTensorContents {
                            fp64_contents: data,
                            ..Default::default()
                        });
                    }
                    TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BYTES => {
                        let len = output_info.len();
                        let base = output_info.base() as *mut u8;
                        let byte_size = output_info.byte_size();
                        let c_data =unsafe{ Vec::from_raw_parts(base, byte_size, byte_size) };
                        let data = c_data.clone();
                        let _ = c_data.into_raw_parts();

                        let mut bytes_contents = Vec::<Vec<u8>>::with_capacity(len);
                        let mut k: usize = 0;
                            while k < len  {
                                let len = u32::from_le_bytes(data[k..k+4].try_into()?);
                                k += 4;
                                bytes_contents.push(data[k..k+len as usize].into());
                                k += len as usize;
                            }
                        out_tensor.contents = Some(infer_proto::InferTensorContents {
                            bytes_contents,
                            ..Default::default()
                        });
                    }

                    _ => (),
                }
            }

            pb_response.outputs.push(out_tensor);
        }
        Ok(pb_response)
    }

    pub async fn infer(
        &self,
        request: Box<infer_proto::ModelInferRequest>,
        timeout: u64,
    ) -> Result<infer_proto::ModelInferResponse> {
        let mut request = match self.parse_pb_request(request) {
            Ok(r) => r,
            Err(e) => return Err(e),
        };
        request.set_timeout_micro_seconds(timeout)?;
        if let Err(e) = self.infer_async(&mut request) {
            request.delete();
            return Err(InferError::new(e.msg()).into());
        };

        match request.receiver.unwrap().recv().await {
            Some(r) => {
                if let Some(e) = r.error() {
                    return Err(InferError::new(e.msg()).into());
                }
                // TODO: implement fine control on whether to use raw output
                self.create_pb_response(r, request.raw_output)
            }
            None => Err(InferError::new("receive error".to_string()).into()),
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        unsafe {
            TRITONSERVER_ServerStop(self._server);
            TRITONSERVER_ServerDelete(self._server);
        }
    }
}

impl Default for Server {
    fn default() -> Self {
        let options = ServerOptions::new().unwrap();
        // let mut //project_root = get_project_root().unwrap();
        //project_root.push("models");
        options.set_model_repository_path("../../../models").unwrap();
        let server = Server::new(options);
        match server {
            Ok(server) => {
                assert!(server.is_live().unwrap());
                assert!(server.is_ready().unwrap());
                server
            }
            Err(e) => panic!("{:?}", e.msg()),
        }
    }
}
unsafe impl Send for Server {}
unsafe impl Sync for Server {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let options = ServerOptions::new().unwrap();
        // let  mut //project_root = get_project_root().unwrap();
        //project_root.push("models");

        options.set_model_repository_path("../../../models").unwrap();
        let server = Server::new(options);
        match server {
            Ok(server) => {
                assert!(server.is_live().unwrap());
                assert!(server.is_ready().unwrap());
            }
            Err(e) => error!("{:?}", e.msg()),
        }
    }

    use ::infer_proto::infer_proto::InferTensorContents;
    use std::collections::HashMap;
    use std::fs;


    #[tokio::test]
    async fn test_addsub() {
        let options = ServerOptions::new().unwrap();
        // let mut //project_root = get_project_root().unwrap();
        //project_root.push("models");
        options.set_model_repository_path("../../../models").unwrap();
        let model_control_mode =
            tritonserver_modelcontrolmode_enum_TRITONSERVER_MODEL_CONTROL_EXPLICIT;
        options.set_model_control_mode(model_control_mode).unwrap();

        // create request

        let request = infer_proto::ModelInferRequest {
            parameters: HashMap::new(),
            model_name: "addsub".to_string(),
            model_version: "-1".to_owned(),
            id: "1".to_string(),
            inputs: vec![
                infer_proto::model_infer_request::InferInputTensor {
                    name: "INPUT0".to_string(),
                    datatype: "FP32".to_string(),
                    shape: vec![4],
                    contents: Some(InferTensorContents {
                        fp32_contents: vec![1.0, 2.0, 3.0, 4.0],
                        ..Default::default()
                    }),
                    parameters: HashMap::new(),
                },
                infer_proto::model_infer_request::InferInputTensor {
                    name: "INPUT1".to_string(),
                    datatype: "FP32".to_string(),
                    shape: vec![4],
                    contents: Some(InferTensorContents {
                        fp32_contents: vec![1.0, 2.0, 3.0, 4.0],
                        ..Default::default()
                    }),
                    parameters: HashMap::new(),
                },
            ],
            outputs: vec![
                infer_proto::model_infer_request::InferRequestedOutputTensor {
                    name: "OUTPUT0".to_string(),
                    parameters: HashMap::new(),
                },
                infer_proto::model_infer_request::InferRequestedOutputTensor {
                    name: "OUTPUT1".to_string(),
                    parameters: HashMap::new(),
                },
            ],
            raw_input_contents: vec![],
        };

        let server = Server::new(options);
        match server {
            Ok(server) => {
                assert!(server.is_live().unwrap());
                assert!(server.is_ready().unwrap());
                match server.load_model("addsub") {
                    Ok(_) => info!("model loading successful"),
                    Err(e) => error!("{:?}", e.msg()),
                }

                let resp = server.infer(Box::new(request), 5_000_000).await;
                info!("resp is {:?}", resp);
                match resp {
                    Ok(r) => match &r.outputs[0].contents {
                        Some(content) => {
                            assert_eq!(content.fp32_contents, vec![2.0, 4.0, 6.0, 8.0])
                        }
                        None => assert_eq!(true, false),
                    },
                    Err(e) => error!("Error: {:?}", e),
                }
            }
            Err(e) => error!("{:?}", e.msg()),
        }
    }

    #[tokio::test]
    async fn test_uint8() {
        let options = ServerOptions::new().unwrap();
        // let mut //project_root = get_project_root().unwrap();
        //project_root.push("modes");
        options.set_model_repository_path("../../../models").unwrap();
        let model_control_mode =
            tritonserver_modelcontrolmode_enum_TRITONSERVER_MODEL_CONTROL_EXPLICIT;
        options.set_model_control_mode(model_control_mode).unwrap();

        let server = Server::new(options);
        match server {
            Ok(server) => {
                assert!(server.is_live().unwrap());
                assert!(server.is_ready().unwrap());
                match server.load_model("uint8") {
                    Ok(_) => info!("model loading successful"),
                    Err(e) => error!("{:?}", e.msg()),
                }
                let request = infer_proto::ModelInferRequest {
                    parameters: HashMap::new(),
                    model_name: "uint8".to_string(),
                    model_version: "-1".to_owned(),
                    id: "1".to_string(),
                    inputs: vec![infer_proto::model_infer_request::InferInputTensor {
                        name: "INPUT".to_string(),
                        datatype: "UINT8".to_string(),
                        shape: vec![4],
                        contents: Some(InferTensorContents {
                            uint_contents: vec![1, 2, 3, 4],
                            ..Default::default()
                        }),
                        parameters: HashMap::new(),
                    }],
                    outputs: vec![
                        infer_proto::model_infer_request::InferRequestedOutputTensor {
                            name: "OUTPUT".to_string(),
                            parameters: HashMap::new(),
                        },
                    ],
                    raw_input_contents: vec![],
                };
                let resp = server.infer(Box::new(request), 5_000_000).await;
                info!("resp is {:?}", resp);
                match resp {
                    Ok(r) => match &r.outputs[0].contents {
                        Some(content) => assert_eq!(content.uint_contents, vec![1, 2, 3, 4]),
                        None => assert_eq!(true, false),
                    },
                    Err(e) => error!("Error: {:?}", e),
                }
            }
            Err(e) => error!("{:?}", e.msg()),
        }
    }

    #[tokio::test]
    async fn test_uint16() {
        let options = ServerOptions::new().unwrap();
        // let mut //project_root = get_project_root().unwrap();
        //project_root.push("models");
        options.set_model_repository_path("../../../models").unwrap();
        let model_control_mode =
            tritonserver_modelcontrolmode_enum_TRITONSERVER_MODEL_CONTROL_EXPLICIT;
        options.set_model_control_mode(model_control_mode).unwrap();

        let server = Server::new(options);
        match server {
            Ok(server) => {
                assert!(server.is_live().unwrap());
                assert!(server.is_ready().unwrap());
                match server.load_model("uint16") {
                    Ok(_) => info!("model loading successful"),
                    Err(e) => error!("{:?}", e.msg()),
                }

                let request = infer_proto::ModelInferRequest {
                    parameters: HashMap::new(),
                    model_name: "uint16".to_string(),
                    model_version: "-1".to_owned(),
                    id: "1".to_string(),
                    inputs: vec![infer_proto::model_infer_request::InferInputTensor {
                        name: "INPUT".to_string(),
                        datatype: "UINT16".to_string(),
                        shape: vec![4],
                        contents: Some(InferTensorContents {
                            uint_contents: vec![1, 2, 3, 4],
                            ..Default::default()
                        }),
                        parameters: HashMap::new(),
                    }],
                    outputs: vec![
                        infer_proto::model_infer_request::InferRequestedOutputTensor {
                            name: "OUTPUT".to_string(),
                            parameters: HashMap::new(),
                        },
                    ],
                    raw_input_contents: vec![],
                };
                let resp = server.infer(Box::new(request), 5_000_000).await;
                info!("resp is {:?}", resp);
                match resp {
                    Ok(r) => match &r.outputs[0].contents {
                        Some(content) => assert_eq!(content.uint_contents, vec![1, 2, 3, 4]),
                        None => assert_eq!(true, false),
                    },
                    Err(e) => error!("Error: {:?}", e),
                }
            }
            Err(e) => error!("{:?}", e.msg()),
        }
    }

    #[tokio::test]
    async fn test_uint32() {
        let options = ServerOptions::new().unwrap();
        // let mut //project_root = get_project_root().unwrap();
        //project_root.push("models");
        options.set_model_repository_path("../../../models").unwrap();
        let model_control_mode =
            tritonserver_modelcontrolmode_enum_TRITONSERVER_MODEL_CONTROL_EXPLICIT;
        options.set_model_control_mode(model_control_mode).unwrap();

        let server = Server::new(options);
        match server {
            Ok(server) => {
                assert!(server.is_live().unwrap());
                assert!(server.is_ready().unwrap());
                match server.load_model("uint32") {
                    Ok(_) => info!("model loading successful"),
                    Err(e) => error!("{:?}", e.msg()),
                }

                let request = infer_proto::ModelInferRequest {
                    parameters: HashMap::new(),
                    model_name: "uint32".to_string(),
                    model_version: "-1".to_owned(),
                    id: "1".to_string(),
                    inputs: vec![infer_proto::model_infer_request::InferInputTensor {
                        name: "INPUT".to_string(),
                        datatype: "UINT32".to_string(),
                        shape: vec![4],
                        contents: Some(InferTensorContents {
                            uint_contents: vec![1, 2, 3, 4],
                            ..Default::default()
                        }),
                        parameters: HashMap::new(),
                    }],
                    outputs: vec![
                        infer_proto::model_infer_request::InferRequestedOutputTensor {
                            name: "OUTPUT".to_string(),
                            parameters: HashMap::new(),
                        },
                    ],
                    raw_input_contents: vec![],
                };
                let resp = server.infer(Box::new(request), 5_000_000).await;
                info!("resp is {:?}", resp);
                match resp {
                    Ok(r) => match &r.outputs[0].contents {
                        Some(content) => assert_eq!(content.uint_contents, vec![1, 2, 3, 4]),
                        None => assert_eq!(true, false),
                    },
                    Err(e) => error!("Error: {:?}", e),
                }
            }
            Err(e) => error!("{:?}", e.msg()),
        }
    }

    #[tokio::test]
    async fn test_fp16() {
        let options = ServerOptions::new().unwrap();
        // let mut //project_root = get_project_root().unwrap();
        //project_root.push("models");
        options.set_model_repository_path("../../../models").unwrap();
        let model_control_mode =
            tritonserver_modelcontrolmode_enum_TRITONSERVER_MODEL_CONTROL_EXPLICIT;
        options.set_model_control_mode(model_control_mode).unwrap();

        let server = Server::new(options);
        match server {
            Ok(server) => {
                assert!(server.is_live().unwrap());
                assert!(server.is_ready().unwrap());
                match server.load_model("fp16") {
                    Ok(_) => info!("model loading successful"),
                    Err(e) => error!("{:?}", e.msg()),
                }

                let request = infer_proto::ModelInferRequest {
                    parameters: HashMap::new(),
                    model_name: "pf16".to_string(),
                    model_version: "-1".to_owned(),
                    id: "1".to_string(),
                    inputs: vec![infer_proto::model_infer_request::InferInputTensor {
                        name: "INPUT".to_string(),
                        datatype: "FP16".to_string(),
                        shape: vec![4],
                        contents: Some(InferTensorContents {
                            fp32_contents: vec![1., 2., 3., 4.],
                            ..Default::default()
                        }),
                        parameters: HashMap::new(),
                    }],
                    outputs: vec![
                        infer_proto::model_infer_request::InferRequestedOutputTensor {
                            name: "OUTPUT".to_string(),
                            parameters: HashMap::new(),
                        },
                    ],
                    raw_input_contents: vec![],
                };
                let resp = server.infer(Box::new(request), 5_000_000).await;
                info!("resp is {:?}", resp);
                match resp {
                    Ok(r) => match &r.outputs[0].contents {
                        Some(content) => assert_eq!(content.fp32_contents, vec![1., 2., 3., 4.]),
                        None => assert_eq!(true, false),
                    },
                    Err(e) => error!("Error: {:?}", e),
                }
            }
            Err(e) => error!("{:?}", e.msg()),
        }
    }

    #[tokio::test]
    async fn test_bytes() {
        let options = ServerOptions::new().unwrap();
        options.set_model_repository_path("../../../models").unwrap();
        let model_control_mode =
            tritonserver_modelcontrolmode_enum_TRITONSERVER_MODEL_CONTROL_EXPLICIT;
        options.set_model_control_mode(model_control_mode).unwrap();

        let server = Server::new(options);
        match server {
            Ok(server) => {
                assert!(server.is_live().unwrap());
                assert!(server.is_ready().unwrap());
                match server.load_model("bytes") {
                    Ok(_) => info!("model loading successful"),
                    Err(e) => error!("{:?}", e.msg()),
                }

                let request = infer_proto::ModelInferRequest {
                    parameters: HashMap::new(),
                    model_name: "bytes".to_string(),
                    model_version: "-1".to_owned(),
                    id: "1".to_string(),
                    inputs: vec![infer_proto::model_infer_request::InferInputTensor {
                        name: "INPUT".to_string(),
                        datatype: "BYTES".to_string(),
                        shape: vec![4],
                        contents: Some(InferTensorContents {
                            bytes_contents: vec![
                                "stringa".as_bytes().to_vec(),
                                "stringb".as_bytes().to_vec(),
                                "stringc".as_bytes().to_vec(),
                                "stringd".as_bytes().to_vec(),
                            ],
                            ..Default::default()
                        }),
                        parameters: HashMap::new(),
                    }],
                    outputs: vec![
                        infer_proto::model_infer_request::InferRequestedOutputTensor {
                            name: "OUTPUT".to_string(),
                            parameters: HashMap::new(),
                        },
                    ],
                    raw_input_contents: vec![],
                };
                let resp = server.infer(Box::new(request), 5_000_000).await;
                info!("resp is {:?}", resp);
                match resp {
                    Ok(r) => match &r.outputs[0].contents {
                        Some(content) => assert_eq!(
                            content.bytes_contents,
                            vec![
                                b"stringa".to_vec(),
                                b"stringb".to_vec(),
                                b"stringc".to_vec(),
                                b"stringd".to_vec()
                            ]
                        ),
                        None => assert_eq!(true, false),
                    },
                    Err(e) => error!("Error: {:?}", e),
                }
            }
            Err(e) => error!("{:?}", e.msg()),
        }
    }
}
