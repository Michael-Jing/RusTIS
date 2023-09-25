use super::{
    error::TritonError, inference_response::InferResponse, response_allocator::ResponseAllocator,
    server::Server, utils,
};
use crate::sys::*;
use log::error;
use std::{ffi::{c_char, CString}};

use futures::executor::block_on;
use infer_proto::infer_proto;
use std::{ffi::c_void, ptr};
use tokio::{sync::mpsc::Receiver, sync::mpsc::Sender};

pub struct RequestDataHolder {
    _request: Box<infer_proto::ModelInferRequest>,
    _bytes: Vec<Vec<u8>>,
}

impl RequestDataHolder {
    pub fn new(request: Box<infer_proto::ModelInferRequest>, bytes: Vec<Vec<u8>>) -> Self {
        RequestDataHolder { _request: request, _bytes: bytes }
    }
}

#[derive(Debug)]
pub struct InferRequest {
    _request: *mut TRITONSERVER_InferenceRequest,
    pub receiver: Option<Receiver<InferResponse>>,
    pub raw_output: bool,
}

impl InferRequest {
    pub fn new(
        server: &Server,
        model_name: &str,
        model_version: i64,
        raw_output: bool,
    ) -> Result<Self, TritonError> {
        let mut request = ptr::null_mut() as *mut TRITONSERVER_InferenceRequest;
        let c_model_name = std::ffi::CString::new(model_name).unwrap();
        let err = unsafe {
            TRITONSERVER_InferenceRequestNew(
                &mut request as *mut *mut TRITONSERVER_InferenceRequest,
                server._server,
                c_model_name.as_ptr(),
                model_version,
            )
        };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        Ok(InferRequest {
            _request: request,
            receiver: None,
            raw_output,
        })
    }

    pub fn id(&self) -> Result<String, TritonError> {
        let mut id = ptr::null_mut() as *const c_char;
        let err = unsafe {
            TRITONSERVER_InferenceRequestId(self._request, &mut id as *mut *const c_char)
        };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        let id_slice = unsafe { std::ffi::CStr::from_ptr(id).to_bytes() };
        let id = String::from_utf8_lossy(id_slice).into_owned();
        Ok(id)
    }

    pub fn set_id(&mut self, id: &str) -> Result<(), TritonError> {
        let c_id = std::ffi::CString::new(id).unwrap();
        let err = unsafe { TRITONSERVER_InferenceRequestSetId(self._request, c_id.as_ptr()) };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        Ok(())
    }

    pub fn flags(&self) -> Result<u32, TritonError> {
        let mut flags = 0;
        let err =
            unsafe { TRITONSERVER_InferenceRequestFlags(self._request, &mut flags as *mut u32) };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        Ok(flags)
    }

    pub fn set_flags(&mut self, flags: u32) -> Result<(), TritonError> {
        let err = unsafe { TRITONSERVER_InferenceRequestSetFlags(self._request, flags) };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        Ok(())
    }

    pub fn correlation_id(&self) -> Result<u64, TritonError> {
        let mut correlation_id = 0;
        let err = unsafe {
            TRITONSERVER_InferenceRequestCorrelationId(
                self._request,
                &mut correlation_id as *mut u64,
            )
        };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        Ok(correlation_id)
    }

    pub fn correlation_id_string(&self) -> Result<String, TritonError> {
        let mut correlation_id = ptr::null_mut() as *const c_char;
        let err = unsafe {
            TRITONSERVER_InferenceRequestCorrelationIdString(
                self._request,
                &mut correlation_id as *mut *const c_char,
            )
        };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        let correlation_id_slice = unsafe { std::ffi::CStr::from_ptr(correlation_id).to_bytes() };
        let correlation_id = String::from_utf8_lossy(correlation_id_slice).into_owned();
        Ok(correlation_id)
    }

    pub fn priority(&self) -> Result<u32, TritonError> {
        let mut priority = 0u32;
        let err = unsafe {
            TRITONSERVER_InferenceRequestPriority(self._request, &mut priority as *mut u32)
        };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        Ok(priority)
    }

    pub fn set_priority(&mut self, priority: u32) -> Result<(), TritonError> {
        let err = unsafe { TRITONSERVER_InferenceRequestSetPriority(self._request, priority) };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        Ok(())
    }

    pub fn timeout_micro_seconds(&self) -> Result<u64, TritonError> {
        let mut timeout = 0u64;
        let err = unsafe {
            TRITONSERVER_InferenceRequestTimeoutMicroseconds(
                self._request,
                &mut timeout as *mut u64,
            )
        };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        Ok(timeout)
    }

    pub fn set_timeout_micro_seconds(&mut self, timeout: u64) -> Result<(), TritonError> {
        let err =
            unsafe { TRITONSERVER_InferenceRequestSetTimeoutMicroseconds(self._request, timeout) };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        Ok(())
    }

    pub fn add_input(
        &mut self,
        name: &str,
        datatype: TRITONSERVER_DataType,
        shape: &[i64],
    ) -> Result<(), TritonError> {
        let c_name = std::ffi::CString::new(name).unwrap();
        let dim_count = shape.len() as u64;
        let shape = shape.as_ptr();

        let err = unsafe {
            TRITONSERVER_InferenceRequestAddInput(
                self._request,
                c_name.as_ptr(),
                datatype,
                shape,
                dim_count,
            )
        };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        Ok(())
    }

    pub fn add_requested_output(&self, name: &str) -> Result<(), TritonError> {
        let c_name = CString::new(name).unwrap();
        let err = unsafe {
            TRITONSERVER_InferenceRequestAddRequestedOutput(self._request, c_name.as_ptr())
        };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        Ok(())
    }

    pub fn add_raw_input(&mut self, name: &str) -> Result<(), TritonError> {
        let c_name = std::ffi::CString::new(name).unwrap();
        let err =
            unsafe { TRITONSERVER_InferenceRequestAddRawInput(self._request, c_name.as_ptr()) };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        Ok(())
    }

    pub fn remove_input(&mut self, name: &str) -> Result<(), TritonError> {
        let c_name = std::ffi::CString::new(name).unwrap();
        let err =
            unsafe { TRITONSERVER_InferenceRequestRemoveInput(self._request, c_name.as_ptr()) };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        Ok(())
    }

    pub fn remove_all_inputs(&mut self) -> Result<(), TritonError> {
        let err = unsafe { TRITONSERVER_InferenceRequestRemoveAllInputs(self._request) };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        Ok(())
    }

    unsafe fn append_input_data(
        &mut self,
        name: &str,
        base: *const c_void,
        byte_size: usize,
        memory_type: TRITONSERVER_MemoryType,
        memory_type_id: i64,
    ) -> Result<(), TritonError> {
        let c_name = std::ffi::CString::new(name).unwrap();
        let err = unsafe {
            TRITONSERVER_InferenceRequestAppendInputData(
                self._request,
                c_name.as_ptr(),
                base,
                byte_size,
                memory_type,
                memory_type_id,
            )
        };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        Ok(())
    }

    pub fn append_input_data_generic1<T>(
        &mut self,
        name: &str,
        data: &[T] ,
        byte_size: usize,
        memory_type: TRITONSERVER_MemoryType,
        memory_type_id: i64,
    ) -> Result<(), TritonError> {
        let c_name = std::ffi::CString::new(name).unwrap();

        let err = unsafe {
            let base = data.as_ptr() as *const c_void;
            TRITONSERVER_InferenceRequestAppendInputData(
                self._request,
                c_name.as_ptr(),
                base,
                byte_size,
                memory_type,
                memory_type_id,
            )
        };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        Ok(())
    }


    pub fn append_input_data_generic2<T>(
        &mut self,
        name: &str,
        data: &[T] ,
        dtype: TRITONSERVER_DataType,
        memory_type: TRITONSERVER_MemoryType,
        memory_type_id: i64,
    ) -> Result<(), TritonError> {
        let c_name = std::ffi::CString::new(name).unwrap();
        let byte_size = data.len() * utils::data_type_size(dtype);
        let err = unsafe {
            let base = data.as_ptr() as *const c_void;
            TRITONSERVER_InferenceRequestAppendInputData(
                self._request,
                c_name.as_ptr(),
                base,
                byte_size,
                memory_type,
                memory_type_id,
            )
        };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        Ok(())
    }


    pub fn raw_pointer(&self) -> *mut TRITONSERVER_InferenceRequest {
        self._request
    }

    pub fn set_release_callback(
        &self,
        request_data_holder: Box<RequestDataHolder>,
    ) -> Result<(), TritonError> {
        #[allow(unused_variables)]
        #[no_mangle]
        extern "C" fn release_callback(
            request: *mut TRITONSERVER_InferenceRequest,
            flags: u32,
            userp: *mut c_void,
        ) {
            if flags & tritonserver_requestreleaseflag_enum_TRITONSERVER_REQUEST_RELEASE_ALL != 0 {
                let pb_request = unsafe { Box::from_raw(userp as *mut RequestDataHolder) };
                let err = unsafe { TRITONSERVER_InferenceRequestDelete(request) };
                if !err.is_null() {
                    error!(
                        "error deleting request: {:?}",
                        TritonError::from_ptr(err).msg()
                    );
                }
            }
        }
        let ptr = Box::into_raw(request_data_holder) as *mut c_void;
        let err = unsafe {
            TRITONSERVER_InferenceRequestSetReleaseCallback(
                self._request,
                Some(release_callback),
                ptr as *mut c_void,
            )
        };
        if !err.is_null() {
            let _request_data_holder = unsafe { Box::from_raw(ptr as *mut RequestDataHolder) };
            return Err(TritonError::from_ptr(err));
        }
        Ok(())
    }

    // design 1: a response object can be created, then a response raw pointer can be passedto set_response_callback,
    // then later the pointer can be set to the actual response object from the callback, then caller need to repeatedly
    // check if the pointer has been set
    // design 2: pass in a future and then caller can wait for that future?
    // design 3: use rust channels,
    pub fn set_response_callback(
        &self,
        response_allocator: &ResponseAllocator,
        response_userp: Sender<InferResponse>,
    ) -> Result<(), TritonError> {
        #[no_mangle]
        extern "C" fn response_callback(
            response: *mut TRITONSERVER_InferenceResponse,
            flags: u32,
            userp: *mut c_void,
        ) {
            // convert userp to a channel
            let tx = unsafe { Box::from_raw(userp as *mut Sender<InferResponse>) };
            if !response.is_null() {
                let infer_response = InferResponse::from_ptr(response);

                match block_on((tx).send(infer_response)) {
                    Ok(_) => {}
                    Err(e) => error!("error sending response: {:?}", e),
                };
            }
            if flags & tritonserver_responsecompleteflag_enum_TRITONSERVER_RESPONSE_COMPLETE_FINAL
                != 0
            {}
        }
        let response_userp: *mut Sender<InferResponse> = Box::into_raw(Box::new(response_userp));
        let response_userp: *mut c_void = response_userp as *mut c_void;
        let err = unsafe {
            TRITONSERVER_InferenceRequestSetResponseCallback(
                self._request,
                response_allocator.raw_pointer(),
                ptr::null_mut(),
                Some(response_callback),
                response_userp,
            )
        };

        if !err.is_null() {
            let _sender = unsafe { Box::from_raw(response_userp as *mut Sender<InferResponse>) };
            return Err(TritonError::from_ptr(err));
        }
        Ok(())
    }

    pub fn set_response_receiver(&mut self, receiver: Receiver<InferResponse>) {
        self.receiver = Some(receiver);
    }

    pub fn delete(&mut self) {
        let err = unsafe { TRITONSERVER_InferenceRequestDelete(self._request) };
        if !err.is_null() {
            error!(
                "error deleting request: {:?}",
                TritonError::from_ptr(err).msg()
            );
        }
    }
}
unsafe impl Send for InferRequest {}

/*
impl Drop for InferRequest {
    fn drop(&mut self) {
        unsafe {
            TRITONSERVER_InferenceRequestDelete(self._request);
        }
    }
}
*/
