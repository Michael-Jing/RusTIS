use super::error::TritonError;
use super::utils;
use crate::sys::*;
use std::ffi::c_char;
use std::ffi::CStr;
use std::{ffi::c_void, ptr};

#[derive(Debug)]
pub struct InferResponse {
    _response: *mut TRITONSERVER_InferenceResponse,
}

impl InferResponse {
    pub fn from_ptr(response: *mut TRITONSERVER_InferenceResponse) -> Self {
        InferResponse {
            _response: response,
        }
    }

    pub fn error(&self) -> Option<TritonError> {
        let err = unsafe { TRITONSERVER_InferenceResponseError(self._response) };
        if err.is_null() {
            return None;
        }
        Some(TritonError::from_ptr(err))
    }

    pub fn model(&self) -> Result<(String, i64), TritonError> {
        let mut model_name = ptr::null_mut() as *const c_char;
        let mut model_version = 0;
        let err = unsafe {
            TRITONSERVER_InferenceResponseModel(self._response, &mut model_name, &mut model_version)
        };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        let name = unsafe { CStr::from_ptr(model_name) }
            .to_string_lossy()
            .into_owned();
        Ok((name, model_version))
    }

    pub fn id(&self) -> Result<String, TritonError> {
        let mut id = ptr::null_mut() as *const c_char;
        let err = unsafe { TRITONSERVER_InferenceResponseId(self._response, &mut id) };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        return Ok(unsafe { CStr::from_ptr(id) }.to_string_lossy().into_owned());
    }

    pub fn output_count(&self) -> Result<u32, TritonError> {
        let mut count = 0;
        let err = unsafe { TRITONSERVER_InferenceResponseOutputCount(self._response, &mut count) };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        Ok(count)
    }

    pub fn output(&self, index: u32) -> Result<InferResponseOutput, TritonError> {
        let mut name = ptr::null_mut() as *const c_char;
        let mut datatype = TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INVALID;
        let mut shape = ptr::null_mut() as *const i64;
        let mut dim_count = 0;
        let mut base = ptr::null_mut() as *const c_void;
        let mut byte_size = 0;
        let mut memory_type = TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU;
        let mut memory_type_id = 0;
        let mut userp = ptr::null_mut() as *mut c_void;

        let err = unsafe {
            TRITONSERVER_InferenceResponseOutput(
                self._response,
                index,
                &mut name,
                &mut datatype,
                &mut shape,
                &mut dim_count,
                &mut base,
                &mut byte_size,
                &mut memory_type,
                &mut memory_type_id,
                &mut userp,
            )
        };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }

        unsafe {
            InferResponseOutput::new(
                name,
                datatype,
                shape,
                dim_count,
                base,
                byte_size,
                memory_type,
                memory_type_id,
                userp,
            )
        }
    }
}
//  uncomment causes a double free error  TODO: fix
unsafe impl Send for InferResponse {}
impl Drop for InferResponse {
    fn drop(&mut self) {
        unsafe {
            TRITONSERVER_InferenceResponseDelete(self._response);
        }
    }
}
//

pub struct InferResponseOutput {
    name: String,
    datatype: TRITONSERVER_DataType,
    shape: Vec<i64>,
    memory_type: TRITONSERVER_MemoryType,
    memory_type_id: i64,
    user_pointer: *mut c_void,
    base: *const c_void,
    byte_size: usize,
}

impl InferResponseOutput {
    unsafe fn new(
        name: *const c_char,
        datatype: TRITONSERVER_DataType,
        shape: *const i64,
        dim_count: u64,
        base: *const c_void,
        byte_size: usize,
        memory_type: TRITONSERVER_MemoryType,
        memory_type_id: i64,
        userp: *mut c_void,
    ) -> Result<Self, TritonError> {
        let name = unsafe { std::ffi::CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned();
        let shape = unsafe { std::slice::from_raw_parts(shape, dim_count as usize) }.to_vec();
        let output = InferResponseOutput {
            name,
            datatype,
            shape,
            memory_type,
            memory_type_id,
            user_pointer: userp,
            base,
            byte_size,
        };
        Ok(output)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn datatype(&self) -> TRITONSERVER_DataType {
        self.datatype
    }

    pub fn shape(&self) -> &Vec<i64> {
        &self.shape
    }

    pub fn memory_type(&self) -> TRITONSERVER_MemoryType {
        self.memory_type
    }

    pub fn memory_type_id(&self) -> i64 {
        self.memory_type_id
    }

    pub fn user_pointer(&self) -> *mut c_void {
        self.user_pointer
    }

    pub fn base(&self) -> *const c_void {
        self.base
    }

    pub fn byte_size(&self) -> usize {
        self.byte_size
    }

    pub fn len(&self) -> usize {
        return self.shape.iter().product::<i64>() as usize;
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn capacity(&self) -> usize {
        self.byte_size / utils::data_type_size(self.datatype)
    }
}
