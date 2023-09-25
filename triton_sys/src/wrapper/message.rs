use crate::sys::*;
use std::ffi::c_char;
use std::ptr;

use super::error::TritonError;

pub struct TritonMessage {
    _message: *mut TRITONSERVER_Message,
}

impl TritonMessage {
    pub fn from_ptr(ptr: *mut TRITONSERVER_Message) -> Self {
        TritonMessage { _message: ptr }
    }
    pub fn from_serialized_json(json: &str) -> Result<Self, TritonError> {
        let mut message = ptr::null_mut() as *mut TRITONSERVER_Message;
        let err = unsafe {
            TRITONSERVER_MessageNewFromSerializedJson(
                &mut message as *mut *mut TRITONSERVER_Message,
                json.as_ptr() as *const i8,
                json.len(),
            )
        };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        Ok(TritonMessage { _message: message })
    }

    pub fn to_serialized_json(&self) -> Result<String, TritonError> {
        let mut buffer = ptr::null_mut() as *const c_char;
        let mut buffer_size = 0;
        let err = unsafe {
            TRITONSERVER_MessageSerializeToJson(
                self._message,
                &mut buffer as *mut *const c_char,
                &mut buffer_size as *mut usize,
            )
        };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        let buffer_slice = unsafe { std::slice::from_raw_parts(buffer as *const u8, buffer_size) };
        let json = String::from_utf8_lossy(buffer_slice).into_owned();
        Ok(json)
    }
}

impl Drop for TritonMessage {
    fn drop(&mut self) {
        unsafe {
            TRITONSERVER_MessageDelete(self._message);
        }
    }
}
