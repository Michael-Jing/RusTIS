use crate::sys::*;
use core::fmt;
use std::error::Error;

#[derive(Debug)]
pub struct TritonError {
    pub _err: *mut TRITONSERVER_Error,
}

impl TritonError {
    pub fn from_ptr(err: *mut TRITONSERVER_Error) -> Self {
        TritonError { _err: err }
    }
    pub fn new(code: TRITONSERVER_Error_Code, msg: String) -> Self {
        let err_pointer = unsafe { TRITONSERVER_ErrorNew(code, msg.as_ptr() as *const i8) };
        TritonError { _err: err_pointer }
    }

    pub fn msg(&self) -> String {
        let msg = unsafe { TRITONSERVER_ErrorMessage(self._err) };
        return unsafe { std::ffi::CStr::from_ptr(msg).to_string_lossy().into_owned() };
    }

    pub fn code(&self) -> TRITONSERVER_Error_Code {
        unsafe { TRITONSERVER_ErrorCode(self._err) }
    }

    pub fn code_string(&self) -> String {
        let code = unsafe { TRITONSERVER_ErrorCodeString(self._err) };
        return unsafe {
            std::ffi::CStr::from_ptr(code)
                .to_string_lossy()
                .into_owned()
        };
    }
}

impl fmt::Display for TritonError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.msg())
    }
}

impl Error for TritonError {}

impl Drop for TritonError {
    fn drop(&mut self) {
        unsafe {
            TRITONSERVER_ErrorDelete(self._err);
        }
    }
}

unsafe impl Send for TritonError {}
unsafe impl Sync for TritonError {}

#[derive(Debug)]
pub struct InferError {
    msg: String,
}
impl InferError {
    pub fn new(msg: String) -> Self {
        InferError { msg }
    }
}
impl fmt::Display for InferError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}

impl Error for InferError {}
