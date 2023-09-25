use std::{ffi::CString, fmt::Error, ptr};

use crate::sys::*;

use super::error::TritonError;

pub struct ServerOptions {
    pub _options: *mut TRITONSERVER_ServerOptions,
}

impl ServerOptions {
    pub fn new() -> Result<Self, Error> {
        let mut options = ptr::null_mut() as *mut TRITONSERVER_ServerOptions;
        let err = unsafe {
            TRITONSERVER_ServerOptionsNew(&mut options as *mut *mut TRITONSERVER_ServerOptions)
        };
        // TODO: error handling
        if !err.is_null() {
            return Err(Error);
        }
        Ok(ServerOptions { _options: options })
    }

    pub fn set_model_repository_path(&self, path: &str) -> Result<(), Error> {
        let c_path = CString::new(path).unwrap();
        let err = unsafe {
            TRITONSERVER_ServerOptionsSetModelRepositoryPath(self._options, c_path.as_ptr())
        };
        if !err.is_null() {
            return Err(Error);
        }
        Ok(())
    }

    pub fn set_backend_config(&self, backend: &str, key: &str, value: &str) -> Result<(), Error> {
        let c_backend = CString::new(backend).unwrap();
        let c_key = CString::new(key).unwrap();
        let c_value = CString::new(value).unwrap();
        let err = unsafe {
            TRITONSERVER_ServerOptionsSetBackendConfig(
                self._options,
                c_backend.as_ptr(),
                c_key.as_ptr(),
                c_value.as_ptr(),
            )
        };
        if !err.is_null() {
            return Err(Error);
        }
        Ok(())
    }

    pub fn set_model_control_mode(
        &self,
        mode: TRITONSERVER_ModelControlMode,
    ) -> Result<(), TritonError> {
        let err = unsafe { TRITONSERVER_ServerOptionsSetModelControlMode(self._options, mode) };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        Ok(())
    }
}

impl Drop for ServerOptions {
    fn drop(&mut self) {
        unsafe {
            TRITONSERVER_ServerOptionsDelete(self._options);
        }
    }
}
