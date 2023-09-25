use std::ffi::CString;
use std::{alloc::Layout, ffi::c_void, ptr, u8};

use crate::sys::*;
use log::error;

use super::error::TritonError;

pub struct ResponseAllocator {
    _allocator: *mut TRITONSERVER_ResponseAllocator,
}

impl ResponseAllocator {
    pub fn new() -> Result<Self, TritonError> {
        #[allow(unused_variables)]
        #[no_mangle]
        unsafe extern "C" fn ResponseAlloc(
            allocator: *mut TRITONSERVER_ResponseAllocator,
            tensor_name: *const i8,
            byte_size: usize,
            memory_type: TRITONSERVER_MemoryType,
            memory_type_id: i64,
            userp: *mut c_void,
            buffer: *mut *mut c_void,
            buffer_userp: *mut *mut c_void,
            actual_memory_type: *mut TRITONSERVER_MemoryType,
            actual_memory_type_id: *mut i64,
        ) -> *mut TRITONSERVER_Error {
            *actual_memory_type = TRITONSERVER_memorytype_enum_TRITONSERVER_MEMORY_CPU;
            *actual_memory_type_id = 0;
            *buffer = ptr::null_mut();
            *buffer_userp = ptr::null_mut();
            let layout_msg = CString::new("allocator failed to create layout").unwrap();
            let failed_msg = CString::new("allocator failed").unwrap();
            if byte_size != 0 {
                let layout = match Layout::from_size_align(byte_size, 8) {
                    Ok(l) => l,
                    Err(e) => {
                        error!("error: {:?}", e);
                        return TRITONSERVER_ErrorNew(
                            TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_INTERNAL,
                            layout_msg.as_ptr(),
                        );
                    }
                };
                *buffer = std::alloc::alloc(layout) as *mut c_void;
                if (*buffer).is_null() {
                    return TRITONSERVER_ErrorNew(
                        TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_INTERNAL,
                        failed_msg.as_ptr(),
                    );
                }
            }
            ptr::null_mut()
        }
        #[allow(unused_variables)]
        #[no_mangle]
        unsafe extern "C" fn ResponseRelease(
            allocator: *mut TRITONSERVER_ResponseAllocator,
            buffer: *mut c_void,
            buffer_userp: *mut c_void,
            byte_size: usize,
            memory_type: TRITONSERVER_MemoryType,
            memory_type_id: i64,
        ) -> *mut TRITONSERVER_Error {
            let layout_msg = CString::new("allocator failed to create layout").unwrap();
            let layout = match Layout::from_size_align(byte_size, 8) {
                Ok(l) => l,
                Err(e) => {
                    error!("error: {:?}", e);
                    return TRITONSERVER_ErrorNew(
                        TRITONSERVER_errorcode_enum_TRITONSERVER_ERROR_INTERNAL,
                        layout_msg.as_ptr(),
                    );
                }
            };
            std::alloc::dealloc(buffer as *mut u8, layout);
            // std::alloc::dealloc(buffer_userp, layout);
            ptr::null_mut()
        }
        ResponseAllocator::from_fn(Some(ResponseAlloc), Some(ResponseRelease), None)
    }

    pub fn from_fn(
        alloc_fn: TRITONSERVER_ResponseAllocatorAllocFn_t,
        release_fn: TRITONSERVER_ResponseAllocatorReleaseFn_t,
        start_fn: TRITONSERVER_ResponseAllocatorStartFn_t,
    ) -> Result<Self, TritonError> {
        let mut allocator = ptr::null_mut() as *mut TRITONSERVER_ResponseAllocator;
        let err = unsafe {
            TRITONSERVER_ResponseAllocatorNew(
                &mut allocator as *mut *mut TRITONSERVER_ResponseAllocator,
                alloc_fn,
                release_fn,
                start_fn,
            )
        };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        Ok(ResponseAllocator {
            _allocator: allocator,
        })
    }

    pub fn set_buffer_attributes_function(
        &self,
        buffer_attributes_fn: TRITONSERVER_ResponseAllocatorBufferAttributesFn_t,
    ) -> Result<(), TritonError> {
        let err = unsafe {
            TRITONSERVER_ResponseAllocatorSetBufferAttributesFunction(
                self._allocator,
                buffer_attributes_fn,
            )
        };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        Ok(())
    }

    pub fn set_query_function(
        &self,
        query_fn: TRITONSERVER_ResponseAllocatorQueryFn_t,
    ) -> Result<(), TritonError> {
        let err =
            unsafe { TRITONSERVER_ResponseAllocatorSetQueryFunction(self._allocator, query_fn) };
        if !err.is_null() {
            return Err(TritonError::from_ptr(err));
        }
        Ok(())
    }

    pub fn raw_pointer(&self) -> *mut TRITONSERVER_ResponseAllocator {
        self._allocator
    }
}

impl Drop for ResponseAllocator {
    fn drop(&mut self) {
        unsafe {
            TRITONSERVER_ResponseAllocatorDelete(self._allocator);
        }
    }
}

impl Default for ResponseAllocator {
    fn default() -> Self {
        Self::new().unwrap()
    }
}
unsafe impl Sync for ResponseAllocator {}
unsafe impl Send for ResponseAllocator {}
