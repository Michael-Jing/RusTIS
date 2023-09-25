use crate::sys::*;
pub struct TritonMetrics {
    _metrics: *mut TRITONSERVER_Metrics,
}

impl Drop for TritonMetrics {
    fn drop(&mut self) {
        unsafe {
            TRITONSERVER_MetricsDelete(self._metrics);
        }
    }
}
