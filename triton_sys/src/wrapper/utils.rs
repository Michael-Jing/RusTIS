use crate::sys::*;

pub fn string_to_data_type(dtype: &str) -> TRITONSERVER_DataType {
    match dtype {
        "BOOL" => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BOOL,
        "UINT8" => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT8,
        "UINT16" => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT16,
        "UINT32" => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT32,
        "UINT64" => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT64,
        "INT8" => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT8,
        "INT16" => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT16,
        "INT32" => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT32,
        "INT64" => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT64,
        "FP16" => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP16,
        "FP32" => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP32,
        "FP64" => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP64,
        "BYTES" => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BYTES,
        "BF16" => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BF16,
        _ => TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INVALID,
    }
}

#[allow(non_upper_case_globals)]
pub fn data_type_to_string(dtype: TRITONSERVER_DataType) -> &'static str {
    match dtype {
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BOOL => "BOOL",
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT8 => "UINT8",
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT16 => "UINT16",
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT32 => "UINT32",
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT64 => "UINT64",
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT8 => "INT8",
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT16 => "INT16",
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT32 => "INT32",
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT64 => "INT64",
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP16 => "FP16",
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP32 => "FP32",
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP64 => "FP64",
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BYTES => "BYTES",
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BF16 => "BF16",
        _ => "INVALID",
    }
}

#[allow(non_upper_case_globals)]
pub fn data_type_size(dtype: TRITONSERVER_datatype_enum) -> usize {
    match dtype {
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BOOL => 1,
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT8 => 1,
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT16 => 2,
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT32 => 4,
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_UINT64 => 8,
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT8 => 1,
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT16 => 2,
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT32 => 4,
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_INT64 => 8,
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP16 => 2,
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP32 => 4,
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_FP64 => 8,
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BYTES => 0,
        TRITONSERVER_datatype_enum_TRITONSERVER_TYPE_BF16 => 2,
        _ => 0,
    }
}
