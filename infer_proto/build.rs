fn main() -> Result<(), Box<dyn std::error::Error>> {
    let builder = tonic_build::configure();
    let mut config = prost_build::Config::new();
    config.type_attribute(".", "#[derive(::serde::Serialize, ::serde::Deserialize)]");
    config.type_attribute(".", "#[serde(rename_all = \"camelCase\")]");
    builder.compile_with_config(config, &["src/grpc_predict_v2.proto"], &[""])?;
    // config.compile_protos(&["src/grpc_predict_v2.proto"], &["src"] )?;
    Ok(())
}
