use naga::{
    back::glsl::{Options, PipelineOptions, Version, Writer},
    proc::BoundsCheckPolicies,
    valid::ModuleInfo,
};

pub fn parse_and_validate_wgsl(
    src: &str,
) -> Result<(naga::Module, ModuleInfo), Box<dyn std::error::Error>> {
    let mut frontend = naga::front::wgsl::Frontend::new();

    let module = frontend.parse(src)?;

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );

    let info = validator.validate(&module)?;

    Ok((module, info))
}

fn main() {
    let src = "
                @group(0)
                @binding(0)
                var<storage, read> x: array<f32>;

                @group(0)
                @binding(1)
                var<storage, read_write> out: array<f32>;
                
                @compute
                @workgroup_size(32)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    /*if global_id.x >= arrayLength(&out) {
                        return;    
                    }*/
                    out[global_id.x] = 3.0 * x[global_id.x];
                }

            ";
    let (module, info) = parse_and_validate_wgsl(&src).unwrap();

    // 310 is required for compute shaders
    let version = Version::Embedded {
        version: 310,
        is_webgl: true,
    };
    let mut glsl = String::new();
    let options = Options {
        version,
        ..Default::default()
    };
    let pipeline_options = PipelineOptions {
        shader_stage: naga::ShaderStage::Compute,
        entry_point: "main".into(),
        multiview: None,
    };

    let mut glsl_complete = String::new();
    let mut writer = Writer::new(
        &mut glsl_complete,
        &module,
        &info,
        &options,
        &pipeline_options,
        BoundsCheckPolicies::default(),
    )
    .unwrap();

    writer.write().unwrap();
    println!("glsl: {glsl_complete}");

    println!("modu: {module:?}");

    let mut writer = Writer::new(
        &mut glsl,
        &module,
        &info,
        &options,
        &pipeline_options,
        BoundsCheckPolicies::default(),
    )
    .unwrap();
    writer.write_webgl_compute().unwrap();

    println!("compute glsl: {glsl}");
}
