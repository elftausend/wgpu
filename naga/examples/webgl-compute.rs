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

    let info = validator.validate(&module).unwrap();

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
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
                if global_id.x >= arrayLength(&out) {
                    return;    
                }

                var counter = 0.0;
                for (var i = 0u; i < global_id.x; i++) {
                    counter += 1.0;
                }

                // if out is used on the right side: problem at the moment
                out[global_id.x] = 3.0 * x[global_id.x];
            }

            ";
    let src = "
            @group(0)
            @binding(0)
            var<storage, read> x: array<u32>;

            @group(0)
            @binding(1)
            var<storage, read_write> out: array<u32>;
            
            @compute
            @workgroup_size(32)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
                if global_id.x >= arrayLength(&out) {
                    return;    
                }
                out[global_id.x] = x[global_id.x];
            }

            ";
    let _src = "
            @group(0)
            @binding(0)
            var<storage, read> x: array<f32>;

            @group(0)
            @binding(1)
            var<storage, read_write> out: array<f32>;
            
            @group(0)
            @binding(2)
            var<uniform> add: f32;
            
            @group(0)
            @binding(3)
            var<uniform> add_another: f32;

            @compute
            @workgroup_size(32)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
                if global_id.x >= arrayLength(&out) {
                    return;    
                }

                var counter = add;
                for (var i = 0u; i < global_id.x; i++) {
                    counter += 1.0;
                }

                // if out is used on the right side: problem at the moment
                out[global_id.x] = 3.0 * x[global_id.x] + add_another;
            }

            ";

            let src = "
            @group(0) @binding(0)
            var<storage, read_write> labels: array<u32>;
            
            @group(0) @binding(1)
            var<storage, read_write> links: array<u32>;
    
            @group(0) @binding(2)
            var<storage, read_write> classified_labels: array<u32>;
            
            @group(0) @binding(3)
            var<uniform> width: u32;
            
            @compute
            @workgroup_size(32)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                var out_idx = global_id.y * width + global_id.x; 
                if out_idx >= arrayLength(&classified_labels) {
                    return;
                }
                var current_label = labels[out_idx];
                var link_idx = global_id.y * width * 4u + global_id.x * 4u; 
                var link_x = links[link_idx];
                var link_y = links[link_idx + 1];
                var link_z = links[link_idx + 2];
                var link_w = links[link_idx + 3];
    
                if link_x == 0 && link_y == 0 {
                    var root_candidate_label = (1u << 31u) | current_label;
                    classified_labels[out_idx] = root_candidate_label;
                } else {
                    classified_labels[out_idx] = current_label;
                }
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
    std::fs::write("output", glsl).unwrap();
}
