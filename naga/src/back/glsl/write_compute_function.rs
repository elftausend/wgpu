use std::{collections::HashSet, fmt::Write};

use crate::{
    back,
    proc::{NameKey, TypeResolution},
    valid, Arena, Binding, BuiltIn, Expression, GlobalVariable, Handle, Statement, TypeInner,
};

use super::{
    glsl_storage_format, is_value_init_supported, BackendResult, VaryingName, VaryingOptions,
    Writer,
};

impl<'a, W: Write> Writer<'a, W> {
    pub fn extract_global_variable(
        &self,
        expressions: &Arena<Expression>,
        pointer: Handle<Expression>,
    ) -> Option<(Handle<GlobalVariable>, &GlobalVariable)> {
        let out = &expressions[pointer];
        let Expression::Access { base, index: _ } = out else {
            return None;
        };
        let Expression::GlobalVariable(global_var_handle) = &expressions[*base] else {
            return None;
        };
        let global_var = &self.module.global_variables[*global_var_handle];
        Some((*global_var_handle, global_var))
    }
    pub fn write_outputs(
        &mut self,
        func: &crate::Function,
        output_global: &mut Vec<Handle<GlobalVariable>>,
    ) -> Result<(), std::fmt::Error> {
        for stmt in self.entry_point.function.body.iter() {
            match stmt {
                Statement::Store { pointer, value: _ } => {
                    let Some((global_var_handle, global_var)) =
                        self.extract_global_variable(&func.expressions, *pointer)
                    else {
                        continue;
                    };
                    let global_name = self.get_global_name(global_var_handle, global_var);
                    writeln!(
                        self.out,
                        "layout(location = {}) out vec4 {global_name};",
                        output_global.len()
                    )?;
                    output_global.push(global_var_handle);
                    // dbg!(global_var);
                    // if let TypeResolution::Value(TypeInner::Pointer { base, space }) = out {
                    //     &self.module.global_variables[*base];
                    //     let out = &self.module.types[*base];
                    //     dbg!(out);
                    // }
                }
                _ => continue,
            }
        }
        write!(self.out, "")
    }

    pub fn write_gws(&mut self) -> Result<(), std::fmt::Error> {
        writeln!(
            self.out,
            "
uniform uint gws_x;
uniform uint gws_y;
uniform uint gws_z;
        "
        )
    }

    pub fn write_global_invocation_vec(&mut self) -> Result<(), std::fmt::Error> {
        writeln!(
            self.out,
            "
    uint absolute_col = uint(thread_uv.x * float(thread_viewport_width));
    uint absolute_row = uint(thread_uv.y * float(thread_viewport_height));
    uint idx = absolute_row * thread_viewport_width + absolute_col;

    uint x_idx = idx % gws_x;
    uint y_idx = (idx / gws_x) % gws_y;
    uint z_idx = (idx / (gws_x * gws_y)) % gws_z;

    uvec3 global_id = uvec3(x_idx, y_idx, z_idx);
        "
        )
    }

    pub fn write_nd_select_fns(&mut self) -> Result<(), std::fmt::Error> {
        // select_1D is practically the same as select_2D
        // thread_uv is already 2D -> calculation from 2D to 1D happens before -> redundant
        writeln!(self.out, "
vec2 select_from_idx(uint textureWidth, uint textureHeight, uint idx) {{
  float col = float(idx % textureWidth) + 0.5;
  float row = float(idx / textureWidth) + 0.5;
  return vec2 (
    col / float(textureWidth),
    row / float(textureHeight)
  );
}}
        ")
    }

    pub fn write_decode_encode(&mut self) -> Result<(), std::fmt::Error> {
        writeln!(
            self.out,
            "vec2 int_mod(vec2 x, float y) {{
  vec2 res = floor(mod(x, y));
  return res * step(1.0 - floor(y), -res);
}}

vec3 int_mod(vec3 x, float y) {{
  vec3 res = floor(mod(x, y));
  return res * step(1.0 - floor(y), -res);
}}

vec4 int_mod(vec4 x, vec4 y) {{
  vec4 res = floor(mod(x, y));
  return res * step(1.0 - floor(y), -res);
}}

highp float int_mod(highp float x, highp float y) {{
  highp float res = floor(mod(x, y));
  return res * (res > floor(y) - 1.0 ? 0.0 : 1.0);
}}

highp int int_mod(highp int x, highp int y) {{
  return int(int_mod(float(x), float(y)));
}}

const vec2 MAGIC_VEC        = vec2(1.0, -256.0);
const vec4 SCALE_FACTOR     = vec4(1.0, 256.0, 65536.0, 0.0);
const vec4 SCALE_FACTOR_INV = vec4(1.0, 0.00390625, 0.0000152587890625, 0.0); // 1, 1/256, 1/65536);

highp float decode(highp vec4 rgba) {{

  rgba *= 255.0;
  vec2 gte128;
  gte128.x = rgba.b >= 128.0 ? 1.0 : 0.0;
  gte128.y = rgba.a >= 128.0 ? 1.0 : 0.0;
  float exponent = 2.0 * rgba.a - 127.0 + dot(gte128, MAGIC_VEC);
  float res = exp2(round(exponent));
  rgba.b = rgba.b - 128.0 * gte128.x;
  res = dot(rgba, SCALE_FACTOR) * exp2(round(exponent-23.0)) + res;
  res *= gte128.y * -2.0 + 1.0;
  return res;
}}

highp vec4 encode(highp float f) {{
  highp float F = abs(f);
  highp float sign = f < 0.0 ? 1.0 : 0.0;
  highp float exponent = floor(log2(F));
  highp float mantissa = (exp2(-exponent) * F);
  // exponent += floor(log2(mantissa));
  vec4 rgba = vec4(F * exp2(23.0-exponent)) * SCALE_FACTOR_INV;
  rgba.rg = int_mod(rgba.rg, 256.0);
  rgba.b = int_mod(rgba.b, 128.0);
  rgba.a = exponent*0.5 + 63.5;
  rgba.ba += vec2(int_mod(exponent+127.0, 2.0), sign) * 128.0;
  rgba = floor(rgba);
  rgba *= 0.003921569; // 1/255

  return rgba;
}}
        "
        )
    }

    pub fn write_compute_function(
        &mut self,
        ty: back::FunctionType,
        func: &crate::Function,
        info: &valid::FunctionInfo,
        output_globals: &Vec<Handle<GlobalVariable>>,
    ) -> BackendResult {
        // Create a function context for the function being written
        let ctx = back::FunctionCtx {
            ty,
            info,
            expressions: &func.expressions,
            named_expressions: &func.named_expressions,
        };

        self.named_expressions.clear();
        self.update_expressions_to_bake(func, info);

        // Write the function header
        //
        // glsl headers are the same as in c:
        // `ret_type name(args)`
        // `ret_type` is the return type
        // `name` is the function name
        // `args` is a comma separated list of `type name`
        //  | - `type` is the argument type
        //  | - `name` is the argument name

        // Start by writing the return type if any otherwise write void
        // This is the only place where `void` is a valid type
        // (though it's more a keyword than a type)
        if let back::FunctionType::EntryPoint(_) = ctx.ty {
            write!(self.out, "void")?;
        } else if let Some(ref result) = func.result {
            self.write_type(result.ty)?;
            if let TypeInner::Array { base, size, .. } = self.module.types[result.ty].inner {
                self.write_array_size(base, size)?
            }
        } else {
            write!(self.out, "void")?;
        }

        // Write the function name and open parentheses for the argument list
        let function_name = match ctx.ty {
            back::FunctionType::Function(handle) => &self.names[&NameKey::Function(handle)],
            back::FunctionType::EntryPoint(_) => "main",
        };
        write!(self.out, " {function_name}(")?;

        // Write the comma separated argument list
        //
        // We need access to `Self` here so we use the reference passed to the closure as an
        // argument instead of capturing as that would cause a borrow checker error
        let arguments = match ctx.ty {
            back::FunctionType::EntryPoint(_) => &[][..],
            back::FunctionType::Function(_) => &func.arguments,
        };
        let arguments: Vec<_> = arguments
            .iter()
            .enumerate()
            .filter(|&(_, arg)| match self.module.types[arg.ty].inner {
                TypeInner::Sampler { .. } => false,
                _ => true,
            })
            .collect();
        self.write_slice(&arguments, |this, _, &(i, arg)| {
            // Write the argument type
            match this.module.types[arg.ty].inner {
                // We treat images separately because they might require
                // writing the storage format
                TypeInner::Image {
                    dim,
                    arrayed,
                    class,
                } => {
                    // Write the storage format if needed
                    if let TypeInner::Image {
                        class: crate::ImageClass::Storage { format, .. },
                        ..
                    } = this.module.types[arg.ty].inner
                    {
                        write!(this.out, "layout({}) ", glsl_storage_format(format)?)?;
                    }

                    // write the type
                    //
                    // This is way we need the leading space because `write_image_type` doesn't add
                    // any spaces at the beginning or end
                    this.write_image_type(dim, arrayed, class)?;
                }
                TypeInner::Pointer { base, .. } => {
                    // write parameter qualifiers
                    write!(this.out, "inout ")?;
                    this.write_type(base)?;
                }
                // All other types are written by `write_type`
                _ => {
                    this.write_type(arg.ty)?;
                }
            }

            // Write the argument name
            // The leading space is important
            write!(this.out, " {}", &this.names[&ctx.argument_key(i as u32)])?;

            // Write array size
            match this.module.types[arg.ty].inner {
                TypeInner::Array { base, size, .. } => {
                    this.write_array_size(base, size)?;
                }
                TypeInner::Pointer { base, .. } => {
                    if let TypeInner::Array { base, size, .. } = this.module.types[base].inner {
                        this.write_array_size(base, size)?;
                    }
                }
                _ => {}
            }

            Ok(())
        })?;

        // Close the parentheses and open braces to start the function body
        writeln!(self.out, ") {{")?;

        if self.options.zero_initialize_workgroup_memory
            && ctx.ty.is_compute_entry_point(self.module)
        {
            self.write_workgroup_variables_initialization(&ctx)?;
        }

        // Compose the function arguments from globals, in case of an entry point.
        if let back::FunctionType::EntryPoint(ep_index) = ctx.ty {
            let stage = self.module.entry_points[ep_index as usize].stage;
            for (index, arg) in func.arguments.iter().enumerate() {
                if let Binding::BuiltIn(bi) = arg.binding.as_ref().unwrap() {
                    if bi == &BuiltIn::GlobalInvocationId {
                        self.write_global_invocation_vec()?;
                        continue;
                    }
                }

                write!(self.out, "{}", back::INDENT)?;
                self.write_type(arg.ty)?;
                let name = &self.names[&NameKey::EntryPointArgument(ep_index, index as u32)];
                write!(self.out, " {name}")?;
                write!(self.out, " = ")?;
                match self.module.types[arg.ty].inner {
                    crate::TypeInner::Struct { ref members, .. } => {
                        todo!()
                        /*self.write_type(arg.ty)?;
                        write!(self.out, "(")?;
                        for (index, member) in members.iter().enumerate() {
                            let varying_name = VaryingName {
                                binding: member.binding.as_ref().unwrap(),
                                stage,
                                options: VaryingOptions::from_writer_options(self.options, false),
                            };
                            if index != 0 {
                                write!(self.out, ", ")?;
                            }
                            write!(self.out, "{varying_name}")?;
                        }
                        writeln!(self.out, ");")?;*/
                    }
                    _ => {
                        let varying_name = VaryingName {
                            binding: arg.binding.as_ref().unwrap(),
                            stage,
                            options: VaryingOptions::from_writer_options(self.options, false),
                        };

                        writeln!(self.out, "{varying_name};")?;
                    }
                }
            }
        }

        // Write all function locals
        // Locals are `type name (= init)?;` where the init part (including the =) are optional
        //
        // Always adds a newline
        for (handle, local) in func.local_variables.iter() {
            // Write indentation (only for readability) and the type
            // `write_type` adds no trailing space
            write!(self.out, "{}", back::INDENT)?;
            self.write_type(local.ty)?;

            // Write the local name
            // The leading space is important
            write!(self.out, " {}", self.names[&ctx.name_key(handle)])?;
            // Write size for array type
            if let TypeInner::Array { base, size, .. } = self.module.types[local.ty].inner {
                self.write_array_size(base, size)?;
            }
            // Write the local initializer if needed
            if let Some(init) = local.init {
                // Put the equal signal only if there's a initializer
                // The leading and trailing spaces aren't needed but help with readability
                write!(self.out, " = ")?;

                // Write the constant
                // `write_constant` adds no trailing or leading space/newline
                self.write_expr(init, &ctx)?;
            } else if is_value_init_supported(self.module, local.ty) {
                write!(self.out, " = ")?;
                self.write_zero_init_value(local.ty)?;
            }

            writeln!(self.out, ";")?
        }

        // Write the function body (statement list)
        for sta in func.body.iter() {
            if let back::FunctionType::EntryPoint(_) = ctx.ty {
                if let Statement::Return { value: _ } = sta {
                    for output_global_handle in output_globals {
                        let global_var = &self.module.global_variables[*output_global_handle];
                        let name = self.get_global_name(*output_global_handle, global_var);
                        write!(self.out, "{}", back::Level(1))?;
                        writeln!(self.out, "{name} = encode( {name}.r )")?;
                    }
                    write!(self.out, "{}", back::Level(1))?;
                    writeln!(self.out, "return;")?;
                    continue;
                }
            }

            // Write a statement, the indentation should always be 1 when writing the function body
            // `write_stmt` adds a newline
            self.write_compute_stmt(sta, &ctx, back::Level(1))?;
        }

        if let back::FunctionType::EntryPoint(_) = ctx.ty {}

        // Close braces and add a newline
        writeln!(self.out, "}}")?;

        Ok(())
    }
}
