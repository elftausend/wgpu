use std::fmt::Write;

use super::{BackendResult, Writer};
use crate::{
    back::{
        self,
        glsl::{Error, VaryingName, VaryingOptions, WriterFlags},
    },
    proc::{self, NameKey},
    ShaderStage, TypeInner,
};

impl<'a, W: Write> Writer<'a, W> {
    pub(super) fn write_compute_stmt(
        &mut self,
        sta: &crate::Statement,
        ctx: &back::FunctionCtx,
        level: back::Level,
    ) -> BackendResult {
        use crate::Statement;
        match *sta {
            // This is where we can generate intermediate constants for some expression types.
            Statement::Emit(ref range) => {
                for handle in range.clone() {
                    let ptr_class = ctx.resolve_type(handle, &self.module.types).pointer_space();
                    let expr_name = if ptr_class.is_some() {
                        // GLSL can't save a pointer-valued expression in a variable,
                        // but we shouldn't ever need to: they should never be named expressions,
                        // and none of the expression types flagged by bake_ref_count can be pointer-valued.
                        None
                    } else if let Some(name) = ctx.named_expressions.get(&handle) {
                        // Front end provides names for all variables at the start of writing.
                        // But we write them to step by step. We need to recache them
                        // Otherwise, we could accidentally write variable name instead of full expression.
                        // Also, we use sanitized names! It defense backend from generating variable with name from reserved keywords.
                        Some(self.namer.call(name))
                    } else if self.need_bake_expressions.contains(&handle) {
                        Some(format!("{}{}", back::BAKE_PREFIX, handle.index()))
                    } else {
                        None
                    };

                    // If we are going to write an `ImageLoad` next and the target image
                    // is sampled and we are using the `Restrict` policy for bounds
                    // checking images we need to write a local holding the clamped lod.
                    if let crate::Expression::ImageLoad {
                        image,
                        level: Some(level_expr),
                        ..
                    } = ctx.expressions[handle]
                    {
                        if let TypeInner::Image {
                            class: crate::ImageClass::Sampled { .. },
                            ..
                        } = *ctx.resolve_type(image, &self.module.types)
                        {
                            if let proc::BoundsCheckPolicy::Restrict = self.policies.image_load {
                                write!(self.out, "{level}")?;
                                self.write_clamped_lod(ctx, handle, image, level_expr)?
                            }
                        }
                    }

                    if let Some(name) = expr_name {
                        write!(self.out, "{level}")?;
                        self.write_named_compute_expr(handle, name, handle, ctx)?;
                    }
                }
            }
            // Blocks are simple we just need to write the block statements between braces
            // We could also just print the statements but this is more readable and maps more
            // closely to the IR
            Statement::Block(ref block) => {
                write!(self.out, "{level}")?;
                writeln!(self.out, "{{")?;
                for sta in block.iter() {
                    // Increase the indentation to help with readability
                    self.write_compute_stmt(sta, ctx, level.next())?
                }
                writeln!(self.out, "{level}}}")?
            }
            // Ifs are written as in C:
            // ```
            // if(condition) {
            //  accept
            // } else {
            //  reject
            // }
            // ```
            Statement::If {
                condition,
                ref accept,
                ref reject,
            } => {
                write!(self.out, "{level}")?;
                write!(self.out, "if (")?;
                self.write_compute_expr(condition, ctx)?;
                writeln!(self.out, ") {{")?;

                for sta in accept {
                    // Increase indentation to help with readability
                    self.write_compute_stmt(sta, ctx, level.next())?;
                }

                // If there are no statements in the reject block we skip writing it
                // This is only for readability
                if !reject.is_empty() {
                    writeln!(self.out, "{level}}} else {{")?;

                    for sta in reject {
                        // Increase indentation to help with readability
                        self.write_compute_stmt(sta, ctx, level.next())?;
                    }
                }

                writeln!(self.out, "{level}}}")?
            }
            // Switch are written as in C:
            // ```
            // switch (selector) {
            //      // Fallthrough
            //      case label:
            //          block
            //      // Non fallthrough
            //      case label:
            //          block
            //          break;
            //      default:
            //          block
            //  }
            //  ```
            //  Where the `default` case happens isn't important but we put it last
            //  so that we don't need to print a `break` for it
            Statement::Switch {
                selector,
                ref cases,
            } => {
                // Start the switch
                write!(self.out, "{level}")?;
                write!(self.out, "switch(")?;
                self.write_compute_expr(selector, ctx)?;
                writeln!(self.out, ") {{")?;

                // Write all cases
                let l2 = level.next();
                for case in cases {
                    match case.value {
                        crate::SwitchValue::I32(value) => write!(self.out, "{l2}case {value}:")?,
                        crate::SwitchValue::U32(value) => write!(self.out, "{l2}case {value}u:")?,
                        crate::SwitchValue::Default => write!(self.out, "{l2}default:")?,
                    }

                    let write_block_braces = !(case.fall_through && case.body.is_empty());
                    if write_block_braces {
                        writeln!(self.out, " {{")?;
                    } else {
                        writeln!(self.out)?;
                    }

                    for sta in case.body.iter() {
                        self.write_compute_stmt(sta, ctx, l2.next())?;
                    }

                    if !case.fall_through && case.body.last().map_or(true, |s| !s.is_terminator()) {
                        writeln!(self.out, "{}break;", l2.next())?;
                    }

                    if write_block_braces {
                        writeln!(self.out, "{l2}}}")?;
                    }
                }

                writeln!(self.out, "{level}}}")?
            }
            // Loops in naga IR are based on wgsl loops, glsl can emulate the behaviour by using a
            // while true loop and appending the continuing block to the body resulting on:
            // ```
            // bool loop_init = true;
            // while(true) {
            //  if (!loop_init) { <continuing> }
            //  loop_init = false;
            //  <body>
            // }
            // ```
            Statement::Loop {
                ref body,
                ref continuing,
                break_if,
            } => {
                if !continuing.is_empty() || break_if.is_some() {
                    let gate_name = self.namer.call("loop_init");
                    writeln!(self.out, "{level}bool {gate_name} = true;")?;
                    writeln!(self.out, "{level}while(true) {{")?;
                    let l2 = level.next();
                    let l3 = l2.next();
                    writeln!(self.out, "{l2}if (!{gate_name}) {{")?;
                    for sta in continuing {
                        self.write_compute_stmt(sta, ctx, l3)?;
                    }
                    if let Some(condition) = break_if {
                        write!(self.out, "{l3}if (")?;
                        self.write_compute_expr(condition, ctx)?;
                        writeln!(self.out, ") {{")?;
                        writeln!(self.out, "{}break;", l3.next())?;
                        writeln!(self.out, "{l3}}}")?;
                    }
                    writeln!(self.out, "{l2}}}")?;
                    writeln!(self.out, "{}{} = false;", level.next(), gate_name)?;
                } else {
                    writeln!(self.out, "{level}while(true) {{")?;
                }
                for sta in body {
                    self.write_compute_stmt(sta, ctx, level.next())?;
                }
                writeln!(self.out, "{level}}}")?
            }
            // Break, continue and return as written as in C
            // `break;`
            Statement::Break => {
                write!(self.out, "{level}")?;
                writeln!(self.out, "break;")?
            }
            // `continue;`
            Statement::Continue => {
                write!(self.out, "{level}")?;
                writeln!(self.out, "continue;")?
            }
            // `return expr;`, `expr` is optional
            Statement::Return { value } => {
                write!(self.out, "{level}")?;
                match ctx.ty {
                    back::FunctionType::Function(_) => {
                        write!(self.out, "return")?;
                        // Write the expression to be returned if needed
                        if let Some(expr) = value {
                            write!(self.out, " ")?;
                            self.write_compute_expr(expr, ctx)?;
                        }
                        writeln!(self.out, ";")?;
                    }
                    back::FunctionType::EntryPoint(ep_index) => {
                        for output_global_handle in &self.output_globals {
                            let global_var = &self.module.global_variables[*output_global_handle];
                            let Some(datatype_prefix) = self
                                .extract_data_type_prefix_from_array(
                                    &self.module.types[global_var.ty],
                                )
                                .map(|prefix| prefix.to_string())
                            else {
                                return Err(Error::Custom(
                                    "Unsupported datatype in global variable".into(),
                                ));
                            };
                            let name = self.get_global_name(*output_global_handle, global_var);
                            writeln!(self.out, "{name} = {datatype_prefix}encode( {name}.r );")?;
                            write!(self.out, "{level}")?;
                        }

                        // let ep = &self.module.entry_points[ep_index as usize];
                        // if let Some(ref result) = ep.function.result {}
                        writeln!(self.out, "return;")?;
                    }
                }
            }
            // This is one of the places were glsl adds to the syntax of C in this case the discard
            // keyword which ceases all further processing in a fragment shader, it's called OpKill
            // in spir-v that's why it's called `Statement::Kill`
            Statement::Kill => writeln!(self.out, "{level}discard;")?,
            Statement::Barrier(flags) => {
                self.write_barrier(flags, level)?;
            }
            // Stores in glsl are just variable assignments written as `pointer = value;`
            Statement::Store { pointer, value } => {
                write!(self.out, "{level}")?;
                if let Some((handle, global_var)) =
                    self.extract_global_variable(ctx.expressions, pointer)
                {
                    write!(
                        self.out,
                        "{}.r = ",
                        self.get_global_name(handle, global_var)
                    )?;
                    self.write_compute_expr(value, ctx)?;
                    writeln!(self.out, ";")?;
                    return Ok(());
                }
                self.write_compute_expr(pointer, ctx)?;
                write!(self.out, " = ")?;
                self.write_compute_expr(value, ctx)?;
                writeln!(self.out, ";")?
            }
            Statement::WorkGroupUniformLoad { pointer, result } => {
                todo!();
                // GLSL doesn't have pointers, which means that this backend needs to ensure that
                // the actual "loading" is happening between the two barriers.
                // This is done in `Emit` by never emitting a variable name for pointer variables
                self.write_barrier(crate::Barrier::WORK_GROUP, level)?;

                let result_name = format!("{}{}", back::BAKE_PREFIX, result.index());
                write!(self.out, "{level}")?;
                // Expressions cannot have side effects, so just writing the expression here is fine.
                self.write_named_compute_expr(pointer, result_name, result, ctx)?;

                self.write_barrier(crate::Barrier::WORK_GROUP, level)?;
            }
            // Stores a value into an image.
            Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                todo!()
            }
            // A `Call` is written `name(arguments)` where `arguments` is a comma separated expressions list
            Statement::Call {
                function,
                ref arguments,
                result,
            } => {
                todo!();
                write!(self.out, "{level}")?;
                if let Some(expr) = result {
                    let name = format!("{}{}", back::BAKE_PREFIX, expr.index());
                    let result = self.module.functions[function].result.as_ref().unwrap();
                    self.write_type(result.ty)?;
                    write!(self.out, " {name}")?;
                    if let TypeInner::Array { base, size, .. } = self.module.types[result.ty].inner
                    {
                        self.write_array_size(base, size)?
                    }
                    write!(self.out, " = ")?;
                    self.named_expressions.insert(expr, name);
                }
                write!(self.out, "{}(", &self.names[&NameKey::Function(function)])?;
                let arguments: Vec<_> = arguments
                    .iter()
                    .enumerate()
                    .filter_map(|(i, arg)| {
                        let arg_ty = self.module.functions[function].arguments[i].ty;
                        match self.module.types[arg_ty].inner {
                            TypeInner::Sampler { .. } => None,
                            _ => Some(*arg),
                        }
                    })
                    .collect();
                self.write_slice(&arguments, |this, _, arg| {
                    this.write_compute_expr(*arg, ctx)
                })?;
                writeln!(self.out, ");")?
            }
            Statement::Atomic {
                pointer,
                ref fun,
                value,
                result,
            } => {
                todo!()
            }
            Statement::RayQuery { .. } => unreachable!(),
            Statement::SubgroupBallot { result, predicate } => {
                todo!()
            }
            Statement::SubgroupCollectiveOperation {
                op,
                collective_op,
                argument,
                result,
            } => {
                todo!()
            }
            Statement::SubgroupGather {
                mode,
                argument,
                result,
            } => {
                todo!()
            }
        }

        Ok(())
    }
}
