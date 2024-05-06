use core::panic;
use std::fmt::Write;

use crate::{
    back::{
        self,
        glsl::{glsl_scalar, BinaryOperation, Error, Features, FREXP_FUNCTION, MODF_FUNCTION},
    },
    proc::{self, NameKey},
    Handle, TypeInner,
};

use super::{BackendResult, Writer};

impl<'a, W: Write> Writer<'a, W> {
    pub(super) fn write_named_compute_expr(
        &mut self,
        handle: Handle<crate::Expression>,
        name: String,
        // The expression which is being named.
        // Generally, this is the same as handle, except in WorkGroupUniformLoad
        named: Handle<crate::Expression>,
        ctx: &back::FunctionCtx,
    ) -> BackendResult {
        dbg!("called named cer");
        match ctx.info[named].ty {
            proc::TypeResolution::Handle(ty_handle) => match self.module.types[ty_handle].inner {
                TypeInner::Struct { .. } => {
                    let ty_name = &self.names[&NameKey::Type(ty_handle)];
                    write!(self.out, "{ty_name}")?;
                }
                _ => {
                    self.write_type(ty_handle)?;
                }
            },
            proc::TypeResolution::Value(ref inner) => {
                self.write_value_type(inner)?;
            }
        }

        let resolved = ctx.resolve_type(named, &self.module.types);
        write!(self.out, " {name}")?;
        if let TypeInner::Array { base, size, .. } = *resolved {
            self.write_array_size(base, size)?;
        }
        write!(self.out, " = ")?;
        self.write_compute_expr(handle, ctx)?;
        writeln!(self.out, ";")?;
        self.named_expressions.insert(named, name);

        Ok(())
    }

    /// Helper method to write expressions
    ///
    /// # Notes
    /// Doesn't add any newlines or leading/trailing spaces
    pub(super) fn write_compute_expr(
        &mut self,
        expr: Handle<crate::Expression>,
        ctx: &back::FunctionCtx,
    ) -> BackendResult {
        use crate::Expression;

        if let Some(name) = self.named_expressions.get(&expr) {
            write!(self.out, "{name}")?;
            return Ok(());
        }

        match ctx.expressions[expr] {
            Expression::Literal(_)
            | Expression::Constant(_)
            | Expression::ZeroValue(_)
            | Expression::Compose { .. }
            | Expression::Splat { .. } => {
                self.write_possibly_const_expr(
                    expr,
                    ctx.expressions,
                    |expr| &ctx.info[expr].ty,
                    |writer, expr| writer.write_compute_expr(expr, ctx),
                )?;
            }
            Expression::Override(_) => return Err(Error::Override),
            // `Access` is applied to arrays, vectors and matrices and is written as indexing
            Expression::Access { base, index } => {
                dbg!(&ctx.expressions[base]);
                dbg!(&ctx.expressions[index]);

                write!(self.out, "decode( texture( ")?;
                self.write_compute_expr(base, ctx)?;
                write!(self.out, ", select_from_idx( ")?;
                self.write_compute_expr(base, ctx)?;
                write!(self.out, "_texture_width, ")?;

                self.write_compute_expr(base, ctx)?;
                write!(self.out, "_texture_height, ")?;
                self.write_compute_expr(index, ctx)?;
                write!(self.out, " )))")?;
            }
            // `AccessIndex` is the same as `Access` except that the index is a constant and it can
            // be applied to structs, in this case we need to find the name of the field at that
            // index and write `base.field_name`
            Expression::AccessIndex { base, index } => {
                dbg!(&ctx.expressions[base]);
                self.write_compute_expr(base, ctx)?;

                let base_ty_res = &ctx.info[base].ty;
                let mut resolved = base_ty_res.inner_with(&self.module.types);
                let base_ty_handle = match *resolved {
                    TypeInner::Pointer { base, space: _ } => {
                        resolved = &self.module.types[base].inner;
                        Some(base)
                    }
                    _ => base_ty_res.handle(),
                };

                dbg!(resolved);

                match *resolved {
                    TypeInner::Vector { .. } => {
                        // Write vector access as a swizzle
                        write!(self.out, ".{}", back::COMPONENTS[index as usize])?
                    }
                    TypeInner::Matrix { .. }
                    | TypeInner::Array { .. }
                    | TypeInner::ValuePointer { .. } => write!(self.out, "[{index}]")?,
                    TypeInner::Struct { .. } => {
                        // This will never panic in case the type is a `Struct`, this is not true
                        // for other types so we can only check while inside this match arm
                        let ty = base_ty_handle.unwrap();

                        write!(
                            self.out,
                            ".{}",
                            &self.names[&NameKey::StructMember(ty, index)]
                        )?
                    }
                    ref other => return Err(Error::Custom(format!("Cannot index {other:?}"))),
                }
            }
            // `Swizzle` adds a few letters behind the dot.
            Expression::Swizzle {
                size,
                vector,
                pattern,
            } => {
                self.write_compute_expr(vector, ctx)?;
                write!(self.out, ".")?;
                for &sc in pattern[..size as usize].iter() {
                    self.out.write_char(back::COMPONENTS[sc as usize])?;
                }
            }
            // Function arguments are written as the argument name
            Expression::FunctionArgument(pos) => {
                write!(self.out, "{}", &self.names[&ctx.argument_key(pos)])?
            }
            // Global variables need some special work for their name but
            // `get_global_name` does the work for us
            Expression::GlobalVariable(handle) => {
                let global = &self.module.global_variables[handle];
                self.write_global_name(handle, global)?
            }
            // A local is written as it's name
            Expression::LocalVariable(handle) => {
                write!(self.out, "{}", self.names[&ctx.name_key(handle)])?
            }
            // glsl has no pointers so there's no load operation, just write the pointer expression
            Expression::Load { pointer } => self.write_compute_expr(pointer, ctx)?,
            // `ImageSample` is a bit complicated compared to the rest of the IR.
            //
            // First there are three variations depending whether the sample level is explicitly set,
            // if it's automatic or it it's bias:
            // `texture(image, coordinate)` - Automatic sample level
            // `texture(image, coordinate, bias)` - Bias sample level
            // `textureLod(image, coordinate, level)` - Zero or Exact sample level
            //
            // Furthermore if `depth_ref` is some we need to append it to the coordinate vector
            Expression::ImageSample {
                image,
                sampler: _, //TODO?
                gather,
                coordinate,
                array_index,
                offset,
                level,
                depth_ref,
            } => {
                // removed to focus on currently important stuff
                todo!()
            }
            Expression::ImageLoad {
                image,
                coordinate,
                array_index,
                sample,
                level,
            } => self.write_image_load(expr, ctx, image, coordinate, array_index, sample, level)?,
            // Query translates into one of the:
            // - textureSize/imageSize
            // - textureQueryLevels
            // - textureSamples/imageSamples
            Expression::ImageQuery { image, query } => {
                todo!()
            }
            Expression::Unary { op, expr } => {
                let operator_or_fn = match op {
                    crate::UnaryOperator::Negate => "-",
                    crate::UnaryOperator::LogicalNot => {
                        match *ctx.resolve_type(expr, &self.module.types) {
                            TypeInner::Vector { .. } => "not",
                            _ => "!",
                        }
                    }
                    crate::UnaryOperator::BitwiseNot => "~",
                };
                write!(self.out, "{operator_or_fn}(")?;

                self.write_compute_expr(expr, ctx)?;

                write!(self.out, ")")?
            }
            // `Binary` we just write `left op right`, except when dealing with
            // comparison operations on vectors as they are implemented with
            // builtin functions.
            // Once again we wrap everything in parentheses to avoid precedence issues
            Expression::Binary {
                mut op,
                left,
                right,
            } => {
                // Holds `Some(function_name)` if the binary operation is
                // implemented as a function call
                use crate::{BinaryOperator as Bo, ScalarKind as Sk, TypeInner as Ti};

                let left_inner = ctx.resolve_type(left, &self.module.types);
                let right_inner = ctx.resolve_type(right, &self.module.types);

                let function = match (left_inner, right_inner) {
                    (&Ti::Vector { scalar, .. }, &Ti::Vector { .. }) => match op {
                        Bo::Less
                        | Bo::LessEqual
                        | Bo::Greater
                        | Bo::GreaterEqual
                        | Bo::Equal
                        | Bo::NotEqual => BinaryOperation::VectorCompare,
                        Bo::Modulo if scalar.kind == Sk::Float => BinaryOperation::Modulo,
                        Bo::And if scalar.kind == Sk::Bool => {
                            op = crate::BinaryOperator::LogicalAnd;
                            BinaryOperation::VectorComponentWise
                        }
                        Bo::InclusiveOr if scalar.kind == Sk::Bool => {
                            op = crate::BinaryOperator::LogicalOr;
                            BinaryOperation::VectorComponentWise
                        }
                        _ => BinaryOperation::Other,
                    },
                    _ => match (left_inner.scalar_kind(), right_inner.scalar_kind()) {
                        (Some(Sk::Float), _) | (_, Some(Sk::Float)) => match op {
                            Bo::Modulo => BinaryOperation::Modulo,
                            _ => BinaryOperation::Other,
                        },
                        (Some(Sk::Bool), Some(Sk::Bool)) => match op {
                            Bo::InclusiveOr => {
                                op = crate::BinaryOperator::LogicalOr;
                                BinaryOperation::Other
                            }
                            Bo::And => {
                                op = crate::BinaryOperator::LogicalAnd;
                                BinaryOperation::Other
                            }
                            _ => BinaryOperation::Other,
                        },
                        _ => BinaryOperation::Other,
                    },
                };

                match function {
                    BinaryOperation::VectorCompare => {
                        let op_str = match op {
                            Bo::Less => "lessThan(",
                            Bo::LessEqual => "lessThanEqual(",
                            Bo::Greater => "greaterThan(",
                            Bo::GreaterEqual => "greaterThanEqual(",
                            Bo::Equal => "equal(",
                            Bo::NotEqual => "notEqual(",
                            _ => unreachable!(),
                        };
                        write!(self.out, "{op_str}")?;
                        self.write_compute_expr(left, ctx)?;
                        write!(self.out, ", ")?;
                        self.write_compute_expr(right, ctx)?;
                        write!(self.out, ")")?;
                    }
                    BinaryOperation::VectorComponentWise => {
                        self.write_value_type(left_inner)?;
                        write!(self.out, "(")?;

                        let size = match *left_inner {
                            Ti::Vector { size, .. } => size,
                            _ => unreachable!(),
                        };

                        for i in 0..size as usize {
                            if i != 0 {
                                write!(self.out, ", ")?;
                            }

                            self.write_compute_expr(left, ctx)?;
                            write!(self.out, ".{}", back::COMPONENTS[i])?;

                            write!(self.out, " {} ", back::binary_operation_str(op))?;

                            self.write_compute_expr(right, ctx)?;
                            write!(self.out, ".{}", back::COMPONENTS[i])?;
                        }

                        write!(self.out, ")")?;
                    }
                    // TODO: handle undefined behavior of BinaryOperator::Modulo
                    //
                    // sint:
                    // if right == 0 return 0
                    // if left == min(type_of(left)) && right == -1 return 0
                    // if sign(left) == -1 || sign(right) == -1 return result as defined by WGSL
                    //
                    // uint:
                    // if right == 0 return 0
                    //
                    // float:
                    // if right == 0 return ? see https://github.com/gpuweb/gpuweb/issues/2798
                    BinaryOperation::Modulo => {
                        write!(self.out, "(")?;

                        // write `e1 - e2 * trunc(e1 / e2)`
                        self.write_compute_expr(left, ctx)?;
                        write!(self.out, " - ")?;
                        self.write_compute_expr(right, ctx)?;
                        write!(self.out, " * ")?;
                        write!(self.out, "trunc(")?;
                        self.write_compute_expr(left, ctx)?;
                        write!(self.out, " / ")?;
                        self.write_compute_expr(right, ctx)?;
                        write!(self.out, ")")?;

                        write!(self.out, ")")?;
                    }
                    BinaryOperation::Other => {
                        write!(self.out, "(")?;

                        self.write_compute_expr(left, ctx)?;
                        write!(self.out, " {} ", back::binary_operation_str(op))?;
                        self.write_compute_expr(right, ctx)?;

                        write!(self.out, ")")?;
                    }
                }
            }
            // `Select` is written as `condition ? accept : reject`
            // We wrap everything in parentheses to avoid precedence issues
            Expression::Select {
                condition,
                accept,
                reject,
            } => {
                let cond_ty = ctx.resolve_type(condition, &self.module.types);
                let vec_select = if let TypeInner::Vector { .. } = *cond_ty {
                    true
                } else {
                    false
                };

                // TODO: Boolean mix on desktop required GL_EXT_shader_integer_mix
                if vec_select {
                    // Glsl defines that for mix when the condition is a boolean the first element
                    // is picked if condition is false and the second if condition is true
                    write!(self.out, "mix(")?;
                    self.write_compute_expr(reject, ctx)?;
                    write!(self.out, ", ")?;
                    self.write_compute_expr(accept, ctx)?;
                    write!(self.out, ", ")?;
                    self.write_compute_expr(condition, ctx)?;
                } else {
                    write!(self.out, "(")?;
                    self.write_compute_expr(condition, ctx)?;
                    write!(self.out, " ? ")?;
                    self.write_compute_expr(accept, ctx)?;
                    write!(self.out, " : ")?;
                    self.write_compute_expr(reject, ctx)?;
                }

                write!(self.out, ")")?
            }
            // `Derivative` is a function call to a glsl provided function
            Expression::Derivative { axis, ctrl, expr } => {
                use crate::{DerivativeAxis as Axis, DerivativeControl as Ctrl};
                let fun_name = if self.options.version.supports_derivative_control() {
                    match (axis, ctrl) {
                        (Axis::X, Ctrl::Coarse) => "dFdxCoarse",
                        (Axis::X, Ctrl::Fine) => "dFdxFine",
                        (Axis::X, Ctrl::None) => "dFdx",
                        (Axis::Y, Ctrl::Coarse) => "dFdyCoarse",
                        (Axis::Y, Ctrl::Fine) => "dFdyFine",
                        (Axis::Y, Ctrl::None) => "dFdy",
                        (Axis::Width, Ctrl::Coarse) => "fwidthCoarse",
                        (Axis::Width, Ctrl::Fine) => "fwidthFine",
                        (Axis::Width, Ctrl::None) => "fwidth",
                    }
                } else {
                    match axis {
                        Axis::X => "dFdx",
                        Axis::Y => "dFdy",
                        Axis::Width => "fwidth",
                    }
                };
                write!(self.out, "{fun_name}(")?;
                self.write_compute_expr(expr, ctx)?;
                write!(self.out, ")")?
            }
            // `Relational` is a normal function call to some glsl provided functions
            Expression::Relational { fun, argument } => {
                use crate::RelationalFunction as Rf;

                let fun_name = match fun {
                    Rf::IsInf => "isinf",
                    Rf::IsNan => "isnan",
                    Rf::All => "all",
                    Rf::Any => "any",
                };
                write!(self.out, "{fun_name}(")?;

                self.write_compute_expr(argument, ctx)?;

                write!(self.out, ")")?
            }
            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => {
                use crate::MathFunction as Mf;

                let fun_name = match fun {
                    // comparison
                    Mf::Abs => "abs",
                    Mf::Min => "min",
                    Mf::Max => "max",
                    Mf::Clamp => {
                        let scalar_kind = ctx
                            .resolve_type(arg, &self.module.types)
                            .scalar_kind()
                            .unwrap();
                        match scalar_kind {
                            crate::ScalarKind::Float => "clamp",
                            // Clamp is undefined if min > max. In practice this means it can use a median-of-three
                            // instruction to determine the value. This is fine according to the WGSL spec for float
                            // clamp, but integer clamp _must_ use min-max. As such we write out min/max.
                            _ => {
                                write!(self.out, "min(max(")?;
                                self.write_compute_expr(arg, ctx)?;
                                write!(self.out, ", ")?;
                                self.write_compute_expr(arg1.unwrap(), ctx)?;
                                write!(self.out, "), ")?;
                                self.write_compute_expr(arg2.unwrap(), ctx)?;
                                write!(self.out, ")")?;

                                return Ok(());
                            }
                        }
                    }
                    Mf::Saturate => {
                        write!(self.out, "clamp(")?;

                        self.write_compute_expr(arg, ctx)?;

                        match *ctx.resolve_type(arg, &self.module.types) {
                            crate::TypeInner::Vector { size, .. } => write!(
                                self.out,
                                ", vec{}(0.0), vec{0}(1.0)",
                                back::vector_size_str(size)
                            )?,
                            _ => write!(self.out, ", 0.0, 1.0")?,
                        }

                        write!(self.out, ")")?;

                        return Ok(());
                    }
                    // trigonometry
                    Mf::Cos => "cos",
                    Mf::Cosh => "cosh",
                    Mf::Sin => "sin",
                    Mf::Sinh => "sinh",
                    Mf::Tan => "tan",
                    Mf::Tanh => "tanh",
                    Mf::Acos => "acos",
                    Mf::Asin => "asin",
                    Mf::Atan => "atan",
                    Mf::Asinh => "asinh",
                    Mf::Acosh => "acosh",
                    Mf::Atanh => "atanh",
                    Mf::Radians => "radians",
                    Mf::Degrees => "degrees",
                    // glsl doesn't have atan2 function
                    // use two-argument variation of the atan function
                    Mf::Atan2 => "atan",
                    // decomposition
                    Mf::Ceil => "ceil",
                    Mf::Floor => "floor",
                    Mf::Round => "roundEven",
                    Mf::Fract => "fract",
                    Mf::Trunc => "trunc",
                    Mf::Modf => MODF_FUNCTION,
                    Mf::Frexp => FREXP_FUNCTION,
                    Mf::Ldexp => "ldexp",
                    // exponent
                    Mf::Exp => "exp",
                    Mf::Exp2 => "exp2",
                    Mf::Log => "log",
                    Mf::Log2 => "log2",
                    Mf::Pow => "pow",
                    // geometry
                    Mf::Dot => match *ctx.resolve_type(arg, &self.module.types) {
                        crate::TypeInner::Vector {
                            scalar:
                                crate::Scalar {
                                    kind: crate::ScalarKind::Float,
                                    ..
                                },
                            ..
                        } => "dot",
                        crate::TypeInner::Vector { size, .. } => {
                            return self.write_dot_product(arg, arg1.unwrap(), size as usize, ctx)
                        }
                        _ => unreachable!(
                            "Correct TypeInner for dot product should be already validated"
                        ),
                    },
                    Mf::Outer => "outerProduct",
                    Mf::Cross => "cross",
                    Mf::Distance => "distance",
                    Mf::Length => "length",
                    Mf::Normalize => "normalize",
                    Mf::FaceForward => "faceforward",
                    Mf::Reflect => "reflect",
                    Mf::Refract => "refract",
                    // computational
                    Mf::Sign => "sign",
                    Mf::Fma => {
                        if self.options.version.supports_fma_function() {
                            // Use the fma function when available
                            "fma"
                        } else {
                            // No fma support. Transform the function call into an arithmetic expression
                            write!(self.out, "(")?;

                            self.write_compute_expr(arg, ctx)?;
                            write!(self.out, " * ")?;

                            let arg1 =
                                arg1.ok_or_else(|| Error::Custom("Missing fma arg1".to_owned()))?;
                            self.write_compute_expr(arg1, ctx)?;
                            write!(self.out, " + ")?;

                            let arg2 =
                                arg2.ok_or_else(|| Error::Custom("Missing fma arg2".to_owned()))?;
                            self.write_compute_expr(arg2, ctx)?;
                            write!(self.out, ")")?;

                            return Ok(());
                        }
                    }
                    Mf::Mix => "mix",
                    Mf::Step => "step",
                    Mf::SmoothStep => "smoothstep",
                    Mf::Sqrt => "sqrt",
                    Mf::InverseSqrt => "inversesqrt",
                    Mf::Inverse => "inverse",
                    Mf::Transpose => "transpose",
                    Mf::Determinant => "determinant",
                    // bits
                    Mf::CountTrailingZeros => {
                        match *ctx.resolve_type(arg, &self.module.types) {
                            crate::TypeInner::Vector { size, scalar, .. } => {
                                let s = back::vector_size_str(size);
                                if let crate::ScalarKind::Uint = scalar.kind {
                                    write!(self.out, "min(uvec{s}(findLSB(")?;
                                    self.write_compute_expr(arg, ctx)?;
                                    write!(self.out, ")), uvec{s}(32u))")?;
                                } else {
                                    write!(self.out, "ivec{s}(min(uvec{s}(findLSB(")?;
                                    self.write_compute_expr(arg, ctx)?;
                                    write!(self.out, ")), uvec{s}(32u)))")?;
                                }
                            }
                            crate::TypeInner::Scalar(scalar) => {
                                if let crate::ScalarKind::Uint = scalar.kind {
                                    write!(self.out, "min(uint(findLSB(")?;
                                    self.write_compute_expr(arg, ctx)?;
                                    write!(self.out, ")), 32u)")?;
                                } else {
                                    write!(self.out, "int(min(uint(findLSB(")?;
                                    self.write_compute_expr(arg, ctx)?;
                                    write!(self.out, ")), 32u))")?;
                                }
                            }
                            _ => unreachable!(),
                        };
                        return Ok(());
                    }
                    Mf::CountLeadingZeros => {
                        if self.options.version.supports_integer_functions() {
                            match *ctx.resolve_type(arg, &self.module.types) {
                                crate::TypeInner::Vector { size, scalar } => {
                                    let s = back::vector_size_str(size);

                                    if let crate::ScalarKind::Uint = scalar.kind {
                                        write!(self.out, "uvec{s}(ivec{s}(31) - findMSB(")?;
                                        self.write_compute_expr(arg, ctx)?;
                                        write!(self.out, "))")?;
                                    } else {
                                        write!(self.out, "mix(ivec{s}(31) - findMSB(")?;
                                        self.write_compute_expr(arg, ctx)?;
                                        write!(self.out, "), ivec{s}(0), lessThan(")?;
                                        self.write_compute_expr(arg, ctx)?;
                                        write!(self.out, ", ivec{s}(0)))")?;
                                    }
                                }
                                crate::TypeInner::Scalar(scalar) => {
                                    if let crate::ScalarKind::Uint = scalar.kind {
                                        write!(self.out, "uint(31 - findMSB(")?;
                                    } else {
                                        write!(self.out, "(")?;
                                        self.write_compute_expr(arg, ctx)?;
                                        write!(self.out, " < 0 ? 0 : 31 - findMSB(")?;
                                    }

                                    self.write_compute_expr(arg, ctx)?;
                                    write!(self.out, "))")?;
                                }
                                _ => unreachable!(),
                            };
                        } else {
                            match *ctx.resolve_type(arg, &self.module.types) {
                                crate::TypeInner::Vector { size, scalar } => {
                                    let s = back::vector_size_str(size);

                                    if let crate::ScalarKind::Uint = scalar.kind {
                                        write!(self.out, "uvec{s}(")?;
                                        write!(self.out, "vec{s}(31.0) - floor(log2(vec{s}(")?;
                                        self.write_compute_expr(arg, ctx)?;
                                        write!(self.out, ") + 0.5)))")?;
                                    } else {
                                        write!(self.out, "ivec{s}(")?;
                                        write!(self.out, "mix(vec{s}(31.0) - floor(log2(vec{s}(")?;
                                        self.write_compute_expr(arg, ctx)?;
                                        write!(self.out, ") + 0.5)), ")?;
                                        write!(self.out, "vec{s}(0.0), lessThan(")?;
                                        self.write_compute_expr(arg, ctx)?;
                                        write!(self.out, ", ivec{s}(0u))))")?;
                                    }
                                }
                                crate::TypeInner::Scalar(scalar) => {
                                    if let crate::ScalarKind::Uint = scalar.kind {
                                        write!(self.out, "uint(31.0 - floor(log2(float(")?;
                                        self.write_compute_expr(arg, ctx)?;
                                        write!(self.out, ") + 0.5)))")?;
                                    } else {
                                        write!(self.out, "(")?;
                                        self.write_compute_expr(arg, ctx)?;
                                        write!(self.out, " < 0 ? 0 : int(")?;
                                        write!(self.out, "31.0 - floor(log2(float(")?;
                                        self.write_compute_expr(arg, ctx)?;
                                        write!(self.out, ") + 0.5))))")?;
                                    }
                                }
                                _ => unreachable!(),
                            };
                        }

                        return Ok(());
                    }
                    Mf::CountOneBits => "bitCount",
                    Mf::ReverseBits => "bitfieldReverse",
                    Mf::ExtractBits => {
                        // The behavior of ExtractBits is undefined when offset + count > bit_width. We need
                        // to first sanitize the offset and count first. If we don't do this, AMD and Intel chips
                        // will return out-of-spec values if the extracted range is not within the bit width.
                        //
                        // This encodes the exact formula specified by the wgsl spec, without temporary values:
                        // https://gpuweb.github.io/gpuweb/wgsl/#extractBits-unsigned-builtin
                        //
                        // w = sizeof(x) * 8
                        // o = min(offset, w)
                        // c = min(count, w - o)
                        //
                        // bitfieldExtract(x, o, c)
                        //
                        // extract_bits(e, min(offset, w), min(count, w - min(offset, w))))
                        let scalar_bits = ctx
                            .resolve_type(arg, &self.module.types)
                            .scalar_width()
                            .unwrap()
                            * 8;

                        write!(self.out, "bitfieldExtract(")?;
                        self.write_compute_expr(arg, ctx)?;
                        write!(self.out, ", int(min(")?;
                        self.write_compute_expr(arg1.unwrap(), ctx)?;
                        write!(self.out, ", {scalar_bits}u)), int(min(",)?;
                        self.write_compute_expr(arg2.unwrap(), ctx)?;
                        write!(self.out, ", {scalar_bits}u - min(")?;
                        self.write_compute_expr(arg1.unwrap(), ctx)?;
                        write!(self.out, ", {scalar_bits}u))))")?;

                        return Ok(());
                    }
                    Mf::InsertBits => {
                        // InsertBits has the same considerations as ExtractBits above
                        let scalar_bits = ctx
                            .resolve_type(arg, &self.module.types)
                            .scalar_width()
                            .unwrap()
                            * 8;

                        write!(self.out, "bitfieldInsert(")?;
                        self.write_compute_expr(arg, ctx)?;
                        write!(self.out, ", ")?;
                        self.write_compute_expr(arg1.unwrap(), ctx)?;
                        write!(self.out, ", int(min(")?;
                        self.write_compute_expr(arg2.unwrap(), ctx)?;
                        write!(self.out, ", {scalar_bits}u)), int(min(",)?;
                        self.write_compute_expr(arg3.unwrap(), ctx)?;
                        write!(self.out, ", {scalar_bits}u - min(")?;
                        self.write_compute_expr(arg2.unwrap(), ctx)?;
                        write!(self.out, ", {scalar_bits}u))))")?;

                        return Ok(());
                    }
                    Mf::FindLsb => "findLSB",
                    Mf::FindMsb => "findMSB",
                    // data packing
                    Mf::Pack4x8snorm => "packSnorm4x8",
                    Mf::Pack4x8unorm => "packUnorm4x8",
                    Mf::Pack2x16snorm => "packSnorm2x16",
                    Mf::Pack2x16unorm => "packUnorm2x16",
                    Mf::Pack2x16float => "packHalf2x16",
                    // data unpacking
                    Mf::Unpack4x8snorm => "unpackSnorm4x8",
                    Mf::Unpack4x8unorm => "unpackUnorm4x8",
                    Mf::Unpack2x16snorm => "unpackSnorm2x16",
                    Mf::Unpack2x16unorm => "unpackUnorm2x16",
                    Mf::Unpack2x16float => "unpackHalf2x16",
                };

                let extract_bits = fun == Mf::ExtractBits;
                let insert_bits = fun == Mf::InsertBits;

                // Some GLSL functions always return signed integers (like findMSB),
                // so they need to be cast to uint if the argument is also an uint.
                let ret_might_need_int_to_uint =
                    matches!(fun, Mf::FindLsb | Mf::FindMsb | Mf::CountOneBits | Mf::Abs);

                // Some GLSL functions only accept signed integers (like abs),
                // so they need their argument cast from uint to int.
                let arg_might_need_uint_to_int = matches!(fun, Mf::Abs);

                // Check if the argument is an unsigned integer and return the vector size
                // in case it's a vector
                let maybe_uint_size = match *ctx.resolve_type(arg, &self.module.types) {
                    crate::TypeInner::Scalar(crate::Scalar {
                        kind: crate::ScalarKind::Uint,
                        ..
                    }) => Some(None),
                    crate::TypeInner::Vector {
                        scalar:
                            crate::Scalar {
                                kind: crate::ScalarKind::Uint,
                                ..
                            },
                        size,
                    } => Some(Some(size)),
                    _ => None,
                };

                // Cast to uint if the function needs it
                if ret_might_need_int_to_uint {
                    if let Some(maybe_size) = maybe_uint_size {
                        match maybe_size {
                            Some(size) => write!(self.out, "uvec{}(", size as u8)?,
                            None => write!(self.out, "uint(")?,
                        }
                    }
                }

                write!(self.out, "{fun_name}(")?;

                // Cast to int if the function needs it
                if arg_might_need_uint_to_int {
                    if let Some(maybe_size) = maybe_uint_size {
                        match maybe_size {
                            Some(size) => write!(self.out, "ivec{}(", size as u8)?,
                            None => write!(self.out, "int(")?,
                        }
                    }
                }

                self.write_compute_expr(arg, ctx)?;

                // Close the cast from uint to int
                if arg_might_need_uint_to_int && maybe_uint_size.is_some() {
                    write!(self.out, ")")?
                }

                if let Some(arg) = arg1 {
                    write!(self.out, ", ")?;
                    if extract_bits {
                        write!(self.out, "int(")?;
                        self.write_compute_expr(arg, ctx)?;
                        write!(self.out, ")")?;
                    } else {
                        self.write_compute_expr(arg, ctx)?;
                    }
                }
                if let Some(arg) = arg2 {
                    write!(self.out, ", ")?;
                    if extract_bits || insert_bits {
                        write!(self.out, "int(")?;
                        self.write_compute_expr(arg, ctx)?;
                        write!(self.out, ")")?;
                    } else {
                        self.write_compute_expr(arg, ctx)?;
                    }
                }
                if let Some(arg) = arg3 {
                    write!(self.out, ", ")?;
                    if insert_bits {
                        write!(self.out, "int(")?;
                        self.write_compute_expr(arg, ctx)?;
                        write!(self.out, ")")?;
                    } else {
                        self.write_compute_expr(arg, ctx)?;
                    }
                }
                write!(self.out, ")")?;

                // Close the cast from int to uint
                if ret_might_need_int_to_uint && maybe_uint_size.is_some() {
                    write!(self.out, ")")?
                }
            }
            // `As` is always a call.
            // If `convert` is true the function name is the type
            // Else the function name is one of the glsl provided bitcast functions
            Expression::As {
                expr,
                kind: target_kind,
                convert,
            } => {
                let inner = ctx.resolve_type(expr, &self.module.types);
                match convert {
                    Some(width) => {
                        // this is similar to `write_type`, but with the target kind
                        let scalar = glsl_scalar(crate::Scalar {
                            kind: target_kind,
                            width,
                        })?;
                        match *inner {
                            TypeInner::Matrix { columns, rows, .. } => write!(
                                self.out,
                                "{}mat{}x{}",
                                scalar.prefix, columns as u8, rows as u8
                            )?,
                            TypeInner::Vector { size, .. } => {
                                write!(self.out, "{}vec{}", scalar.prefix, size as u8)?
                            }
                            _ => write!(self.out, "{}", scalar.full)?,
                        }

                        write!(self.out, "(")?;
                        self.write_compute_expr(expr, ctx)?;
                        write!(self.out, ")")?
                    }
                    None => {
                        use crate::ScalarKind as Sk;

                        let target_vector_type = match *inner {
                            TypeInner::Vector { size, scalar } => Some(TypeInner::Vector {
                                size,
                                scalar: crate::Scalar {
                                    kind: target_kind,
                                    width: scalar.width,
                                },
                            }),
                            _ => None,
                        };

                        let source_kind = inner.scalar_kind().unwrap();

                        match (source_kind, target_kind, target_vector_type) {
                            // No conversion needed
                            (Sk::Sint, Sk::Sint, _)
                            | (Sk::Uint, Sk::Uint, _)
                            | (Sk::Float, Sk::Float, _)
                            | (Sk::Bool, Sk::Bool, _) => {
                                self.write_compute_expr(expr, ctx)?;
                                return Ok(());
                            }

                            // Cast to/from floats
                            (Sk::Float, Sk::Sint, _) => write!(self.out, "floatBitsToInt")?,
                            (Sk::Float, Sk::Uint, _) => write!(self.out, "floatBitsToUint")?,
                            (Sk::Sint, Sk::Float, _) => write!(self.out, "intBitsToFloat")?,
                            (Sk::Uint, Sk::Float, _) => write!(self.out, "uintBitsToFloat")?,

                            // Cast between vector types
                            (_, _, Some(vector)) => {
                                self.write_value_type(&vector)?;
                            }

                            // There is no way to bitcast between Uint/Sint in glsl. Use constructor conversion
                            (Sk::Uint | Sk::Bool, Sk::Sint, None) => write!(self.out, "int")?,
                            (Sk::Sint | Sk::Bool, Sk::Uint, None) => write!(self.out, "uint")?,
                            (Sk::Bool, Sk::Float, None) => write!(self.out, "float")?,
                            (Sk::Sint | Sk::Uint | Sk::Float, Sk::Bool, None) => {
                                write!(self.out, "bool")?
                            }

                            (Sk::AbstractInt | Sk::AbstractFloat, _, _)
                            | (_, Sk::AbstractInt | Sk::AbstractFloat, _) => unreachable!(),
                        };

                        write!(self.out, "(")?;
                        self.write_compute_expr(expr, ctx)?;
                        write!(self.out, ")")?;
                    }
                }
            }
            // These expressions never show up in `Emit`.
            Expression::CallResult(_)
            | Expression::AtomicResult { .. }
            | Expression::RayQueryProceedResult
            | Expression::WorkGroupUniformLoadResult { .. }
            | Expression::SubgroupOperationResult { .. }
            | Expression::SubgroupBallotResult => unreachable!(),
            // `ArrayLength` is written as `expr.length()` and we convert it to a uint
            Expression::ArrayLength(expr) => {
                write!(self.out, "uint(")?;
                self.write_compute_expr(expr, ctx)?;
                write!(self.out, "_texture_width * ")?;
                self.write_compute_expr(expr, ctx)?;
                write!(self.out, "_texture_height)")?;
            }
            // not supported yet
            Expression::RayQueryGetIntersection { .. } => unreachable!(),
        }

        Ok(())
    }
}
