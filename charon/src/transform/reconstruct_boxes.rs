//! # Micro-pass: reconstruct piecewise box allocations using `malloc` and `ShallowInitBox`.
use derive_visitor::visitor_enter_fn;
use derive_visitor::Drive;

use crate::ids::*;
use crate::register_error_or_panic;
use crate::transform::TransformCtx;
use crate::ullbc_ast::*;

use super::ctx::UllbcPass;

pub struct Transform;

impl Transform {
    /// The special `#[rustc_box] Box::new(x)` construct becomes the following:
    ///
    /// ```text
    /// @2 := size_of<i32>
    /// @3 := align_of<i32>
    /// @4 := alloc::alloc::exchange_malloc(move (@2), move (@3))
    /// @5 := shallow_init_box::<i32>(move (@4))
    /// // possibly some intermediate statements
    /// *(@5) := x
    /// ```
    ///
    /// We reconstruct this into a call to `Box::new(x)`.
    fn update_statements(
        locals: &mut Vector<VarId, Var>,
        seq: &mut [Statement],
    ) -> Vec<(usize, Vec<Statement>)> {
        let seq_len = seq.len();
        if let [Statement {
            content: RawStatement::Assign(size, Rvalue::NullaryOp(NullOp::SizeOf, _)),
            ..
        }, Statement {
            content: RawStatement::Assign(align, Rvalue::NullaryOp(NullOp::AlignOf, _)),
            ..
        }, Statement {
            content: RawStatement::Call(call_malloc),
            ..
        }, Statement {
            content:
                RawStatement::Assign(box_make, Rvalue::ShallowInitBox(Operand::Move(alloc_use), _)),
            ..
        }, rest @ ..] = seq
        {
            let prefix_len = seq_len - rest.len();
            // TODO: once we have a system to recognize intrinsics, check the call is to exchange_malloc.
            if let [Operand::Move(arg0), Operand::Move(arg1)] = call_malloc.args.as_slice()
                && arg0 == size
                && arg1 == align
                && call_malloc.dest == *alloc_use
                && box_make.projection.is_empty()
                && let var_id = box_make.var_id
                && let Ty::Adt(TypeId::Builtin(BuiltinTy::Box), generics) = &locals[var_id].ty
            {
                // Find the assignment into the box.
                for i in 0..rest.len() {
                    if let Statement {
                        content: RawStatement::Assign(box_deref, val),
                        ..
                    } = &mut rest[i]
                        && box_deref.var_id == box_make.var_id
                        && let [ProjectionElem::Deref] = box_deref.projection.as_slice()
                    {
                        let real_i = prefix_len + i;
                        let mut to_insert = Vec::new();
                        let dest = box_make.clone();
                        let val = val.clone();
                        let generics = generics.clone();
                        seq[0].content = RawStatement::Nop;
                        seq[1].content = RawStatement::Nop;
                        seq[2].content = RawStatement::Nop;
                        seq[3].content = RawStatement::Nop;
                        let val = match val {
                            Rvalue::Use(op) => op,
                            _ => {
                                // We need to create a new variable to store the value.
                                let name = locals[var_id].name.clone();
                                let ty = generics.types[0].clone();
                                let var = locals.push_with(|index| Var { index, name, ty });
                                let st = Statement {
                                    span: seq[real_i].span,
                                    content: RawStatement::Assign(Place::new(var), val),
                                };
                                to_insert.push((real_i, vec![st]));
                                Operand::Move(Place::new(var))
                            }
                        };
                        seq[real_i].content = RawStatement::Call(Call {
                            func: FnOperand::Regular(FnPtr {
                                func: FunIdOrTraitMethodRef::Fun(FunId::Builtin(
                                    BuiltinFunId::BoxNew,
                                )),
                                generics,
                            }),
                            args: vec![val],
                            dest,
                        });
                        return to_insert;
                    }
                }
            }
        }
        Vec::new()
    }
}

impl UllbcPass for Transform {
    fn transform_body(&self, ctx: &mut TransformCtx<'_>, b: &mut ExprBody) {
        for block in &mut b.body {
            block.transform_sequences(&mut |seq| Transform::update_statements(&mut b.locals, seq));
        }

        // Make sure we got all the `ShallowInitBox`es.
        b.body.drive(&mut visitor_enter_fn(|rvalue: &Rvalue| {
            if rvalue.is_shallow_init_box() {
                register_error_or_panic!(
                    ctx,
                    b.span.span.rust_span_data.span(),
                    "Unexpected `ShallowInitBox`"
                );
            }
        }));
    }
}