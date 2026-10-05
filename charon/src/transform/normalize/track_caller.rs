//! Give trait methods a uniform `#[track_caller]` calling convention, then pass locations.
use std::collections::HashMap;
use std::collections::HashSet;

use crate::ast::*;
use crate::transform::TransformCtx;
use crate::transform::ctx::TransformPass;
use crate::ullbc_ast::{self as ullbc, *};

pub struct Transform;

fn is_track_caller_attr(meta: &ItemMeta) -> bool {
    meta.attr_info.attributes.iter().any(|attr| {
        matches!(
            attr,
            Attribute::Builtin(from_rustc::AttributeKind::TrackCaller(_))
        )
    })
}

fn promote_location(
    ctx: &mut TransformCtx,
    name: &Name,
    span: Span,
    location: ConstantExpr,
    filenames: &mut HashMap<Vec<u8>, GlobalDeclRef>,
) -> Operand {
    let location = location.replace_erased_regions(|| Region::Static);
    let ConstantExprKind::Ref(value, None) = location.kind() else {
        panic!("rustc's caller location is not a reference to a sized value")
    };
    let mut value = value.clone();
    // In value mode rustc's location contains the filename inline. Give all locations
    // for the same file a single backing allocation, including the terminating NUL
    // that `Location::file_as_c_str` can read beyond the `str` metadata length.
    #[derive(Visitor)]
    struct SeparateFilename<'a> {
        ctx: &'a mut TransformCtx,
        span: Span,
        filenames: &'a mut HashMap<Vec<u8>, GlobalDeclRef>,
    }
    impl VisitAstMut for SeparateFilename<'_> {
        fn exit_constant_expr(&mut self, expr: &mut ConstantExpr) {
            let ConstantExprKind::Ptr(_, pointee, Some(UnsizingMetadata::Length(length))) =
                expr.kind()
            else {
                return;
            };
            let ConstantExprKind::Adt(None, fields) = pointee.kind() else {
                return;
            };
            let ConstantExprKind::Array(bytes) = fields[FieldId::ZERO].kind() else {
                return;
            };
            let bytes: Vec<u8> = bytes
                .iter()
                .map(|byte| match byte.kind() {
                    ConstantExprKind::Integer(IntegerValue::Unsigned(UIntTy::U8, value)) => {
                        u8::try_from(*value).expect("invalid filename byte")
                    }
                    _ => panic!("invalid filename byte in caller location"),
                })
                .collect();
            let global = if let Some(global) = self.filenames.get(&bytes) {
                global.clone()
            } else {
                let id = self.ctx.translated.global_decls.reserve_slot();
                let mut name = Name::from_path(&[&self.ctx.translated.crate_name]);
                name.name.push(PathElem::Ident(
                    "filename".to_owned(),
                    Disambiguator::new(id.index()),
                ));
                let mut memory: Vec<Byte> = bytes.iter().copied().map(Byte::Value).collect();
                memory.push(Byte::Value(0));
                let ty = pointee.ty().clone();
                self.ctx.translated.set_new_item_slot(
                    id.into(),
                    ItemByVal::Global(GlobalDecl {
                        def_id: id,
                        item_meta: ItemMeta::dummy_public(
                            self.span,
                            name,
                            true,
                            ItemOpacity::Transparent,
                        ),
                        generics: GenericParams::empty(),
                        ty: ty.clone(),
                        size: Size::new(memory.len() as u64),
                        align: Size::new(1),
                        ptr_metadata: length.clone(),
                        src: GlobalSource::Normal,
                        global_kind: GlobalKind::Static {
                            is_mut: false,
                            is_safe: true,
                            is_thread_local: false,
                        },
                        value: ConstantExpr::new(ConstantExprKind::RawMemory(memory), ty),
                    }),
                );
                let global = GlobalDeclRef {
                    id,
                    generics: Box::new(GenericArgs::empty()),
                };
                self.filenames.insert(bytes, global.clone());
                global
            };
            expr.with_contents_mut(|kind, _| {
                let ConstantExprKind::Ptr(_, pointee, _) = kind else {
                    unreachable!()
                };
                *pointee =
                    ConstantExpr::new(ConstantExprKind::Global(global), pointee.ty().clone());
            });
        }
    }
    value.drive_mut(&mut SeparateFilename {
        ctx,
        span,
        filenames,
    });
    let ty = value.ty().clone();
    let id = ctx.translated.global_decls.reserve_slot();
    let mut name = name.clone();
    name.name.push(PathElem::Builtin(
        BuiltinPathElem::PromotedConst,
        Disambiguator::new(id.index()),
    ));
    ctx.translated.set_new_item_slot(
        id.into(),
        ItemByVal::Global(GlobalDecl {
            def_id: id,
            item_meta: ItemMeta::dummy_public(span, name, true, ItemOpacity::Transparent),
            generics: GenericParams::empty(),
            ty: ty.clone(),
            size: Size::from_expr(SizeExpr::size_of(&ty)),
            align: Size::from_expr(SizeExpr::align_of(&ty)),
            ptr_metadata: ConstantExpr::mk_unit(),
            src: GlobalSource::Normal,
            global_kind: GlobalKind::Static {
                is_mut: false,
                is_safe: true,
                is_thread_local: false,
            },
            value,
        }),
    );
    let global = ConstantExpr::new(
        ConstantExprKind::Global(GlobalDeclRef {
            id,
            generics: Box::new(GenericArgs::empty()),
        }),
        ty,
    );
    Operand::Const(ConstantExpr::new(
        ConstantExprKind::Ref(global, None),
        location.ty().clone(),
    ))
}

fn fun_id(fn_ptr: &FnPtr) -> Option<FunDeclId> {
    match fn_ptr.kind.as_ref() {
        FnPtrKind::Fun(id) => Some(*id),
        FnPtrKind::Trait(..) => None,
    }
}

fn is_dyn_method_call(trait_ref: &TraitRef) -> bool {
    let mut proof = trait_ref;
    while let TraitRefKind::ParentClause(parent, _) = &proof.kind {
        proof = parent;
    }
    matches!(proof.kind, TraitRefKind::Dyn)
}

#[derive(Visitor)]
struct FindDecayedFunctions<'a> {
    tracked: &'a HashSet<FunDeclId>,
    decayed: HashSet<FunDeclId>,
}

impl VisitAst for FindDecayedFunctions<'_> {
    fn enter_rvalue(&mut self, value: &Rvalue) {
        if let Rvalue::UnaryOp(UnOp::Cast(CastKind::FnPtr(source, _)), _) = value
            && let TyKind::FnDef(fn_ptr) = source.kind()
            && let Some(id) = fun_id(&fn_ptr.skip_binder)
            && self.tracked.contains(&id)
        {
            self.decayed.insert(id);
        }
    }

    fn enter_constant_expr(&mut self, value: &ConstantExpr) {
        if let ConstantExprKind::RawMemory(bytes) = value.kind() {
            for byte in bytes {
                if let Byte::Provenance(Provenance::Function(fn_ptr), _) = byte
                    && let Some(id) = fun_id(fn_ptr)
                    && self.tracked.contains(&id)
                {
                    self.decayed.insert(id);
                }
            }
        }
        let fn_ptr = match value.kind() {
            ConstantExprKind::FnPtr(fn_ptr) => Some(fn_ptr),
            ConstantExprKind::Cast(value, _) => match value.kind() {
                ConstantExprKind::FnDef(fn_ptr) => Some(fn_ptr),
                _ => None,
            },
            _ => None,
        };
        if let Some(id) = fn_ptr.and_then(fun_id)
            && self.tracked.contains(&id)
        {
            self.decayed.insert(id);
        }
    }
}

#[derive(Visitor)]
struct ApplyDecay<'a> {
    shims: &'a HashMap<FunDeclId, FunDeclId>,
}

impl ApplyDecay<'_> {
    fn shim_ptr(&self, fn_ptr: &FnPtr) -> Option<FnPtr> {
        let shim_id = *self.shims.get(&fun_id(fn_ptr)?)?;
        let mut fn_ptr = fn_ptr.clone();
        fn_ptr.kind = Box::new(FnPtrKind::Fun(shim_id));
        Some(fn_ptr)
    }
}

impl VisitAstMut for ApplyDecay<'_> {
    fn enter_rvalue(&mut self, value: &mut Rvalue) {
        if let Rvalue::UnaryOp(UnOp::Cast(CastKind::FnPtr(source, target)), _) = value
            && let TyKind::FnDef(fn_ptr) = source.kind()
            && let Some(shim) = self.shim_ptr(&fn_ptr.skip_binder)
        {
            let ptr = ConstantExpr::new(ConstantExprKind::FnPtr(shim), target.clone());
            *value = Rvalue::Use(Operand::Const(ptr), WithRetag::No);
        }
    }

    fn exit_constant_expr(&mut self, value: &mut ConstantExpr) {
        if matches!(value.kind(), ConstantExprKind::RawMemory(_)) {
            value.with_contents_mut(|kind, _| {
                let ConstantExprKind::RawMemory(bytes) = kind else {
                    unreachable!()
                };
                for byte in bytes {
                    if let Byte::Provenance(Provenance::Function(fn_ptr), _) = byte
                        && let Some(shim) = self.shim_ptr(fn_ptr)
                    {
                        *fn_ptr = shim;
                    }
                }
            });
        }
        let replacement = match value.kind() {
            ConstantExprKind::FnPtr(fn_ptr) => self.shim_ptr(fn_ptr),
            ConstantExprKind::Cast(inner, _) => match inner.kind() {
                ConstantExprKind::FnDef(fn_ptr) => self.shim_ptr(fn_ptr),
                _ => None,
            },
            _ => None,
        };
        if let Some(shim) = replacement {
            value.with_contents_mut(|kind, _| *kind = ConstantExprKind::FnPtr(shim));
        }
    }
}

fn make_decay_shim(
    ctx: &mut TransformCtx,
    fun_id: FunDeclId,
    filenames: &mut HashMap<Vec<u8>, GlobalDeclRef>,
) -> FunDeclId {
    let fun = ctx.translated.fun_decls[fun_id].clone();
    let span = fun.item_meta.span;
    let location = ctx
        .definition_locations
        .get(&fun_id)
        .cloned()
        .unwrap_or_else(|| {
            panic!(
                "missing definition location for decayed function {:?}",
                fun.item_meta.name
            )
        });
    let location = promote_location(ctx, &fun.item_meta.name, span, location, filenames);
    let mut signature = *fun.signature.clone();
    signature
        .inputs
        .pop()
        .expect("tracked function has a location argument");
    let mut builder = BodyBuilder::new(span, signature.inputs.len());
    let dest = builder.new_var(None, signature.output.clone());
    let mut args: Vec<_> = signature
        .inputs
        .iter()
        .map(|ty| Operand::Move(builder.new_var(None, ty.clone())))
        .collect();
    args.push(location);
    let target = FnPtr::new(FnPtrKind::Fun(fun_id), fun.generics.identity_args());
    builder.call(Call {
        func: FnOperand::Regular(target),
        args,
        dest,
        safety: CallSafety::Inherit,
    });
    let shim_id = ctx.translated.fun_decls.reserve_slot();
    let mut name = fun.item_meta.name;
    name.name.push(PathElem::Ident(
        "track_caller_shim".to_owned(),
        Disambiguator::new(shim_id.index()),
    ));
    let meta = ItemMeta::dummy_public(span, name, true, ItemOpacity::Transparent);
    ctx.translated.set_new_item_slot(
        shim_id.into(),
        ItemByVal::Fun(FunDecl {
            def_id: shim_id,
            item_meta: meta,
            generics: fun.generics,
            signature: Box::new(signature),
            src: FunSource::Normal,
            body: Body::Unstructured(builder.build()),
        }),
    );
    shim_id
}

impl TransformPass for Transform {
    fn transform_ctx(&self, ctx: &mut TransformCtx) {
        let mut tracked_methods = ctx.track_caller_methods.clone();
        for (trait_id, trait_decl) in ctx.translated.trait_decls.iter_indexed() {
            for (method_id, method) in trait_decl.methods.iter_indexed() {
                if is_track_caller_attr(&method.skip_binder.item_meta) {
                    tracked_methods.insert((trait_id, method_id));
                }
            }
        }
        for id in &ctx.track_caller_funs {
            let Some(fun) = ctx.translated.fun_decls.get(*id) else {
                continue;
            };
            match &fun.src {
                FunSource::TraitImpl {
                    trait_ref, item_id, ..
                }
                | FunSource::TraitDefault { trait_ref, item_id } => {
                    tracked_methods.insert((trait_ref.id, *item_id));
                }
                _ => {}
            }
        }

        let mut tracked_funs = ctx.track_caller_funs.clone();
        for (id, fun) in ctx.translated.fun_decls.iter_indexed() {
            match &fun.src {
                FunSource::TraitImpl {
                    trait_ref, item_id, ..
                }
                | FunSource::TraitDefault { trait_ref, item_id }
                    if tracked_methods.contains(&(trait_ref.id, *item_id)) =>
                {
                    tracked_funs.insert(id);
                }
                _ => {}
            }
        }

        let location_ty = ctx.translated.fun_decls.iter().find_map(|fun| {
            let Body::Unstructured(body) = &fun.body else {
                return None;
            };
            body.locals
                .locals
                .iter()
                .find(|local| local.name.as_deref() == Some("caller_location"))
                .map(|local| local.ty.clone().replace_erased_regions(|| Region::Static))
        });
        let Some(location_ty) = location_ty else {
            return;
        };

        for (trait_id, method_id) in &tracked_methods {
            if let Some(method) = ctx
                .translated
                .trait_decls
                .get_mut(*trait_id)
                .and_then(|decl| decl.methods.get_mut(*method_id))
            {
                method
                    .skip_binder
                    .signature
                    .inputs
                    .push(location_ty.clone());
            }
        }

        let intrinsic_ids: HashSet<_> = ctx
            .translated
            .fun_decls
            .iter_indexed()
            .filter_map(|(id, fun)| match &fun.body {
                Body::Intrinsic { name, .. } if name == "caller_location" => Some(id),
                _ => None,
            })
            .collect();

        let mut filenames = HashMap::new();
        ctx.for_each_fun_decl(|ctx, fun| {
            let is_tracked = tracked_funs.contains(&fun.def_id);
            if is_tracked {
                fun.signature.inputs.push(location_ty.clone());
            }
            let Body::Unstructured(body) = &mut fun.body else { return };
            let location_local = LocalId::new(body.locals.arg_count + 1);
            let has_location_local = body.locals.locals.get(location_local)
                .is_some_and(|local| local.name.as_deref() == Some("caller_location"));
            if is_tracked && has_location_local {
                body.locals.arg_count += 1;
            }
            for (block_id, block) in body.body.iter_mut_enumerated() {
                let ullbc::TerminatorKind::Call { call, target, .. } = &mut block.terminator.kind else {
                    continue;
                };
                let FnOperand::Regular(fn_ptr) = &call.func else { continue };
                let is_intrinsic = matches!(fn_ptr.kind.as_ref(), FnPtrKind::Fun(id) if intrinsic_ids.contains(id));
                let callee_tracked = match fn_ptr.kind.as_ref() {
                    FnPtrKind::Fun(id) => tracked_funs.contains(id),
                    FnPtrKind::Trait(tref, method_id) => {
                        tracked_methods.contains(&(tref.trait_id(), *method_id))
                            && !is_dyn_method_call(tref)
                    }
                };
                if !is_intrinsic && !callee_tracked { continue }
                let location = if is_tracked && has_location_local {
                    Operand::Copy(body.locals.place_for_var(location_local))
                } else {
                    let value = ctx.caller_locations.remove(&(fun.def_id, block_id))
                        .or_else(|| match fn_ptr.kind.as_ref() {
                            FnPtrKind::Fun(id) => ctx.definition_locations.get(id).cloned(),
                            FnPtrKind::Trait(..) => None,
                        })
                        .expect("missing rustc caller location for a MIR call");
                    promote_location(
                        ctx,
                        &fun.item_meta.name,
                        block.terminator.span,
                        value,
                        &mut filenames,
                    )
                };
                if is_intrinsic {
                    block.statements.push(ullbc::Statement::new(
                        block.terminator.span,
                        ullbc::StatementKind::Assign(call.dest.clone(), Rvalue::Use(location, WithRetag::No)),
                    ));
                    block.terminator.kind = ullbc::TerminatorKind::Goto { target: *target };
                } else {
                    call.args.push(location);
                }
            }
        });
        let mut find_decay = FindDecayedFunctions {
            tracked: &tracked_funs,
            decayed: HashSet::new(),
        };
        ctx.translated.drive(&mut find_decay);
        let shims: HashMap<_, _> = find_decay
            .decayed
            .into_iter()
            .map(|id| (id, make_decay_shim(ctx, id, &mut filenames)))
            .collect();
        ctx.translated.drive_mut(&mut ApplyDecay { shims: &shims });

        ctx.caller_locations.clear();
        ctx.definition_locations.clear();
        ctx.track_caller_funs.clear();
        ctx.track_caller_methods.clear();
        for id in intrinsic_ids {
            ctx.translated.remove_item(id.into());
        }
    }
}
