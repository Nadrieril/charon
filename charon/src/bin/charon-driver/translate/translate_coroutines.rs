//! Rust async functions and blocks lower to coroutine state machines that implement `Future`.
//!
//! We translate the generated state machine as an ADT and synthesize the matching `Future` impl
//! and `poll` method. This intentionally mirrors the closure translation machinery.

use crate::hax;
use crate::hax::UnderOwnerState;
use itertools::Itertools;
use rustc_middle::ty::CoroutineArgsExt;
use rustc_middle::{mir, ty};

use super::translate_crate::{TraitImplSource, TransItemSourceKind};
use super::translate_ctx::*;
use charon_lib::ast::*;
use charon_lib::ids::IndexVec;

impl<'tcx> ItemTransCtx<'tcx, '_> {
    /// Translate a reference to the coroutine ADT.
    pub fn translate_coroutine_type_ref(
        &mut self,
        span: Span,
        coroutine: &hax::CoroutineArgs,
    ) -> Result<TypeDeclRef, Error> {
        if coroutine.is_async {
            let _: TraitImplRef = self.translate_coroutine_future_impl_ref(span, coroutine)?;
        }
        self.translate_item(span, &coroutine.item, TransItemSourceKind::Type)
    }

    /// Translate a reference to the generated `Future` impl for this coroutine.
    pub fn translate_coroutine_future_impl_ref(
        &mut self,
        span: Span,
        coroutine: &hax::CoroutineArgs,
    ) -> Result<TraitImplRef, Error> {
        self.translate_item(
            span,
            &coroutine.item,
            TransItemSourceKind::TraitImpl(TraitImplSource::CoroutineFuture),
        )
    }

    /// Translate the types of the captured variables. Should be called only in
    /// `translate_item_generics`.
    pub fn translate_coroutine_upvar_tys(
        &mut self,
        span: Span,
        args: &hax::CoroutineArgs,
    ) -> Result<IndexVec<FieldId, Ty>, Error> {
        args.upvar_tys
            .iter()
            .map(|ty| self.translate_ty(span, ty))
            .try_collect()
    }

    fn rustc_coroutine_args(&self, args: &hax::CoroutineArgs) -> ty::GenericArgsRef<'tcx> {
        let hax_state = self.hax_state_with_id();
        let rust_def_id = args.item.def_id.real_rust_def_id();
        let item_args = args.item.rustc_args(hax_state);
        let Some(parent_def_id) = self.tcx.opt_parent(rust_def_id) else {
            return item_args;
        };
        if !matches!(
            self.tcx.def_kind(parent_def_id),
            hax::RDefKind::Fn | hax::RDefKind::AssocFn
        ) {
            return item_args;
        }

        let sig = hax::inst_binder(
            self.tcx,
            hax_state.typing_env(),
            Some(item_args),
            self.tcx.fn_sig(parent_def_id),
        );
        let output = sig.skip_binder().output();
        match output.kind() {
            ty::TyKind::Coroutine(coroutine_def_id, coroutine_args)
                if *coroutine_def_id == rust_def_id =>
            {
                coroutine_args
            }
            _ => item_args,
        }
    }

    pub fn translate_coroutine_adt(
        &mut self,
        span: Span,
        args: &hax::CoroutineArgs,
    ) -> Result<TypeDeclKind, Error> {
        let mut variants: IndexVec<VariantId, Variant> = IndexVec::new();
        let upvar_tys = self
            .the_only_binder()
            .closure_upvar_tys
            .as_ref()
            .unwrap()
            .iter()
            .cloned()
            .collect_vec();
        variants.push(self.mk_coroutine_variant(span, VariantId::ZERO, "Unresumed", upvar_tys)?);
        variants.push(self.mk_coroutine_variant(
            span,
            VariantId::new(1),
            "Returned",
            std::iter::empty(),
        )?);
        variants.push(self.mk_coroutine_variant(
            span,
            VariantId::new(2),
            "Panicked",
            std::iter::empty(),
        )?);

        let rust_def_id = args.item.def_id.real_rust_def_id();
        let rust_args = self.rustc_coroutine_args(args);
        if let Ok(layout) = self.tcx.coroutine_layout(rust_def_id, rust_args) {
            for (variant_idx, fields) in layout.variant_fields.iter_enumerated().skip(3) {
                let tys: Vec<Ty> = fields
                    .iter()
                    .map(|field| {
                        let ty = layout.field_tys[*field].ty;
                        let ty = ty::EarlyBinder::bind(ty)
                            .instantiate(self.tcx, rust_args)
                            .skip_norm_wip();
                        self.translate_rustc_ty(span, &ty)
                    })
                    .try_collect()?;
                let name = ty::CoroutineArgs::variant_name(variant_idx).to_string();
                variants.push(self.mk_coroutine_variant(
                    span,
                    self.translate_variant_id(variant_idx),
                    name,
                    tys,
                )?);
            }
        }
        Ok(TypeDeclKind::Enum(variants))
    }

    fn mk_coroutine_variant(
        &mut self,
        span: Span,
        id: VariantId,
        name: impl Into<String>,
        tys: impl IntoIterator<Item = Ty>,
    ) -> Result<Variant, Error> {
        let fields = tys
            .into_iter()
            .map(|ty| Field {
                span,
                attr_info: AttrInfo::dummy_private(),
                name: None,
                ty,
            })
            .collect();
        Ok(Variant {
            id,
            span,
            attr_info: AttrInfo::dummy_private(),
            name: name.into(),
            fields,
            discriminant: Literal::Scalar(
                ScalarValue::from_uint(
                    self.translated.the_target_information().target_pointer_size,
                    UIntTy::U32,
                    id.index() as u128,
                )
                .unwrap(),
            ),
        })
    }

    fn translate_coroutine_poll_sig(
        &mut self,
        span: Span,
        def: &hax::FullDef<'tcx>,
    ) -> Result<FunSig, Error> {
        let Some(body) = self.get_mir(def.this(), span)? else {
            raise_error!(self, span, "missing MIR body for coroutine poll")
        };
        let output = self.translate_rustc_ty(span, &body.local_decls[mir::Local::ZERO].ty)?;
        let inputs = (1..=body.arg_count)
            .map(|i| {
                let local = mir::Local::from_usize(i);
                self.translate_rustc_ty(span, &body.local_decls[local].ty)
            })
            .try_collect()?;
        Ok(FunSig {
            inputs,
            output,
            is_unsafe: false,
            abi: Abi::rust(),
            is_variadic: false,
        })
    }

    #[tracing::instrument(skip(self, item_meta))]
    pub fn translate_coroutine_poll_method(
        mut self,
        def_id: FunDeclId,
        item_meta: ItemMeta,
        def: &hax::FullDef<'tcx>,
    ) -> Result<FunDecl, Error> {
        let span = item_meta.span;
        let hax::FullDefKind::Coroutine {
            args, future_impl, ..
        } = def.kind()
        else {
            unreachable!()
        };
        let future_impl = future_impl
            .as_ref()
            .ok_or_else(|| register_error!(self, span, "non-async coroutines are not supported"))?;

        let implemented_trait = self.translate_trait_predicate(span, &future_impl.trait_pred)?;
        let method_id =
            self.translate_trait_method_id(implemented_trait.id, &future_impl.methods[0])?;
        let impl_ref = self.translate_coroutine_future_impl_ref(span, args)?;
        let src = ItemSource::TraitImpl {
            impl_ref,
            trait_ref: implemented_trait,
            item_id: method_id.into(),
            reuses_default: false,
        };

        let signature = self.translate_coroutine_poll_sig(span, def)?;
        let body = if item_meta.opacity.with_private_contents().is_opaque() {
            Body::Opaque
        } else {
            self.translate_def_body(span, def)
        };

        Ok(FunDecl {
            def_id,
            item_meta,
            generics: self.into_generics(),
            signature: Box::new(signature),
            src,
            is_global_initializer: None,
            body,
        })
    }

    #[tracing::instrument(skip(self, item_meta))]
    pub fn translate_coroutine_future_trait_impl(
        mut self,
        def_id: TraitImplId,
        item_meta: ItemMeta,
        def: &hax::FullDef<'tcx>,
    ) -> Result<TraitImpl, Error> {
        let span = item_meta.span;
        let hax::FullDefKind::Coroutine {
            args, future_impl, ..
        } = def.kind()
        else {
            unreachable!()
        };
        let future_impl = future_impl
            .as_ref()
            .ok_or_else(|| register_error!(self, span, "non-async coroutines are not supported"))?;
        let mut timpl = self.translate_virtual_trait_impl(def_id, item_meta, future_impl)?;

        if self.monomorphize() {
            return Ok(timpl);
        }

        let trait_decl_id = timpl.impl_trait.id;
        let trait_method_id =
            self.translate_trait_method_id(trait_decl_id, &future_impl.methods[0])?;
        let fn_decl_ref: FunDeclRef =
            self.translate_item(span, &args.item, TransItemSourceKind::CoroutinePollMethod)?;
        let fn_decl_ref = fn_decl_ref.move_under_binder();
        let call_fn_binder = Binder::new(
            BinderKind::TraitMethod(trait_decl_id, trait_method_id),
            GenericParams::empty(),
            fn_decl_ref,
        );
        timpl
            .methods
            .set_slot_extend(trait_method_id, call_fn_binder);

        Ok(timpl)
    }
}
