//! Functions synthesized to implement intrinsics and start the program.

use super::*;
use itertools::Itertools;

struct FunctionBuilder {
    locals: Vec<mini::Type>,
    arg_count: usize,
    blocks: Vec<Option<mini::BasicBlock>>,
    calling_convention: mini::CallingConvention,
}

impl FunctionBuilder {
    fn new<T: mini::Target>(
        ctx: &TranslateCtx<'_, T>,
        span: Span,
        signature: &FunSig,
    ) -> Result<Self> {
        let locals = std::iter::once(&signature.output)
            .chain(&signature.inputs)
            .map(|ty| ctx.ty(span, ty))
            .try_collect()?;
        Ok(Self {
            locals,
            arg_count: signature.inputs.len(),
            blocks: Vec::new(),
            calling_convention: ctx.calling_convention(span, &signature.abi)?,
        })
    }

    fn return_local(&self) -> mini::LocalName {
        mini::LocalName(name(0))
    }

    fn argument(&self, index: usize) -> mini::LocalName {
        assert!(index < self.arg_count);
        mini::LocalName(name((index + 1) as u32))
    }

    fn add_local(&mut self, ty: mini::Type) -> mini::LocalName {
        let name = mini::LocalName(name(self.locals.len() as u32));
        self.locals.push(ty);
        name
    }

    fn declare_block(&mut self) -> mini::BbName {
        let name = mini::BbName(name(self.blocks.len() as u32));
        self.blocks.push(None);
        name
    }

    fn set_block(&mut self, name: mini::BbName, block: mini::BasicBlock) {
        assert!(
            self.blocks[name.0.get_internal() as usize]
                .replace(block)
                .is_none(),
            "MiniRust block was defined twice"
        );
    }

    fn finish(self) -> mini::Function {
        let blocks = self
            .blocks
            .into_iter()
            .map(|block| block.expect("MiniRust block was declared but never defined"))
            .collect_vec();
        let mut function = mb::function(mb::Ret::Yes, self.arg_count, &self.locals, &blocks);
        function.calling_convention = self.calling_convention;
        function
    }
}

pub(super) enum UnwindSource {
    ExplicitPayload,
    OpaquePayload,
}

impl<T: mini::Target> TranslateCtx<'_, T> {
    /// Make a start function with the right calling convention that just calls into `main()`.
    pub(super) fn make_start_function(
        &self,
        span: Span,
        main: FunDeclId,
    ) -> Result<mini::Function> {
        let mut signature = self.krate.fun_decls[main].signature.clone();
        signature.abi = Abi::C;
        check!(
            span,
            signature.inputs.is_empty() && signature.output.is_unit(),
            "MiniRust output only supports an entry point with signature `fn main()`"
        );
        let mut builder = FunctionBuilder::new(self, span, &signature)?;
        let ret = builder.return_local();
        let start = builder.declare_block();
        let exit = builder.declare_block();
        let abort = builder.declare_block();
        builder.set_block(
            start,
            mb::block(
                &[],
                mini::Terminator::Call {
                    callee: self.fn_pointer(main),
                    calling_convention: mini::CallingConvention::Rust,
                    arguments: Default::default(),
                    ret: mini::PlaceExpr::Local(ret),
                    next_block: Some(exit),
                    unwind_block: Some(abort),
                },
                mini::BbKind::Regular,
            ),
        );
        builder.set_block(
            exit,
            mb::block(
                &[],
                mini::Terminator::Intrinsic {
                    intrinsic: mini::IntrinsicOp::Exit,
                    arguments: Default::default(),
                    ret: mini::PlaceExpr::Local(ret),
                    next_block: None,
                },
                mini::BbKind::Regular,
            ),
        );
        builder.set_block(
            abort,
            mb::block(
                &[],
                mini::Terminator::Intrinsic {
                    intrinsic: mini::IntrinsicOp::Abort,
                    arguments: Default::default(),
                    ret: mini::PlaceExpr::Local(ret),
                    next_block: None,
                },
                mini::BbKind::Catch,
            ),
        );
        Ok(builder.finish())
    }

    /// Make a function that invokes the matching MiniRust intrinsic.
    pub(super) fn make_intrinsic_function(
        &self,
        span: Span,
        fdecl: &FunDecl,
        intrinsic: mini::IntrinsicOp,
    ) -> Result<mini::Function> {
        let signature = &fdecl.signature;
        let mut builder = FunctionBuilder::new(self, span, signature)?;
        let ret = builder.return_local();
        let arguments: Vec<_> = (0..signature.inputs.len())
            .map(|index| builder.argument(index))
            .collect();
        let start = builder.declare_block();
        let return_block = builder.declare_block();
        builder.set_block(
            start,
            mb::block(
                &[],
                mini::Terminator::Intrinsic {
                    intrinsic,
                    arguments: arguments
                        .iter()
                        .map(|argument| mb::load(mini::PlaceExpr::Local(*argument)))
                        .collect(),
                    ret: mini::PlaceExpr::Local(ret),
                    next_block: Some(return_block),
                },
                mini::BbKind::Regular,
            ),
        );
        builder.set_block(
            return_block,
            mb::block(&[], mini::Terminator::Return, mini::BbKind::Regular),
        );
        Ok(builder.finish())
    }

    /// Implement `core::intrinsics::catch_unwind` in MiniRust.
    pub(super) fn make_catch_unwind_function(
        &self,
        span: Span,
        fdecl: &FunDecl,
    ) -> Result<mini::Function> {
        let signature = &fdecl.signature;
        check!(
            span,
            signature.inputs.len() == 3
                && matches!(signature.inputs[0].kind(), TyKind::FnPtr(_))
                && matches!(signature.inputs[1].kind(), TyKind::RawPtr(..))
                && matches!(signature.inputs[2].kind(), TyKind::FnPtr(_))
                && signature.output.is_bool(),
            "unexpected signature for `core::intrinsics::catch_unwind`"
        );

        let mut builder = FunctionBuilder::new(self, span, signature)?;
        let ret = builder.return_local();
        let try_fn = builder.argument(0);
        let data = builder.argument(1);
        let catch_fn = builder.argument(2);
        let call_ret = builder.add_local(mini::unit_ty());
        let payload_ty = mini::Type::Ptr(mini::PtrType::Raw {
            meta_kind: mini::PointerMetaKind::None,
        });
        let payload = builder.add_local(payload_ty);

        let start = builder.declare_block();
        let returned = builder.declare_block();
        let get_payload = builder.declare_block();
        let call_catch = builder.declare_block();
        let stop_unwind = builder.declare_block();
        let caught = builder.declare_block();
        let load = |local| mb::load(mini::PlaceExpr::Local(local));
        builder.set_block(
            start,
            mb::block(
                &[
                    mini::Statement::StorageLive(call_ret),
                    mini::Statement::StorageLive(payload),
                ],
                mini::Terminator::Call {
                    callee: load(try_fn),
                    calling_convention: mini::CallingConvention::Rust,
                    arguments: [mini::ArgumentExpr::ByValue(load(data))]
                        .into_iter()
                        .collect(),
                    ret: mini::PlaceExpr::Local(call_ret),
                    next_block: Some(returned),
                    unwind_block: Some(get_payload),
                },
                mini::BbKind::Regular,
            ),
        );
        builder.set_block(
            returned,
            mb::block(
                &[mb::assign(
                    mini::PlaceExpr::Local(ret),
                    mb::const_bool(false),
                )],
                mini::Terminator::Return,
                mini::BbKind::Regular,
            ),
        );
        builder.set_block(
            get_payload,
            mb::block(
                &[],
                mini::Terminator::Intrinsic {
                    intrinsic: mini::IntrinsicOp::GetUnwindPayload,
                    arguments: Default::default(),
                    ret: mini::PlaceExpr::Local(payload),
                    next_block: Some(call_catch),
                },
                mini::BbKind::Catch,
            ),
        );
        builder.set_block(
            call_catch,
            mb::block(
                &[],
                mini::Terminator::Call {
                    callee: load(catch_fn),
                    calling_convention: mini::CallingConvention::Rust,
                    arguments: [
                        mini::ArgumentExpr::ByValue(load(data)),
                        mini::ArgumentExpr::ByValue(load(payload)),
                    ]
                    .into_iter()
                    .collect(),
                    ret: mini::PlaceExpr::Local(call_ret),
                    next_block: Some(stop_unwind),
                    // The intrinsic contract requires this function not to unwind.
                    unwind_block: None,
                },
                mini::BbKind::Catch,
            ),
        );
        builder.set_block(
            stop_unwind,
            mb::block(
                &[],
                mini::Terminator::StopUnwind(caught),
                mini::BbKind::Catch,
            ),
        );
        builder.set_block(
            caught,
            mb::block(
                &[mb::assign(
                    mini::PlaceExpr::Local(ret),
                    mb::const_bool(true),
                )],
                mini::Terminator::Return,
                mini::BbKind::Regular,
            ),
        );
        Ok(builder.finish())
    }

    /// Function that starts unwinding.
    pub(super) fn make_unwind_function(
        &self,
        span: Span,
        fdecl: &FunDecl,
        source: UnwindSource,
    ) -> Result<mini::Function> {
        let mut builder = FunctionBuilder::new(self, span, &fdecl.signature)?;
        let arg = builder.argument(0);
        let start = builder.declare_block();
        let unwind = builder.declare_block();
        builder.set_block(
            start,
            mb::block(
                &[],
                mini::Terminator::StartUnwind {
                    unwind_payload: match source {
                        UnwindSource::ExplicitPayload => mb::load(mini::PlaceExpr::Local(arg)),
                        UnwindSource::OpaquePayload => self.opaque_panic_payload(),
                    },
                    unwind_block: unwind,
                },
                mini::BbKind::Regular,
            ),
        );
        builder.set_block(
            unwind,
            mb::block(&[], mini::Terminator::ResumeUnwind, mini::BbKind::Cleanup),
        );
        Ok(builder.finish())
    }

    /// Compute a DST's layout from the metadata carried by its pointer.
    pub(super) fn make_layout_of_val_function(
        &self,
        span: Span,
        fdecl: &FunDecl,
        alignment: bool,
    ) -> Result<mini::Function> {
        let signature = &fdecl.signature;
        let [input] = signature.inputs.as_slice() else {
            raise!(span, "unexpected signature for layout-of-value intrinsic")
        };
        let pointee = self.ty(span, input.builtin_deref(self.krate).unwrap())?;

        let mut builder = FunctionBuilder::new(self, span, signature)?;
        let ret = mini::PlaceExpr::Local(builder.return_local());
        let arg = mini::PlaceExpr::Local(builder.argument(0));
        let start = builder.declare_block();
        let metadata = mb::get_metadata(mb::load(arg));
        let value = if alignment {
            mb::compute_align(pointee, metadata)
        } else {
            mb::compute_size(pointee, metadata)
        };
        builder.set_block(
            start,
            mb::block(
                &[mb::assign(ret, value)],
                mini::Terminator::Return,
                mini::BbKind::Regular,
            ),
        );
        Ok(builder.finish())
    }
}
