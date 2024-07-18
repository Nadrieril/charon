//! This file groups everything which is linked to implementations about [crate::types]
use derive_visitor::{DriveMut, VisitorMut};

use crate::ids::Vector;
use crate::types::*;
use std::fmt::Debug;
use std::iter::Iterator;

impl DeBruijnId {
    pub fn zero() -> Self {
        DeBruijnId { index: 0 }
    }

    pub fn new(index: usize) -> Self {
        DeBruijnId { index }
    }

    pub fn is_zero(&self) -> bool {
        self.index == 0
    }

    pub fn incr(&self) -> Self {
        DeBruijnId {
            index: self.index + 1,
        }
    }

    pub fn decr(&self) -> Self {
        DeBruijnId {
            index: self.index - 1,
        }
    }
}

impl TypeVar {
    pub fn new(index: TypeVarId, name: String) -> TypeVar {
        TypeVar { index, name }
    }
}

impl GenericParams {
    pub fn len(&self) -> usize {
        let GenericParams {
            regions,
            types,
            const_generics,
            trait_clauses,
            regions_outlive,
            types_outlive,
            trait_type_constraints,
        } = self;
        regions.len()
            + types.len()
            + const_generics.len()
            + trait_clauses.len()
            + regions_outlive.len()
            + types_outlive.len()
            + trait_type_constraints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn empty() -> Self {
        Self::default()
    }

    /// Construct a set of generic arguments in the scope of `self` that matches `self` and feeds
    /// each required parameter with itself. E.g. given parameters for `<T, U> wiere U:
    /// PartialEq<T>`, the arguments would be `<T, U>[@TraitClause0]`.
    pub fn identity_args(&self) -> GenericArgs {
        GenericArgs {
            regions: self
                .regions
                .iter_indexed()
                .map(|(id, _)| Region::BVar(DeBruijnId::new(0), id))
                .collect(),
            types: self
                .types
                .iter_indexed()
                .map(|(id, _)| Ty::TypeVar(id))
                .collect(),
            const_generics: self
                .const_generics
                .iter_indexed()
                .map(|(id, _)| ConstGeneric::Var(id))
                .collect(),
            trait_refs: self
                .trait_clauses
                .iter_indexed()
                .map(|(id, clause)| TraitRef {
                    kind: TraitRefKind::Clause(id),
                    trait_decl_ref: clause.trait_.clone(),
                })
                .collect(),
        }
    }
}

impl GenericArgs {
    pub fn len(&self) -> usize {
        let GenericArgs {
            regions,
            types,
            const_generics,
            trait_refs,
        } = self;
        regions.len() + types.len() + const_generics.len() + trait_refs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn empty() -> Self {
        GenericArgs {
            regions: Vec::new(),
            types: Vec::new(),
            const_generics: Vec::new(),
            trait_refs: Vec::new(),
        }
    }

    pub fn new_from_types(types: Vec<Ty>) -> Self {
        GenericArgs {
            regions: Vec::new(),
            types,
            const_generics: Vec::new(),
            trait_refs: Vec::new(),
        }
    }

    pub fn new(
        regions: Vec<Region>,
        types: Vec<Ty>,
        const_generics: Vec<ConstGeneric>,
        trait_refs: Vec<TraitRef>,
    ) -> Self {
        GenericArgs {
            regions,
            types,
            const_generics,
            trait_refs,
        }
    }

    /// Check whether this matches the given `GenericParams`.
    /// TODO: check more things, e.g. that the trait refs use the correct trait and generics.
    pub fn matches(&self, params: &GenericParams) -> bool {
        params.regions.len() == self.regions.len()
            && params.types.len() == self.types.len()
            && params.const_generics.len() == self.const_generics.len()
            && params.trait_clauses.len() == self.trait_refs.len()
    }

    /// Return the same generics, but where we pop the first type arguments.
    /// This is useful for trait references (for pretty printing for instance),
    /// because the first type argument is the type for which the trait is
    /// implemented.
    pub fn pop_first_type_arg(&self) -> (Ty, Self) {
        let GenericArgs {
            regions,
            types,
            const_generics,
            trait_refs,
        } = self;
        let mut it = types.iter();
        let ty = it.next().unwrap().clone();
        let types = it.cloned().collect();
        (
            ty,
            GenericArgs {
                regions: regions.clone(),
                types,
                const_generics: const_generics.clone(),
                trait_refs: trait_refs.clone(),
            },
        )
    }
}

impl TypeDecl {
    /// The variant id should be `None` if it is a structure and `Some` if it
    /// is an enumeration.
    #[allow(clippy::result_unit_err)]
    pub fn get_fields(&self, variant_id: Option<VariantId>) -> Result<&Vector<FieldId, Field>, ()> {
        match &self.kind {
            TypeDeclKind::Enum(variants) => Ok(&variants.get(variant_id.unwrap()).unwrap().fields),
            TypeDeclKind::Struct(fields) => {
                assert!(variant_id.is_none());
                Ok(fields)
            }
            TypeDeclKind::Alias(..) | TypeDeclKind::Opaque => {
                unreachable!("Opaque or alias type")
            }
            TypeDeclKind::Error(_) => Err(()),
        }
    }
}

impl IntegerTy {
    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            IntegerTy::Isize
                | IntegerTy::I8
                | IntegerTy::I16
                | IntegerTy::I32
                | IntegerTy::I64
                | IntegerTy::I128
        )
    }

    pub fn is_unsigned(&self) -> bool {
        !(self.is_signed())
    }

    /// Return the size (in bytes) of an integer of the proper type
    pub fn size(&self) -> usize {
        use std::mem::size_of;
        match self {
            IntegerTy::Isize => size_of::<isize>(),
            IntegerTy::I8 => size_of::<i8>(),
            IntegerTy::I16 => size_of::<i16>(),
            IntegerTy::I32 => size_of::<i32>(),
            IntegerTy::I64 => size_of::<i64>(),
            IntegerTy::I128 => size_of::<i128>(),
            IntegerTy::Usize => size_of::<isize>(),
            IntegerTy::U8 => size_of::<u8>(),
            IntegerTy::U16 => size_of::<u16>(),
            IntegerTy::U32 => size_of::<u32>(),
            IntegerTy::U64 => size_of::<u64>(),
            IntegerTy::U128 => size_of::<u128>(),
        }
    }
}

#[derive(VisitorMut)]
#[visitor(Ty, Region(exit), ConstGeneric(exit), TraitRefKind(exit))]
struct SubstVisitor<'a> {
    // The arguments to substitute.
    args: &'a GenericArgs,
    // The De Bruijn index of the binder we are substituting for. We leave other binder levels
    // untouched.
    binder_depth: DeBruijnId,
}

// TODO: subst everything
impl<'a> SubstVisitor<'a> {
    fn new(args: &'a GenericArgs) -> Self {
        Self {
            args,
            binder_depth: DeBruijnId::zero(),
        }
    }

    fn enter_ty(&mut self, ty: &mut Ty) {
        match ty {
            Ty::Arrow(_, _, _) => self.binder_depth = self.binder_depth.incr(),
            _ => {}
        }
    }
    fn exit_ty(&mut self, ty: &mut Ty) {
        match ty {
            Ty::TypeVar(v) => {
                // TODO: take into account `ImplElem` binder
                *ty = self.args.types[v.index()].clone();
            }
            Ty::Arrow(_, _, _) => self.binder_depth = self.binder_depth.decr(),
            Ty::Adt(_, _)
            | Ty::DynTrait(_)
            | Ty::Literal(_)
            | Ty::Never
            | Ty::RawPtr(_, _)
            | Ty::Ref(_, _, _)
            | Ty::TraitType(_, _) => {}
        }
    }
    fn exit_region(&mut self, region: &mut Region) {
        match region {
            Region::BVar(dbid, v) if *dbid == self.binder_depth => {
                // TODO: prpbably need to update the binders
                *region = self.args.regions[v.index()].clone();
            }
            _ => {}
        }
    }
    fn exit_const_generic(&mut self, cg: &mut ConstGeneric) {
        match cg {
            ConstGeneric::Var(v) => {
                *cg = self.args.const_generics[v.index()].clone();
            }
            _ => {}
        }
    }
    fn exit_trait_ref_kind(&mut self, tref: &mut TraitRefKind) {
        match tref {
            TraitRefKind::Clause(clause_id) => {
                *tref = self.args.trait_refs[clause_id.index()].kind.clone();
            }
            _ => {}
        }
    }
}

/// A value of type `T` bound by the generic parameters of item
/// `item`. Used when dealing with multiple items at a time, to
/// ensure we don't mix up generics.
///
/// To get the value, use `under_binder_of` or `subst_for`.
pub struct ItemBinder<ItemId, T> {
    pub item_id: ItemId,
    val: T,
}

impl<ItemId, T> ItemBinder<ItemId, T>
where
    ItemId: Debug + Copy + PartialEq,
{
    pub fn new(item_id: ItemId, val: T) -> Self {
        Self { item_id, val }
    }

    pub fn as_ref(&self) -> ItemBinder<ItemId, &T> {
        ItemBinder {
            item_id: self.item_id,
            val: &self.val,
        }
    }

    pub fn map_bound<U>(self, f: impl FnOnce(T) -> U) -> ItemBinder<ItemId, U> {
        ItemBinder {
            item_id: self.item_id,
            val: f(self.val),
        }
    }

    fn assert_item_id(&self, item_id: ItemId) {
        assert_eq!(
            self.item_id, item_id,
            "Trying to use item bound for {:?} as if it belonged to {:?}",
            self.item_id, item_id
        );
    }

    /// Assert that the value is bound for item `item_id`, and returns it. This is used when we
    /// plan to store the returned value inside that item.
    pub fn under_binder_of(self, item_id: ItemId) -> T {
        self.assert_item_id(item_id);
        self.val
    }

    /// Given generic args for `item_id`, assert that the value is bound for `item_id` and
    /// substitute it with the provided generic arguments. Because the arguments are bound in the
    /// context of another item, so it the resulting substituted value.
    pub fn subst_for<OtherItem: Debug + Copy + PartialEq>(
        self,
        item_id: ItemId,
        args: ItemBinder<OtherItem, &GenericArgs>,
    ) -> ItemBinder<OtherItem, T>
    where
        T: DriveMut,
    {
        self.assert_item_id(item_id);
        args.map_bound(|args| {
            let mut val = self.val;
            val.drive_mut(&mut SubstVisitor::new(args));
            val
        })
    }
}

/// Dummy item identifier that represents the current item when not ambiguous.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CurrentItem;

impl<T> ItemBinder<CurrentItem, T> {
    pub fn under_current_binder(self) -> T {
        self.val
    }
}

impl Ty {
    /// Return true if it is actually unit (i.e.: 0-tuple)
    pub fn is_unit(&self) -> bool {
        match self {
            Ty::Adt(TypeId::Tuple, args) => {
                assert!(args.regions.is_empty());
                assert!(args.const_generics.is_empty());
                args.types.is_empty()
            }
            _ => false,
        }
    }

    /// Return the unit type
    pub fn mk_unit() -> Ty {
        Ty::Adt(TypeId::Tuple, GenericArgs::empty())
    }

    /// Return true if this is a scalar type
    pub fn is_scalar(&self) -> bool {
        match self {
            Ty::Literal(kind) => kind.is_integer(),
            _ => false,
        }
    }

    pub fn is_unsigned_scalar(&self) -> bool {
        match self {
            Ty::Literal(LiteralTy::Integer(kind)) => kind.is_unsigned(),
            _ => false,
        }
    }

    pub fn is_signed_scalar(&self) -> bool {
        match self {
            Ty::Literal(LiteralTy::Integer(kind)) => kind.is_signed(),
            _ => false,
        }
    }

    /// Return true if the type is Box
    pub fn is_box(&self) -> bool {
        match self {
            Ty::Adt(TypeId::Assumed(AssumedTy::Box), generics) => {
                assert!(generics.regions.is_empty());
                assert!(generics.types.len() == 1);
                assert!(generics.const_generics.is_empty());
                true
            }
            _ => false,
        }
    }

    pub fn as_box(&self) -> Option<&Ty> {
        match self {
            Ty::Adt(TypeId::Assumed(AssumedTy::Box), generics) => {
                assert!(generics.regions.is_empty());
                assert!(generics.types.len() == 1);
                assert!(generics.const_generics.is_empty());
                Some(generics.types.get(0).unwrap())
            }
            _ => None,
        }
    }
}

impl Field {
    /// The new name for this field, as suggested by the `#[charon::rename]` attribute.
    pub fn renamed_name(&self) -> Option<&str> {
        self.attr_info.rename.as_deref().or(self.name.as_deref())
    }

    /// Whether this field has a `#[charon::opaque]` annotation.
    pub fn is_opaque(&self) -> bool {
        self.attr_info
            .attributes
            .iter()
            .any(|attr| attr.is_opaque())
    }
}

impl Variant {
    /// The new name for this variant, as suggested by the `#[charon::rename]` and
    /// `#[charon::variants_prefix]` attributes.
    pub fn renamed_name(&self) -> &str {
        self.attr_info
            .rename
            .as_deref()
            .unwrap_or(self.name.as_ref())
    }

    /// Whether this variant has a `#[charon::opaque]` annotation.
    pub fn is_opaque(&self) -> bool {
        self.attr_info
            .attributes
            .iter()
            .any(|attr| attr.is_opaque())
    }
}

impl PredicateOrigin {
    /// Whether the corresponding clause was declared on a trait.
    pub(crate) fn comes_from_trait(&self) -> bool {
        match self {
            Self::TraitSelf | Self::WhereClauseOnTrait | Self::TraitItem(_) => true,
            Self::WhereClauseOnFn | Self::WhereClauseOnType | Self::WhereClauseOnImpl => false,
        }
    }
}
