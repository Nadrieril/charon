//! Change trait associated types to be type parameters instead. E.g.
//! ```rust,ignore
//! trait Iterator {
//!     type Item;
//! }
//! fn merge<I, J>(...) -> ...
//! where
//!     I: Iterator,
//!     J: Iterator<Item = I::Item>,
//! {}
//! // becomes
//! trait Iterator<Item> {
//! }
//! fn merge<I, J, Item>(...) -> ...
//! where
//!     I: Iterator<Item>,
//!     J: Iterator<Item>,
//! {}
//! ```
//!
//! Our algorithm works as follows:
//! 1. We collect for each trait the list of extra parameters it will need, along with paths
//!    describing what associated types these parameters used to correspond to. E.g. the above
//!    example gives us one extra param for `Iterator`, corresponding to `Self::Item`.
//! 2. Given a `TraitRef` and the `TraitTypeConstraint`s that relate to it, we compute how many (nested)
//!    associated types are unbound and require extra parameters. E.g. for `merge` above, the `I:
//!    Iterator` constraint requires an extra type parameter, and the `J: Iterator` constraint
//!    comes with `J::Item = I::Item`, which requires no extra type parameter.
//!
//! We then update every single signature to add the necessary extra parameters and fix the trait
//! references. This includes going into function and type bodies to replace all trait type
//! references with our new parameters.
//!
//! Note that the two steps are mutually recursive: they each use the other to compute the needed
//! parameters. This reflects the recursiveness of the situation:
//! ```rust,ignore
//! trait Foo {
//!     // The `Output` constraint means we don't need an extra `Output` param on `Foo`. Without
//!     // that constraint, we would need one. Hence step1 on `Foo` calls step 2 on `Item: Bar`,
//!     which calls step1 on `Bar`.
//!     type Item: Bar<Output=u32>;
//! }
//! trait Bar {
//!     type Output;
//! }
//! ```
//!
//! In this process we detect recursive cases that we can't handle and skip them. For example:
//! ```rust,ignore
//! trait Bar {
//!     type BarTy;
//! }
//! trait Foo {
//!     type FooTy: Foo + Bar;
//! }
//! // becomes:
//! trait Bar<BarTy> {}
//! trait Foo {
//!     type FooTy: Foo + Bar<Self::FooTy_BarTy>;
//!     // We need to supply an argument to `Bar` now so we add a new associated type.
//!     type FooTy_BarTy;
//! }
//! ```
use std::collections::{HashMap, HashSet, VecDeque};

use derive_visitor::VisitorMut;
use itertools::Either;
use macros::{EnumAsGetters, EnumIsA, EnumToGetters};

use crate::{ast::*, ids::Vector};

use super::{ctx::UllbcPass, TransformCtx};

/// Compute for each trait whether it (transitively) references itself in its trait clauses.
fn compute_self_referential_traits(translated: &TranslatedCrate) -> Vector<TraitDeclId, bool> {
    // Whether a given trait is self-referential.
    #[derive(Clone, Copy)]
    enum IsSelfRef {
        Unprocessed,
        Processing,
        Computed(bool),
    }

    let mut self_ref = translated.trait_decls.map_ref(|_| IsSelfRef::Unprocessed);
    // We explore depth-first the clause dependencies of each trait. This is sufficient to detect
    // loops. This is guaranteed to only process each trait once, hence this terminates.
    fn compute(
        self_ref: &mut Vector<TraitDeclId, IsSelfRef>,
        translated: &TranslatedCrate,
        id: TraitDeclId,
    ) {
        match self_ref[id] {
            IsSelfRef::Unprocessed => {
                // Set a sentinel value so we can detect loops.
                self_ref[id] = IsSelfRef::Processing;
                let tr = &translated.trait_decls[id];
                for clause in tr
                    .generics
                    .trait_clauses
                    .iter()
                    .chain(tr.parent_clauses.iter())
                {
                    compute(self_ref, translated, clause.trait_.trait_id);
                }
                match self_ref[id] {
                    IsSelfRef::Unprocessed => unreachable!(),
                    IsSelfRef::Processing => {
                        // If `Processing` is still there, there was no loop.
                        self_ref[id] = IsSelfRef::Computed(false);
                    }
                    // `Computed` happens if there was a loop.
                    IsSelfRef::Computed(_) => {}
                }
            }
            IsSelfRef::Processing => {
                // That's a loop.
                self_ref[id] = IsSelfRef::Computed(true);
            }
            IsSelfRef::Computed(_) => {}
        }
    }
    for tr in &translated.trait_decls {
        compute(&mut self_ref, translated, tr.def_id);
    }
    self_ref.map(|is_self_ref| match is_self_ref {
        IsSelfRef::Computed(b) => b,
        IsSelfRef::Unprocessed | IsSelfRef::Processing => unreachable!(),
    })
}

/// A base clause: the special `Self: Trait` clause present in the declaration and
/// implementations of a trait, a local clause, or a top-level trait implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, EnumIsA, EnumToGetters)]
enum BaseClause {
    SelfClause,
    Local(TraitClauseId),
    Impl(TraitImplId),
}

/// A `TraitRef` represented as a path.
/// TODO: rename, it's not so local anymore.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LocalTraitRef {
    /// The base clause we start from.
    base: BaseClause,
    /// Starting from `base`, recursively take the ith parent clause of the current clause. Each id
    /// corresponds to a parent of the previous one, e.g.
    /// ```rust,ignore
    /// trait Foo {
    ///     type FooTy;
    /// }
    /// trait Bar: Foo {} // `Self: Foo` is parent clause 0
    /// trait Baz: Copy // `Self: Copy` is parent clause 0
    /// {
    ///     // `Self::BazTy: Bar` is parent clause 1
    ///     // `<Self::BazTy as Foo>::FooTy = bool` is type constraint 0
    ///     // The local ref for `<Self::BazTy as Foo>` has base `Self` and `parent_path == [1, 0]`.
    ///     type BazTy: Bar<FooTy = bool>;
    /// }
    /// ```
    parent_path: Vec<TraitClauseId>,
}

impl LocalTraitRef {
    /// Make a trait ref that refers to the special `Self` clause.
    fn self_ref() -> Self {
        Self {
            base: BaseClause::SelfClause,
            parent_path: vec![],
        }
    }
    /// Make a trait ref that refers to the given clause.
    fn local_clause(id: TraitClauseId) -> Self {
        Self {
            base: BaseClause::Local(id),
            parent_path: vec![],
        }
    }

    fn with_assoc_type(self, type_name: TraitItemName) -> AssocTypePath {
        AssocTypePath {
            tref: self,
            type_name,
        }
    }

    /// Given a ref on a local clause, move it to refer to `Self`. This is used when we generate
    /// paths for the `GenericParams` of a trait: within the `GenericParams` we don't know whether
    /// we're talking about local clauses of trait implied clauses; we fix this here.
    fn as_parent_clause(&self) -> Self {
        let BaseClause::Local(id) = self.base else {
            panic!()
        };
        let mut new_ref = self.clone();
        new_ref.base = BaseClause::SelfClause;
        new_ref.parent_path.insert(0, id);
        new_ref
    }

    /// Given a ref on `Self`, make it into a ref on the given clause.
    fn on_local_clause(&self, id: TraitClauseId) -> Self {
        assert!(matches!(self.base, BaseClause::SelfClause));
        let mut new_ref = self.clone();
        new_ref.base = BaseClause::Local(id);
        new_ref
    }

    /// Given a ref on `Self`, apply it on top of given ref.
    fn on_local_tref(&self, tref: &LocalTraitRef) -> LocalTraitRef {
        assert!(matches!(self.base, BaseClause::SelfClause));
        let mut new_ref = tref.clone();
        new_ref.parent_path.extend(self.parent_path.iter().copied());
        new_ref
    }

    /// References the same clause one level up.
    /// TODO: maybe we shouldn't mix parent and non-parent clauses like that,
    fn pop_base(&self) -> Self {
        let mut new_ref = self.clone();
        match self.base {
            BaseClause::Local(_) | BaseClause::Impl(_) => {
                new_ref.base = BaseClause::SelfClause;
            }
            BaseClause::SelfClause => {
                let new_base_id = new_ref.parent_path.remove(0);
                new_ref.base = BaseClause::Local(new_base_id);
            }
        }
        new_ref
    }

    // /// Remove the base and return it.
    // fn pop_base(&mut self) -> BaseClauseId {
    //     let old_base = self.base;
    //     self.base = if self.parent_path.is_empty() {
    //         BaseClauseId::SelfClause
    //     } else {
    //         let new_base_id = self.parent_path.remove(0);
    //         BaseClauseId::Local(new_base_id)
    //     };
    //     old_base
    // }
}

impl TraitRefKind {
    fn to_local(&self) -> Option<LocalTraitRef> {
        match self {
            TraitRefKind::SelfId => Some(LocalTraitRef {
                base: BaseClause::SelfClause,
                parent_path: vec![],
            }),
            TraitRefKind::Clause(id) => Some(LocalTraitRef {
                base: BaseClause::Local(*id),
                parent_path: vec![],
            }),
            TraitRefKind::ParentClause(tref, _, id) => {
                let mut path = tref.to_local()?;
                path.parent_path.push(*id);
                Some(path)
            }
            TraitRefKind::ItemClause(_, _, _, _) => unreachable!(),
            TraitRefKind::TraitImpl(_, _)
            | TraitRefKind::BuiltinOrAuto(_)
            | TraitRefKind::Dyn(_)
            | TraitRefKind::Unknown(_) => None,
        }
    }
}

impl TraitRef {
    fn to_local(&self) -> Option<LocalTraitRef> {
        self.kind.to_local()
    }
}

/// The path to an associated type that depends on a local clause.
#[derive(Debug, Clone)]
struct AssocTypePath {
    /// The trait clause that has the associated type.
    tref: LocalTraitRef,
    /// The name of the associated type.
    type_name: TraitItemName,
}

impl AssocTypePath {
    /// Construct a path that points to the associated type named `type_name` of `Self`.
    fn assoc_type_of_self(type_name: TraitItemName) -> Self {
        Self {
            tref: LocalTraitRef::self_ref(),
            type_name,
        }
    }

    /// See [`LocalTraitRef::on_self`].
    fn on_self(&self) -> AssocTypePath {
        Self {
            tref: self.tref.as_parent_clause(),
            type_name: self.type_name.clone(),
        }
    }

    /// See [`LocalTraitRef::on_local_clause`].
    fn on_local_clause(&self, id: TraitClauseId) -> AssocTypePath {
        Self {
            tref: self.tref.on_local_clause(id),
            type_name: self.type_name.clone(),
        }
    }

    /// See [`LocalTraitRef::on_local_tref`].
    fn on_local_tref(&self, tref: &LocalTraitRef) -> AssocTypePath {
        Self {
            tref: self.tref.on_local_tref(&tref),
            type_name: self.type_name.clone(),
        }
    }

    /// See [`LocalTraitRef::pop_base`].
    fn pop_base(&self) -> Self {
        Self {
            tref: self.tref.pop_base(),
            type_name: self.type_name.clone(),
        }
    }

    // /// See [`LocalTraitRef::pop_base`].
    // fn pop_base(&mut self) -> BaseClauseId {
    //     self.tref.pop_base()
    // }

    /// Transform a trait type constraint into a (path, ty) pair. If the constraint does not refer
    /// to a local clause, we return it unchanged.
    fn from_constraint(cstr: &TraitTypeConstraint) -> Option<Self> {
        cstr.trait_ref.to_local().map(|local_tref| AssocTypePath {
            tref: local_tref,
            type_name: cstr.type_name.clone(),
        })
    }

    fn to_name(&self) -> String {
        use std::fmt::Write;
        let mut buf = match &self.tref.base {
            BaseClause::SelfClause => "Self".to_string(),
            BaseClause::Local(id) => format!("Clause{id}"),
            BaseClause::Impl(id) => format!("Impl{id}"),
        };
        for id in &self.tref.parent_path {
            let _ = write!(&mut buf, "_Clause{id}");
        }
        let _ = write!(&mut buf, "_{}", self.type_name);
        buf
    }
}

fn lookup_path_in_impl(
    translated: &TranslatedCrate,
    mut path: AssocTypePath,
    impl_id: TraitImplId,
) -> Option<ItemBinder<TraitImplId, Ty>> {
    assert!(path.tref.base.is_self_clause());
    let timpl = translated.trait_impls.get(impl_id)?;
    Some(match path.tref.parent_path.as_slice() {
        [] => {
            let ty = timpl
                .types
                .iter()
                .find(|(name, _)| name == &path.type_name)?
                .1
                .clone();
            ItemBinder::new(timpl.def_id, ty)
        }
        [_, ..] => {
            let clause_id = path.tref.parent_path.remove(0);
            let tref = &timpl.parent_trait_refs[clause_id];
            if let TraitRefKind::TraitImpl(impl_id, generics) = &tref.kind {
                let generics = ItemBinder::new(timpl.def_id, generics);
                lookup_path_in_impl(translated, path, *impl_id)?.subst_for(*impl_id, generics)
            } else {
                return None;
            }
        }
    })
}

/// A set of local `TraitTypeConstraint`s, represented as a trie.
#[derive(Default, Clone)]
struct TypeConstraintSet {
    /// For each base clause, a sub-trie of type constraints.
    clauses: HashMap<BaseClause, TypeConstraintSetInner>,
    /// What the `Self` clause refers to, if any.
    self_clause: Option<SelfClause>,
}

/// A set of `TraitTypeConstraint`s that apply to a trait.
#[derive(Debug, Default, Clone)]
struct TypeConstraintSetInner {
    /// The types that correspond to `Self::<Name>` for each `Name`. We also remember the id of the
    /// original constraint.
    assoc_tys: HashMap<TraitItemName, (Option<TraitTypeConstraintId>, Ty)>,
    /// The types that depend on the ith parent clause.
    parent_clauses: Vector<TraitClauseId, Self>,
}

impl TypeConstraintSet {
    fn from_constraints(
        self_clause: Option<SelfClause>,
        constraints: &Vector<TraitTypeConstraintId, TraitTypeConstraint>,
    ) -> Self {
        let mut this = TypeConstraintSet {
            clauses: Default::default(),
            self_clause,
        };
        for (i, c) in constraints.iter_indexed() {
            this.insert_type_constraint(i, c);
        }
        this
    }

    /// Add a type constraint to the set.
    fn insert_inner(&mut self, path: &AssocTypePath, cid: Option<TraitTypeConstraintId>, ty: Ty) {
        let mut trie = self.clauses.entry(path.tref.base).or_default();
        for id in &path.tref.parent_path {
            trie = trie
                .parent_clauses
                .get_or_extend_and_insert(*id, Default::default);
        }
        trie.assoc_tys.insert(path.type_name.clone(), (cid, ty));
    }

    /// Add a type constraint to the set.
    fn insert_path(&mut self, path: &AssocTypePath, ty: Ty) {
        self.insert_inner(path, None, ty);
    }

    /// Add a type constraint to the set.
    fn insert_type_constraint(&mut self, cid: TraitTypeConstraintId, cstr: &TraitTypeConstraint) {
        // We ignore non-local constraints.
        if let Some(path) = AssocTypePath::from_constraint(cstr) {
            self.insert_inner(&path, Some(cid), cstr.ty.clone());
        }
    }

    /// Find the entry at that path from the trie, if it exists.
    fn find(&self, path: &AssocTypePath) -> Option<(Option<TraitTypeConstraintId>, &Ty)> {
        let mut trie = self.clauses.get(&path.tref.base)?;
        for id in &path.tref.parent_path {
            trie = trie.parent_clauses.get(*id)?;
        }
        trie.assoc_tys
            .get(&path.type_name)
            .map(|(id, ty)| (*id, ty))
    }

    /// Find the entry in the set or if relevant use the `self_clause` information.
    fn find_here_or_self_clause(
        &self,
        path: &AssocTypePath,
        translated: Option<&TranslatedCrate>,
    ) -> Option<(Option<TraitTypeConstraintId>, Ty)> {
        if let Some((cstr, ty)) = self.find(path) {
            Some((cstr, ty.clone()))
        } else if path.tref.base.is_self_clause()
            && let Some(translated) = translated
            && let Some(SelfClause::Impl(impl_id)) = self.self_clause
            && let Some(ty) = lookup_path_in_impl(translated, path.clone(), impl_id)
        {
            // Remove the binder.
            let ty = ty.under_binder_of(impl_id);
            Some((None, ty))
        } else {
            None
        }
    }
}

impl TypeConstraintSetInner {
    fn debug_fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        base: BaseClause,
        parents: &[TraitClauseId],
    ) -> std::fmt::Result {
        for (name, (_, ty)) in &self.assoc_tys {
            match base {
                BaseClause::SelfClause => write!(f, "Self")?,
                BaseClause::Local(id) => write!(f, "Clause{id}")?,
                BaseClause::Impl(id) => write!(f, "Impl{id}")?,
            }
            for id in parents {
                write!(f, "::Clause{id}")?;
            }
            write!(f, "::{name} = {ty:?}, ")?;
        }
        for (parent_id, parent_trie) in self.parent_clauses.iter_indexed() {
            let mut new_parents = parents.to_vec();
            new_parents.push(parent_id);
            parent_trie.debug_fmt(f, base, &new_parents)?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for TypeConstraintSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("{ ")?;
        for (base, inner) in self.clauses.iter() {
            inner.debug_fmt(f, *base, &[])?;
        }
        f.write_str("}")?;
        Ok(())
    }
}

/// Items can have the special `Self` clause, and it comes in two kinds: trait declarations and
/// trait implementations.
#[derive(Debug, Copy, Clone)]
enum SelfClause {
    Impl(TraitImplId),
    Decl(TraitDeclId),
}

/// Records the modifications we are operating on the given item. =
#[derive(Debug, Clone)]
struct ItemModifications {
    /// The constraints on associated types for this item.
    type_constraints: TypeConstraintSet,
    /// Associated item paths that will be replaced. If there was an appropriate
    /// `TraitTypeConstraint`, we know which type to use for this path. Otherwise, the type is
    /// `None` and we will have to add a new type parameter to the signature of the item.
    replace_paths: Vec<(AssocTypePath, Option<Ty>)>,
    /// These constraints refer to now-removed associated type. We have taken them into account,
    /// they should be removed.
    remove_constraints: HashSet<TraitTypeConstraintId>,
    /// For trait declarations only: we remove any existing associated types. This is used by trait
    /// implementations to update themselves accordingly.
    remove_assoc_types: bool,
    /// If `add_assoc_types_instead` is `true`, instead of adding new type parameters we add new
    /// associated types. This only makes sense for trait declarations.
    add_assoc_types_instead: bool,
}

impl ItemModifications {
    fn new(
        type_constraints: &Vector<TraitTypeConstraintId, TraitTypeConstraint>,
        self_clause: Option<SelfClause>,
        is_self_referential_trait: bool,
    ) -> Self {
        Self {
            type_constraints: TypeConstraintSet::from_constraints(self_clause, type_constraints),
            replace_paths: Default::default(),
            remove_constraints: Default::default(),
            remove_assoc_types: !is_self_referential_trait,
            add_assoc_types_instead: is_self_referential_trait,
        }
    }

    /// Record that we must replace this path. If there is a relevant type constraint we can use
    /// it, otherwise we will need to add a new type parameter to supply a value for this path.
    fn replace_path(&mut self, path: AssocTypePath, translated: Option<&TranslatedCrate>) {
        if let Some((cstr_id, ty)) = self
            .type_constraints
            .find_here_or_self_clause(&path, translated)
        {
            // The local constraints already give us a value for this associated type; we
            // use that.
            if let Some(cstr_id) = cstr_id {
                self.remove_constraints.insert(cstr_id);
            }
            self.replace_paths.push((path, Some(ty)));
        } else {
            // We don't have a value for this associated type; we add a new type parameter
            // to supply it.
            self.replace_paths.push((path, None));
        }
    }

    /// Iterate over the paths that need to be filled.
    fn required_extra_paths(&self) -> impl Iterator<Item = &AssocTypePath> {
        self.replace_paths
            .iter()
            .filter(|(_, opt_ty)| opt_ty.is_none())
            .map(|(path, _)| path)
    }

    /// Iterate over the paths that require adding new type parameters.
    fn required_extra_params(&self) -> impl Iterator<Item = &AssocTypePath> {
        if self.add_assoc_types_instead {
            Either::Left([].into_iter())
        } else {
            Either::Right(self.required_extra_paths())
        }
    }

    /// Iterate over the paths that require adding new associated types.
    fn required_extra_assoc_types(&self) -> impl Iterator<Item = &AssocTypePath> {
        if self.add_assoc_types_instead {
            Either::Right(self.required_extra_paths())
        } else {
            Either::Left([].into_iter())
        }
    }

    /// Compute the type replacements needed to update the body of this item. We use the provided
    /// function to make new types for each missing type. That function will typically add new type
    /// parameters to the item signature.
    fn compute_replacements(&self, mut f: impl FnMut(&AssocTypePath) -> Ty) -> TypeConstraintSet {
        let mut set = TypeConstraintSet::default();
        for (path, opt_ty) in &self.replace_paths {
            let ty = opt_ty.clone().unwrap_or_else(|| f(path));
            set.insert_path(path, ty);
        }
        set
    }
}

/// Records what we computed in step1 for each trait declaration.
#[derive(Debug, EnumAsGetters)]
enum TraitModifications {
    /// We haven't analyzed this trait yet.
    Unprocessed,
    /// Sentinel value that we set when starting step1 on a trait. If we ever encounter this, we
    /// know we encountered a loop that we can't handle. We therefore skip that trait and leave its
    /// associated types untouched.
    Processing,
    /// The result of computing the modifications needed for this trait.
    Processed(ItemModifications),
}

struct ComputeItemModifications {
    self_referential_traits: Vector<TraitDeclId, bool>,
    trait_modifications: Vector<TraitDeclId, TraitModifications>,
}

impl ComputeItemModifications {
    /// Compute the modifications required to each item.
    fn compute(translated: &TranslatedCrate) -> HashMap<AnyTransId, ItemModifications> {
        let mut computer = ComputeItemModifications {
            self_referential_traits: compute_self_referential_traits(translated),
            trait_modifications: translated
                .trait_decls
                .map_ref(|_| TraitModifications::Unprocessed),
        };
        translated
            .all_items_with_ids()
            .map(|(id, item)| {
                let self_clause = match item {
                    AnyTransItem::TraitDecl(tdecl) => Some(SelfClause::Decl(tdecl.def_id)),
                    AnyTransItem::TraitImpl(timpl) => Some(SelfClause::Impl(timpl.def_id)),
                    AnyTransItem::Fun(FunDecl {
                        kind: ItemKind::TraitItemImpl { impl_id, .. },
                        ..
                    })
                    | AnyTransItem::Global(GlobalDecl {
                        kind: ItemKind::TraitItemImpl { impl_id, .. },
                        ..
                    }) => Some(SelfClause::Impl(*impl_id)),
                    AnyTransItem::Fun(FunDecl {
                        kind:
                            ItemKind::TraitItemDecl(trait_id, _)
                            | ItemKind::TraitItemProvided(trait_id, _),
                        ..
                    })
                    | AnyTransItem::Global(GlobalDecl {
                        kind:
                            ItemKind::TraitItemDecl(trait_id, _)
                            | ItemKind::TraitItemProvided(trait_id, _),
                        ..
                    }) => Some(SelfClause::Decl(*trait_id)),
                    _ => None,
                };

                let modifications = if let AnyTransId::TraitDecl(id) = id {
                    // The complex case is traits: we call `compute_extra_params_for_trait` to
                    // compute the right thing.
                    let _ = computer.compute_extra_params_for_trait(translated, id);
                    computer.trait_modifications[id].as_processed().clone()
                } else {
                    // For non-traits, we simply iterate through the clauses of the item and
                    // collect new paths to replace.
                    let params = item.generic_params();
                    let mut modifications =
                        ItemModifications::new(&params.trait_type_constraints, self_clause, false);
                    // For each clause, we need to supply types for the new parameters.
                    // `replace_path` either finds the right type in the type constraints, or
                    // records that we need to add new parameters to this trait's signature.
                    for clause in &params.trait_clauses {
                        let trait_id = clause.trait_.trait_id;
                        for path in computer.compute_extra_params_for_trait(translated, trait_id) {
                            let path = path.on_local_clause(clause.clause_id);
                            modifications.replace_path(path, Some(translated));
                        }
                    }
                    // Add the virtual `Self: Trait` clause. If we're in a trait impl, also add the
                    // required extra associated types.
                    if let Some(self_clause) = self_clause {
                        let trait_id = match self_clause {
                            SelfClause::Impl(impl_id) => {
                                // TODO: avoid this unwrap
                                translated
                                    .trait_impls
                                    .get(impl_id)
                                    .unwrap()
                                    .impl_trait
                                    .trait_id
                            }
                            SelfClause::Decl(trait_id) => trait_id,
                        };
                        let _ = computer.compute_extra_params_for_trait(translated, trait_id);
                        let decl_modifs = computer.trait_modifications[trait_id].as_processed();
                        if matches!(self_clause, SelfClause::Impl(_))
                            || !decl_modifs.add_assoc_types_instead
                        {
                            for path in decl_modifs.required_extra_paths() {
                                modifications.replace_path(path.clone(), Some(translated));
                            }
                        }
                    }
                    modifications
                };
                (id, modifications)
            })
            .collect()
    }

    /// Returns the extra parameters that we add to the given trait. If we hadn't processed this
    /// trait before, we modify it to add the parameters in question and remove its associated
    /// types.
    fn compute_extra_params_for_trait(
        &mut self,
        translated: &TranslatedCrate,
        id: TraitDeclId,
    ) -> impl Iterator<Item = &AssocTypePath> {
        if let TraitModifications::Unprocessed = &self.trait_modifications[id] {
            // Put the sentinel value to detect loops.
            self.trait_modifications[id] = TraitModifications::Processing;

            let is_self_referential = self.self_referential_traits[id];
            let tr = &translated.trait_decls[id];
            let mut modifications = ItemModifications::new(
                &tr.generics.trait_type_constraints,
                Some(SelfClause::Decl(id)),
                is_self_referential,
            );

            if modifications.remove_assoc_types {
                // Remove all associated types and turn them into new parameters.
                for type_name in &tr.types {
                    let path = AssocTypePath::assoc_type_of_self(type_name.clone());
                    modifications.replace_path(path, Some(translated));
                }
            }

            // The heart of the recursion: we process the trait clauses, which recursively computes
            // the extra parameters needed for each corresponding trait. Each of the referenced
            // traits may need to be supplied extra type parameters. `replace_path` either finds
            // the right type in the type constraints, or records that we need to add new
            // parameters to this trait's signature.
            for clause in &tr.generics.trait_clauses {
                let trait_id = clause.trait_.trait_id;
                for path in self.compute_extra_params_for_trait(translated, trait_id) {
                    let path = path.on_local_clause(clause.clause_id);
                    modifications.replace_path(path, Some(translated));
                }
            }
            for clause in &tr.parent_clauses {
                let trait_id = clause.trait_.trait_id;
                for path in self.compute_extra_params_for_trait(translated, trait_id) {
                    let path = path.on_local_clause(clause.clause_id).on_self();
                    modifications.replace_path(path, Some(translated));
                }
            }

            self.trait_modifications[id] = TraitModifications::Processed(modifications);
        }

        match &self.trait_modifications[id] {
            TraitModifications::Unprocessed => unreachable!(),
            TraitModifications::Processing => {
                // We're recursively processing ourselves. We already know we won't add new type
                // parameters, so we correctly return an empty iterator.
                assert!(self.self_referential_traits[id]);
                Either::Left([].into_iter())
            }
            TraitModifications::Processed(modifs) => Either::Right(modifs.required_extra_params()),
        }
    }
}

/// Visitor that will traverse item bodies and update `GenericArgs` to match the new item
/// signatures. This also replaces `Ty::TraitType`s for which we have a replacement. This is
/// modeled onto `check_generics::CheckGenericsVisitor`.
#[derive(VisitorMut)]
#[visitor(
    AggregateKind(enter),
    FnPtr(enter),
    GenericArgs(enter),
    GlobalDeclRef(enter),
    ImplElem,
    RawConstantExpr(enter),
    TraitClause(enter),
    TraitImpl(enter),
    TraitRefKind(enter),
    TraitTypeConstraint(enter),
    Ty(enter)
)]
struct UpdateItemBody<'a> {
    item_modifications: &'a HashMap<AnyTransId, ItemModifications>,
    // It's a reversed stack, for when we go under binders. Atm the only case of binders is `ImplElem`;
    type_replacements: VecDeque<TypeConstraintSet>,
    // Count how many `GenericArgs` we handled. This is to make sure we don't miss one.
    discharged_args: u32,
}

impl UpdateItemBody<'_> {
    /// Count that we just discharged one instance of `GenericArgs`.
    fn discharged_one_generics(&mut self) {
        self.discharged_args += 1;
    }

    fn lookup_type_replacement(&self, path: &AssocTypePath) -> Option<&Ty> {
        Some(self.type_replacements[0].find(&path)?.1)
    }

    fn lookup_path_in_trait_ref(&self, path: &AssocTypePath, tref: &TraitRefKind) -> Option<Ty> {
        match tref {
            TraitRefKind::TraitImpl(impl_id, generics) => {
                let generics = ItemBinder::new(CurrentItem, generics);
                assert!(path.tref.base.is_self_clause());
                let impl_modifs = self.item_modifications.get(&(*impl_id).into())?;
                // Some(match path.tref.parent_path.as_slice() {
                //     [] => {
                let ty = impl_modifs.type_constraints.find(path)?.1.clone();
                Some(
                    ItemBinder::new(impl_id, ty)
                        .subst_for(impl_id, generics)
                        .under_current_binder(),
                )
                // }
                // [_, ..] => {
                //     let clause_id = path.tref.parent_path.remove(0);
                //     let tref = &timpl.parent_trait_refs[clause_id];
                //     if let TraitRefKind::TraitImpl(impl_id, generics) = &tref.kind {
                //         let generics = ItemBinder::new(timpl.def_id, generics);
                //         lookup_path_in_impl(translated, path, *impl_id)?
                //             .subst_for(*impl_id, generics)
                //     } else {
                //         return None;
                //     }
                // }
                // })
            }
            _ => None,
        }
    }

    fn update_generics_for_item_inner(
        &mut self,
        args: &mut GenericArgs,
        item_id: impl Into<AnyTransId> + Copy,
        local_tref: Option<LocalTraitRef>,
    ) {
        self.discharged_one_generics();
        if let Some(modifications) = self.item_modifications.get(&item_id.into()) {
            for path in modifications.required_extra_params() {
                let mut path = path.clone();
                if let Some(tref) = &local_tref {
                    // `path.tref.base` is `Self` because we know `item_id` refers to a trait and
                    // traits don't have non-parent clauses atm.
                    path = path.on_local_tref(tref);
                }
                if let Some(ty) = self.lookup_type_replacement(&path) {
                    trace!("Adding type argument {ty:?}");
                    args.types.push(ty.clone());
                } else if let BaseClause::Local(clause_id) = path.tref.base {
                    let tref = &args.trait_refs[clause_id.index()];
                    let path = path.pop_base();
                    if let Some(ty) = self.lookup_path_in_trait_ref(&path, &tref.kind) {
                        args.types.push(ty.clone());
                    } else {
                        // TODO: error case
                    }
                } else {
                    // panic!(
                    //     "Could not find type for path {}.\nItem is {:?}\nLocal tref is {:?}\nAvailable: {:?}\nArgs: {args:?}",
                    //     path.to_name(),
                    //     item_id.into(),
                    //     local_tref,
                    //     self.type_replacements
                    // );
                }
            }
        }
    }

    fn update_generics_for_item(
        &mut self,
        args: &mut GenericArgs,
        item_id: impl Into<AnyTransId> + Copy,
    ) {
        self.update_generics_for_item_inner(args, item_id, None);
    }

    /// A `TraitDeclRef` must always be paired with an implementation in order to get access to the
    /// appropriate associated types. We have cases: `Self`, `Self::Clause`, arbitrary `TraitRef`.
    fn process_trait_decl_ref(&mut self, tref: &mut TraitDeclRef, local_tref: LocalTraitRef) {
        trace!("{tref:?}");
        self.update_generics_for_item_inner(&mut tref.generics, tref.trait_id, Some(local_tref));
    }

    /// Same as above, this needs more context.
    fn process_trait_ref(&mut self, tref: &mut TraitRef, local_tref: Option<LocalTraitRef>) {
        trace!("{tref:?}");
        // TODO: the types I need might be in the referenced implementation
        // TODO: pass a BaseClause::Impl and be very careful with generics.
        // Should I set a `Self` type for paths that contains an impl and generic args?
        // Should a path really contain `Impl`?
        // What's the general case here?
        // TODO: is `local_ref` useful at all?
        let local_tref = tref.to_local().or(local_tref);
        self.update_generics_for_item_inner(
            &mut tref.trait_decl_ref.generics,
            tref.trait_decl_ref.trait_id,
            local_tref,
        )
    }
}

// Visitor functions
impl UpdateItemBody<'_> {
    fn enter_aggregate_kind(&mut self, agg: &mut AggregateKind) {
        match agg {
            AggregateKind::Adt(TypeId::Adt(id), _, args) => {
                self.update_generics_for_item(args, *id);
            }
            AggregateKind::Closure(id, args) => {
                self.update_generics_for_item(args, *id);
            }
            AggregateKind::Adt(TypeId::Tuple | TypeId::Assumed(..), ..) => {
                // These generics don't need to be updated.
                self.discharged_one_generics();
            }
            AggregateKind::Array(..) => {}
        }
    }

    fn enter_fn_ptr(&mut self, fn_ptr: &mut FnPtr) {
        let args = &mut fn_ptr.generics;
        match &mut fn_ptr.func {
            FunIdOrTraitMethodRef::Fun(FunId::Regular(id)) => {
                self.update_generics_for_item(args, *id);
            }
            FunIdOrTraitMethodRef::Trait(tref, _, id) => {
                self.update_generics_for_item(args, *id);
                self.process_trait_ref(tref, None);
            }
            FunIdOrTraitMethodRef::Fun(FunId::Assumed(..)) => {
                // These generics don't need to be updated.
                self.discharged_one_generics();
            }
        }
    }

    fn enter_generic_args(&mut self, args: &mut GenericArgs) {
        if self.discharged_args == 0 {
            // Ensure we counted all `GenericArgs`
            panic!("Unexpected `GenericArgs` in the AST! {args:?}")
        }
        self.discharged_args -= 1;
        for tref in args.trait_refs.iter_mut() {
            self.process_trait_ref(tref, None);
        }
    }

    fn enter_global_decl_ref(&mut self, global_ref: &mut GlobalDeclRef) {
        self.update_generics_for_item(&mut global_ref.generics, global_ref.id);
    }

    fn enter_impl_elem(&mut self, elem: &mut ImplElem) {
        match elem {
            ImplElem::Ty(generics, _ty) => {
                // An `ImplElem` functions like a binder. Because it's not a globally-addressable item, we
                // haven't computed appropriate modifications yet, so we compute them.
                let mut modifications =
                    ItemModifications::new(&generics.trait_type_constraints, None, false);
                for clause in &generics.trait_clauses {
                    let trait_id = clause.trait_.trait_id;
                    if let Some(trait_mods) = self.item_modifications.get(&trait_id.into()) {
                        for path in trait_mods.required_extra_params() {
                            let path = path.on_local_clause(clause.clause_id);
                            modifications.replace_path(path, None);
                        }
                    }
                }
                let replacements = modifications.compute_replacements(|path| {
                    let var_id = generics
                        .types
                        .push_with(|id| TypeVar::new(id, path.to_name()));
                    Ty::TypeVar(var_id)
                });
                self.type_replacements.push_front(replacements);
            }
            ImplElem::Trait(..) => {}
        }
    }
    fn exit_impl_elem(&mut self, elem: &mut ImplElem) {
        match elem {
            ImplElem::Ty(..) => {
                self.type_replacements.pop_front();
            }
            ImplElem::Trait(..) => {}
        }
    }

    fn enter_raw_constant_expr(&mut self, cexpr: &mut RawConstantExpr) {
        match cexpr {
            RawConstantExpr::TraitConst(tref, _) => {
                self.process_trait_ref(tref, None);
            }
            _ => {}
        }
    }

    fn enter_trait_clause(&mut self, clause: &mut TraitClause) {
        let mut tref = LocalTraitRef::local_clause(clause.clause_id);
        if clause.origin.comes_from_trait() {
            // Currently all clauses on traits are treated as parent clauses, hence must be
            // referred to as `Self::Clausei`.
            tref = tref.as_parent_clause();
        }
        self.process_trait_decl_ref(&mut clause.trait_, tref)
    }

    fn enter_trait_impl(&mut self, timpl: &mut TraitImpl) {
        self.process_trait_decl_ref(&mut timpl.impl_trait, LocalTraitRef::self_ref());
        for (clause_id, tref) in timpl.parent_trait_refs.iter_mut_indexed() {
            self.process_trait_ref(
                tref,
                Some(LocalTraitRef::local_clause(clause_id).as_parent_clause()),
            );
        }
    }

    fn enter_trait_ref_kind(&mut self, kind: &mut TraitRefKind) {
        trace!("{kind:?}");
        match kind {
            TraitRefKind::TraitImpl(id, args) => self.update_generics_for_item(args, *id),
            // Nothing to update: built-in traits don't have associated types.
            TraitRefKind::BuiltinOrAuto(..) => self.discharged_one_generics(),
            // TODO: `TraitRefKind::Dyn` is missing assoc type info, we can't update this. THis
            // will cause type errors.
            TraitRefKind::Dyn(..) => self.discharged_one_generics(),
            TraitRefKind::Clause(..)
            | TraitRefKind::ParentClause(..)
            | TraitRefKind::ItemClause(..)
            | TraitRefKind::SelfId
            | TraitRefKind::Unknown(_) => {}
        }
    }

    fn enter_trait_type_constraint(&mut self, constraint: &mut TraitTypeConstraint) {
        self.process_trait_ref(&mut constraint.trait_ref, None);
    }

    fn enter_ty(&mut self, ty: &mut Ty) {
        trace!("{ty:?}");
        match ty {
            Ty::Adt(TypeId::Adt(id), args) => self.update_generics_for_item(args, *id),
            Ty::Adt(TypeId::Tuple | TypeId::Assumed(..), ..) => {
                // These generics don't need to be updated.
                self.discharged_one_generics();
            }
            Ty::TraitType(tref, name) => {
                self.process_trait_ref(tref, None);
                if let Some(local_tref) = tref.to_local() {
                    let path = local_tref.with_assoc_type(name.clone());
                    if let Some(new_ty) = self.lookup_type_replacement(&path) {
                        *ty = new_ty.clone();
                        // Fix the newly-substituted type.
                        self.visit(ty, derive_visitor::Event::Enter);
                    }
                }
            }
            _ => (),
        }
    }
}

pub struct Transform;
impl UllbcPass for Transform {
    fn transform_ctx(&self, ctx: &mut TransformCtx<'_>) {
        // Compute the required signature modifications.
        let item_modifications: HashMap<AnyTransId, ItemModifications> =
            ComputeItemModifications::compute(&ctx.translated);

        // Apply the computed modifications.
        for (id, modifications) in &item_modifications {
            let mut item = ctx.translated.get_item_mut(*id).unwrap();

            // Remove trait associated types.
            if let AnyTransItemMut::TraitDecl(tr) = &mut item
                && modifications.remove_assoc_types
            {
                tr.types.clear();
            }

            // Remove used constraints.
            for cid in &modifications.remove_constraints {
                item.generic_params().trait_type_constraints.remove(*cid);
            }

            // Add new parameters or associated types in order to have types to fill in all the
            // replaced paths. We then collect the replaced paths and their associated value.
            let type_replacements: TypeConstraintSet = if let AnyTransItemMut::TraitDecl(tr) =
                &mut item
                && modifications.add_assoc_types_instead
            {
                // If we're self-referential, instead of adding new type parameters to pass to our
                // parent clauses, we add new associated types. That way the parameter list of the
                // trait stays unchanged.
                let self_tref = TraitRef {
                    kind: TraitRefKind::SelfId,
                    trait_decl_ref: TraitDeclRef {
                        trait_id: tr.def_id,
                        generics: tr.generics.identity_args(),
                    },
                };
                modifications.compute_replacements(|path| {
                    let new_type_name = TraitItemName(path.to_name());
                    tr.types.push(new_type_name.clone());
                    Ty::TraitType(self_tref.clone(), new_type_name)
                })
            } else {
                modifications.compute_replacements(|path| {
                    let var_id = item
                        .generic_params()
                        .types
                        .push_with(|id| TypeVar::new(id, path.to_name()));
                    Ty::TypeVar(var_id)
                })
            };

            if let AnyTransItemMut::TraitImpl(timpl) = &mut item {
                let decl_modifs = item_modifications
                    .get(&timpl.impl_trait.trait_id.into())
                    .unwrap();
                if decl_modifs.remove_assoc_types {
                    timpl.types.clear();
                }
                for path in decl_modifs.required_extra_assoc_types() {
                    if let Some((_, ty)) = type_replacements.find(&path) {
                        let new_type_name = TraitItemName(path.to_name());
                        trace!("Adding associated type {new_type_name} = {ty:?}");
                        timpl.types.push((new_type_name, ty.clone()));
                    } else {
                        // panic!(
                        //     "Could not find type for path {}.\nItem is {:?}\nLocal tref is {:?}\nAvailable: {:?}",
                        //     path.to_name(),
                        //     item_id.into(),
                        //     self.local_tref,
                        //     self.type_replacements
                        // );
                    }
                }
                // TODO: do we need to do something special for `parent_trait_refs`?
            }

            item.drive_mut(&mut UpdateItemBody {
                item_modifications: &item_modifications,
                type_replacements: vec![type_replacements].into(),
                discharged_args: 0,
            });
        }
    }
}
