(** WARNING: this file is partially auto-generated. Do not edit `src/Meta.ml`
    by hand. Edit `templaces/Meta.ml` instead, or improve the code
    generation tool so avoid the need for hand-writing things.

    `templaces/Meta.ml` contains the manual definitions and some `(*
    __REPLACEn__ *)` comments. These comments are replaced by auto-generated
    definitions by running `make generate-ml` in the crate root. The
    code-generation code is in `charon/src/bin/generate-ml`.
 *)

(** Meta data like code spans *)

type path_buf = string
and loc = { line : int; col : int }
and file_name = Virtual of path_buf | Local of path_buf [@@deriving show, ord]

(** Span data *)
type raw_span = { file : file_name; beg_loc : loc; end_loc : loc }
[@@deriving show, ord]

type __meta_1 = unit (* to start the recursive group *)
and span = { span : raw_span; generated_from_span : raw_span option }
and inline_attr = Hint | Never | Always

and attribute =
  | AttrOpaque
  | AttrRename of string
  | AttrVariantsPrefix of string
  | AttrVariantsSuffix of string
  | AttrDocComment of string
  | AttrUnknown of string

and attr_info = {
  attributes : attribute list;
  inline : inline_attr option;
  rename : string option;
  public : bool;
}
[@@deriving show, ord]
