//! # aither-derive
//!
//! Procedural macros for converting Rust functions into AI tools that can be called by language models.
//!
//! This crate provides the `#[tool]` attribute macro that automatically generates the necessary
//! boilerplate code to make your async functions callable by AI models through `aither` or `aither-core`.
//!
//! ## Quick Start
//!
//! Transform any async function into an AI tool by adding the `#[tool]` attribute.
//! The tool description is the function's rustdoc, or, when the function has
//! none, the rustdoc on its single Args struct:
//!
//! ```rust
//! use aither_derive::tool;
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! /// Get the current UTC time.
//! #[derive(JsonSchema, Deserialize)]
//! pub struct GetTimeArgs;
//!
//! #[tool]
//! pub async fn get_time(_args: GetTimeArgs) -> &'static str {
//!     "2023-10-01T12:00:00Z"
//! }
//! ```
//!
//! A function returning `Result<T, E>` fails with `E` as its
//! [`Tool::Error`](aither_core::llm::Tool::Error); any other return type makes
//! the tool infallible.
//!
//! ## Requirements
//!
//! - Functions must be `async`
//! - Return type must be convertible via `IntoToolResult`
//! - Parameters must implement `serde::Deserialize` and `schemars::JsonSchema`
//! - No `self` parameters (static functions only)
//! - No lifetime or generic parameters

use std::path::Path;

use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Attribute, FnArg, Ident, ItemFn, LitStr, Pat, Path as SynPath, Token, Type, Visibility,
    parse::{Parse, ParseStream},
    parse_macro_input, parse_quote,
};

/// Arguments for the `#[tool]` attribute macro.
#[derive(Default)]
struct ToolArgs {
    rename: Option<String>,
    krate: Option<SynPath>,
}

impl Parse for ToolArgs {
    /// Parse the arguments from the `#[tool(...)]` attribute.
    ///
    /// Supports:
    /// - `rename = "..."` (optional): Custom name for the tool (defaults to function name)
    /// - `crate = ...` (optional): Path to `aither_core` or `aither`
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut rename = None;
        let mut krate = None;

        while !input.is_empty() {
            if input.peek(Token![crate]) {
                let _: Token![crate] = input.parse()?;
                let _: Token![=] = input.parse()?;
                let path = if input.peek(LitStr) {
                    let lit: LitStr = input.parse()?;
                    lit.parse::<SynPath>()?
                } else {
                    input.parse::<SynPath>()?
                };
                krate = Some(path);
            } else {
                let ident: Ident = input.parse()?;
                let _: Token![=] = input.parse()?;
                let value: LitStr = input.parse()?;

                match ident.to_string().as_str() {
                    "rename" => rename = Some(value.value()),
                    _ => {
                        return Err(syn::Error::new_spanned(
                            ident,
                            "unknown attribute. Supported: rename, crate",
                        ));
                    }
                }
            }

            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            }
        }

        Ok(Self { rename, krate })
    }
}

/// Converts an async function into an AI tool that can be called by language models.
///
/// Generates boilerplate implementing `Tool` against either `aither_core` or `aither`,
/// resolved automatically from the caller crate's `Cargo.toml` (or overridden with
/// `#[tool(crate = ...)]`).
#[proc_macro_attribute]
pub fn tool(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as ToolArgs);
    let input_fn = parse_macro_input!(input as ItemFn);

    match tool_impl(args, input_fn) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn resolve_runtime_crate(explicit: Option<SynPath>) -> SynPath {
    if let Some(path) = explicit {
        return path;
    }

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let pkg_name = std::env::var("CARGO_PKG_NAME").unwrap_or_default();
    let crate_name = std::env::var("CARGO_CRATE_NAME").unwrap_or_default();
    let resolved = detect_runtime_crate_path(&manifest_dir, &pkg_name, &crate_name);

    syn::parse_str::<SynPath>(&resolved).unwrap_or_else(|_| parse_quote!(::aither_core))
}

fn detect_runtime_crate_path(manifest_dir: &str, pkg_name: &str, crate_name: &str) -> String {
    if pkg_name == "aither-core" {
        return if crate_name == "aither_core" {
            String::from("crate")
        } else {
            String::from("::aither_core")
        };
    }
    if pkg_name == "aither" {
        return if crate_name == "aither" {
            String::from("crate")
        } else {
            String::from("::aither")
        };
    }

    if !manifest_dir.is_empty()
        && let Ok(manifest_text) =
            std::fs::read_to_string(Path::new(manifest_dir).join("Cargo.toml"))
        && let Some(found) = find_runtime_crate_in_manifest(&manifest_text)
    {
        return found;
    }

    String::from("::aither_core")
}

fn find_runtime_crate_in_manifest(manifest_text: &str) -> Option<String> {
    let doc = toml_edit::ImDocument::parse(manifest_text).ok()?;
    let root = doc.as_table();

    let dep_sections = ["dependencies", "dev-dependencies"];
    for section in dep_sections {
        for wanted in ["aither-core", "aither"] {
            if let Some(table) = root.get(section).and_then(toml_edit::Item::as_table_like)
                && let Some(found) = find_dep_in_table(table, wanted)
            {
                return Some(found);
            }
            if let Some(targets) = root.get("target").and_then(toml_edit::Item::as_table_like) {
                for (_, target_cfg) in targets.iter() {
                    if let Some(target_table) = target_cfg.as_table_like()
                        && let Some(table) = target_table
                            .get(section)
                            .and_then(toml_edit::Item::as_table_like)
                        && let Some(found) = find_dep_in_table(table, wanted)
                    {
                        return Some(found);
                    }
                }
            }
        }
    }

    None
}

fn find_dep_in_table(table: &dyn toml_edit::TableLike, wanted_pkg: &str) -> Option<String> {
    for (key, item) in table.iter() {
        let pkg = if let Some(dep_table) = item.as_table_like() {
            dep_table
                .get("package")
                .and_then(toml_edit::Item::as_str)
                .unwrap_or(key)
        } else if item.is_str() {
            key
        } else {
            continue;
        };
        if pkg == wanted_pkg {
            let ident = key.replace('-', "_");
            return Some(format!("::{ident}"));
        }
    }
    None
}

/// Whether `attr` belongs on a generated `Args` struct field rather than a raw Rust `fn` parameter.
fn is_field_attr(attr: &Attribute) -> bool {
    attr.path().is_ident("doc") || attr.path().is_ident("serde") || attr.path().is_ident("schemars")
}

/// The tool's `Error` type and the call expression the impl emits: a
/// `Result<T, E>` return makes `E` the error and the call `?`-propagates;
/// anything else stays infallible.
fn tool_error_and_call(
    input_fn: &ItemFn,
    call_expr: &proc_macro2::TokenStream,
    krate: &syn::Path,
) -> (proc_macro2::TokenStream, proc_macro2::TokenStream) {
    let error_ty = match &input_fn.sig.output {
        syn::ReturnType::Type(_, ty) => result_error_type(ty).cloned(),
        syn::ReturnType::Default => None,
    };
    error_ty.map_or_else(
        || {
            (
                quote! { ::core::convert::Infallible },
                quote! { #krate::llm::IntoToolResult::into_tool_result(#call_expr) },
            )
        },
        |error_ty| {
            (
                quote! { #error_ty },
                quote! { #krate::llm::IntoToolResult::into_tool_result(#call_expr?) },
            )
        },
    )
}

/// Implementation details for the `#[tool]` macro.
fn tool_impl(args: ToolArgs, mut input_fn: ItemFn) -> syn::Result<proc_macro2::TokenStream> {
    if !input_fn.sig.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &input_fn.sig.generics,
            "#[tool] functions cannot have lifetime or generic parameters",
        ));
    }
    if let Some(where_clause) = &input_fn.sig.generics.where_clause {
        return Err(syn::Error::new_spanned(
            where_clause,
            "#[tool] functions cannot have generic where clauses",
        ));
    }
    if input_fn.sig.asyncness.is_none() {
        return Err(syn::Error::new_spanned(
            &input_fn.sig,
            "Tool functions must be async",
        ));
    }
    for arg in &input_fn.sig.inputs {
        if let FnArg::Receiver(receiver) = arg {
            return Err(syn::Error::new_spanned(
                receiver,
                "#[tool] cannot be applied to a method taking `self`; use a free function",
            ));
        }
    }

    let krate = resolve_runtime_crate(args.krate);
    let fn_name = input_fn.sig.ident.clone();
    let tool_name = args.rename.unwrap_or_else(|| fn_name.to_string());
    let fn_vis = input_fn.vis.clone();

    let tool_struct_name = format_ident!("{}", fn_name.to_string().to_case(Case::Pascal));

    // The function's rustdoc describes the tool. Without it the description
    // falls back to the rustdoc on the arguments type, which only a
    // single-parameter tool has.
    let description = doc_text(&input_fn.attrs).map(|text| {
        quote! {
            fn description(&self) -> #krate::llm::tool::Cow<'static, str> {
                #text.into()
            }
        }
    });

    // The context parameter, if any, is not an argument the model fills in.
    let (context_position, data_inputs) = split_context_parameter(&input_fn.sig.inputs)?;

    // Analyze function signature before stripping field attributes from the emitted fn.
    let AnalyzedArgs {
        args_type,
        params,
        uses_generated_struct,
        stream,
    } = analyze_function_args(&fn_vis, &tool_struct_name, &data_inputs)?;

    // Strip field-only attributes (`///`, `#[serde(...)]`, `#[schemars(...)]`) from the
    // emitted function's parameter list so `rustc` compiles the function cleanly.
    for input in &mut input_fn.sig.inputs {
        if let FnArg::Typed(pat_type) = input {
            pat_type.attrs.retain(|attr| !is_field_attr(attr));
        }
    }

    // Call the function with its parameters in declaration order, the
    // context in the position the function declared it.
    let mut call_args: Vec<proc_macro2::TokenStream> =
        params.iter().map(|param| quote! { #param }).collect();
    if let Some(position) = context_position {
        call_args.insert(position, quote! { cx });
    }
    let call_expr = quote! { #fn_name(#(#call_args),*).await };

    // A `Result<T, E>` return type makes `E` the tool's error; anything else —
    // including single-argument aliases like `io::Result<T>` — converts through
    // `IntoToolResult` and stays infallible.
    let (error_type, call_body) = tool_error_and_call(&input_fn, &call_expr, &krate);

    let extractor = if uses_generated_struct {
        quote! { let Self::Arguments { #(#params),* } = args; }
    } else {
        quote! {}
    };

    let args_binding = if params.is_empty() {
        quote! { _args }
    } else {
        quote! { args }
    };
    let context_binding = if context_position.is_some() {
        quote! { cx }
    } else {
        quote! { _cx }
    };

    let expanded = quote! {
        #input_fn

        #stream

        #[allow(missing_docs)]
        #[derive(::core::default::Default, ::core::fmt::Debug)]
        #fn_vis struct #tool_struct_name;

        impl #krate::llm::Tool for #tool_struct_name {
            fn name(&self) -> #krate::llm::tool::Cow<'static, str> {
                #tool_name.into()
            }
            #description
            type Arguments = #args_type;
            type Res = #krate::llm::ToolResult;
            type Error = #error_type;

            async fn call(
                &self,
                #args_binding: Self::Arguments,
                #context_binding: #krate::llm::ToolContext,
            ) -> ::core::result::Result<Self::Res, Self::Error> {
                #extractor
                ::core::result::Result::Ok(#call_body)
            }
        }
    };

    Ok(expanded)
}

/// The `E` in a `Result<T, E>` return type, when the function returns one.
///
/// A single generic argument is a `Result<T>`-style alias whose error type
/// cannot be named here; it converts through `IntoToolResult` instead.
fn result_error_type(ty: &Type) -> Option<&Type> {
    let Type::Path(path) = ty else {
        return None;
    };
    if path.qself.is_some() {
        return None;
    }
    let segment = path.path.segments.last()?;
    if segment.ident != "Result" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    let mut types = args.args.iter().filter_map(|arg| match arg {
        syn::GenericArgument::Type(ty) => Some(ty),
        _ => None,
    });
    match (types.next(), types.next(), types.next()) {
        (Some(_ok), Some(error), None) => Some(error),
        _ => None,
    }
}

/// The text of the `///` comments in `attrs`, one line per attribute with the
/// single leading space rustdoc strips removed, or `None` when there are none.
fn doc_text(attrs: &[Attribute]) -> Option<String> {
    let lines: Vec<String> = attrs
        .iter()
        .filter(|attr| attr.path().is_ident("doc"))
        .filter_map(|attr| match &attr.meta {
            syn::Meta::NameValue(syn::MetaNameValue {
                value:
                    syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Str(text),
                        ..
                    }),
                ..
            }) => Some(text.value()),
            _ => None,
        })
        .map(|line| {
            line.strip_prefix(' ')
                .map_or_else(|| line.clone(), str::to_owned)
        })
        .collect();
    let text = lines.join("\n");
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_owned())
}

/// Separates the call's `ToolContext` parameter from the model-facing ones.
fn split_context_parameter(
    inputs: &syn::punctuated::Punctuated<FnArg, syn::Token![,]>,
) -> syn::Result<(
    Option<usize>,
    syn::punctuated::Punctuated<FnArg, syn::Token![,]>,
)> {
    let mut position = None;
    let mut data = syn::punctuated::Punctuated::new();
    for (index, input) in inputs.iter().enumerate() {
        if is_context_parameter(input) {
            if position.is_some() {
                return Err(syn::Error::new_spanned(
                    input,
                    "a tool function takes at most one `ToolContext` parameter",
                ));
            }
            if let FnArg::Typed(pat_type) = input {
                validate_context_pattern(&pat_type.pat)?;
                for attr in &pat_type.attrs {
                    if attr.path().is_ident("serde") || attr.path().is_ident("schemars") {
                        return Err(syn::Error::new_spanned(
                            attr,
                            "serde and schemars attributes are not supported on `ToolContext` parameters",
                        ));
                    }
                }
            }
            position = Some(index);
        } else {
            data.push(input.clone());
        }
    }
    Ok((position, data))
}

fn validate_context_pattern(pat: &Pat) -> syn::Result<()> {
    match pat {
        Pat::Ident(ident) => {
            if ident.by_ref.is_some() {
                return Err(syn::Error::new_spanned(
                    ident,
                    "`ref` parameter bindings are not supported in #[tool] functions",
                ));
            }
            if ident.subpat.is_some() {
                return Err(syn::Error::new_spanned(
                    ident,
                    "`@` subpattern bindings are not supported in #[tool] functions",
                ));
            }
            Ok(())
        }
        Pat::Wild(_) => Ok(()),
        other => Err(syn::Error::new_spanned(
            other,
            "unsupported parameter pattern in #[tool] function; expected an identifier",
        )),
    }
}

fn is_context_parameter(input: &FnArg) -> bool {
    let FnArg::Typed(pat_type) = input else {
        return false;
    };
    let Type::Path(path) = pat_type.ty.as_ref() else {
        return false;
    };
    path.qself.is_none()
        && path
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "ToolContext")
}

/// Container for analyzed function arguments and generated types.
struct AnalyzedArgs {
    /// The type used for the Tool's Arguments associated type
    args_type: Type,
    /// Parameter names extracted from the function signature
    params: Vec<Ident>,
    /// Whether `args` must be destructured out of a generated struct
    uses_generated_struct: bool,
    /// Generated argument struct definition (if needed)
    stream: proc_macro2::TokenStream,
}

fn extract_param_ident(pat: &Pat) -> syn::Result<Ident> {
    match pat {
        Pat::Ident(pat_ident) => {
            if pat_ident.by_ref.is_some() {
                return Err(syn::Error::new_spanned(
                    pat_ident,
                    "`ref` parameter bindings are not supported in #[tool] functions",
                ));
            }
            if pat_ident.subpat.is_some() {
                return Err(syn::Error::new_spanned(
                    pat_ident,
                    "`@` subpattern bindings are not supported in #[tool] functions",
                ));
            }
            Ok(pat_ident.ident.clone())
        }
        other => Err(syn::Error::new_spanned(
            other,
            "unsupported parameter pattern in #[tool] function; expected an identifier",
        )),
    }
}

/// Analyzes function parameters and generates appropriate argument types.
fn analyze_function_args(
    fn_vis: &Visibility,
    struct_name: &Ident,
    inputs: &syn::punctuated::Punctuated<FnArg, syn::Token![,]>,
) -> syn::Result<AnalyzedArgs> {
    if inputs.is_empty() {
        return Ok(AnalyzedArgs {
            args_type: parse_quote! { () },
            params: vec![],
            uses_generated_struct: false,
            stream: quote! {},
        });
    }

    if inputs.len() == 1 {
        let FnArg::Typed(pat_type) = &inputs[0] else {
            return Err(syn::Error::new_spanned(
                &inputs[0],
                "#[tool] cannot be applied to a method taking `self`; use a free function",
            ));
        };
        let has_field_attrs = pat_type.attrs.iter().any(is_field_attr);
        if !has_field_attrs {
            if !matches!(pat_type.pat.as_ref(), Pat::Wild(_)) {
                let _ = extract_param_ident(&pat_type.pat)?;
            }
            return Ok(AnalyzedArgs {
                args_type: (*pat_type.ty).clone(),
                params: vec![format_ident!("args")],
                uses_generated_struct: false,
                stream: quote! {},
            });
        }
    }

    let mut fields = Vec::with_capacity(inputs.len());
    let mut params = Vec::with_capacity(inputs.len());

    for arg in inputs {
        let FnArg::Typed(pat_type) = arg else {
            return Err(syn::Error::new_spanned(
                arg,
                "#[tool] cannot be applied to a method taking `self`; use a free function",
            ));
        };
        let ident = extract_param_ident(&pat_type.pat)?;
        let ty = &pat_type.ty;
        let field_attrs: Vec<&Attribute> = pat_type
            .attrs
            .iter()
            .filter(|attr| is_field_attr(attr))
            .collect();
        fields.push(quote! {
            #(#field_attrs)*
            #ident: #ty,
        });
        params.push(ident);
    }

    let arg_struct_name = format_ident!("{struct_name}Args");
    let new_type_gen = quote! {
        #[allow(missing_docs)]
        #[derive(::schemars::JsonSchema, ::serde::Deserialize, ::core::fmt::Debug)]
        #fn_vis struct #arg_struct_name {
            #(#fields)*
        }
    };

    Ok(AnalyzedArgs {
        args_type: parse_quote! { #arg_struct_name },
        params,
        uses_generated_struct: true,
        stream: new_type_gen,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_generic_type_and_lifetime_parameters() {
        let generic_fn: ItemFn = parse_quote! {
            /// Generic tool.
            async fn run<T: Send>(x: T) -> Result<String> {
                let _ = x;
                Ok(String::new())
            }
        };
        let err = tool_impl(ToolArgs::default(), generic_fn).unwrap_err();
        assert!(err.to_string().contains("generic"));

        let lifetime_fn: ItemFn = parse_quote! {
            /// Lifetime tool.
            async fn borrow<'a>(x: &'a str) -> Result<String> {
                Ok(x.to_owned())
            }
        };
        let err = tool_impl(ToolArgs::default(), lifetime_fn).unwrap_err();
        assert!(err.to_string().contains("lifetime or generic"));

        let where_fn: ItemFn = parse_quote! {
            /// Where-clause tool.
            async fn constrained(x: String) -> Result<String>
            where
                String: Clone,
            {
                Ok(x)
            }
        };
        let err = tool_impl(ToolArgs::default(), where_fn).unwrap_err();
        assert!(err.to_string().contains("where"));
    }

    #[test]
    fn rejects_unsupported_patterns_without_panicking() {
        let tuple_fn: ItemFn = parse_quote! {
            /// Tuple pattern.
            async fn pair((a, b): (u32, u32), c: u32) -> Result<String> {
                Ok(format!("{}", a + b + c))
            }
        };
        let err = tool_impl(ToolArgs::default(), tuple_fn).unwrap_err();
        assert!(err.to_string().contains("unsupported parameter pattern"));

        let ref_fn: ItemFn = parse_quote! {
            /// Ref binding.
            async fn by_ref(ref a: String, b: u32) -> Result<String> {
                Ok(format!("{a}:{b}"))
            }
        };
        let err = tool_impl(ToolArgs::default(), ref_fn).unwrap_err();
        assert!(err.to_string().contains("`ref` parameter bindings"));

        let self_fn: ItemFn = parse_quote! {
            /// Method.
            async fn method(&self, a: u32) -> Result<String> {
                let _ = a;
                Ok(String::new())
            }
        };
        let err = tool_impl(ToolArgs::default(), self_fn).unwrap_err();
        assert!(err.to_string().contains("taking `self`"));
    }

    #[test]
    fn expands_mut_bindings_and_preserves_field_attributes() {
        let input: ItemFn = parse_quote! {
            /// Rewrites a query.
            async fn rewrite(
                /// Search text.
                #[schemars(length(min = 1))]
                mut query: String,
                /// Maximum hits.
                #[serde(default)]
                mut limit: usize,
            ) -> Result<String> {
                query.push('!');
                limit += 1;
                Ok(format!("{query}:{limit}"))
            }
        };
        let tokens = tool_impl(
            ToolArgs {
                rename: None,
                krate: Some(parse_quote!(::aither_core)),
            },
            input,
        )
        .expect("expands cleanly")
        .to_string();

        assert!(
            tokens.contains(":: aither_core :: llm :: Tool"),
            "expected aither_core path in: {tokens}"
        );
        assert!(
            !tokens.contains("__hidden"),
            "must not emit __hidden::CowStr: {tokens}"
        );
        assert!(
            tokens.contains("query : String") && !tokens.contains("struct RewriteArgs { mut"),
            "struct field must strip mut: {tokens}"
        );
        assert!(
            tokens.contains("schemars") && tokens.contains("serde"),
            "field attributes must reach RewriteArgs: {tokens}"
        );
    }

    #[test]
    fn manifest_detection_prefers_aither_core_and_handles_renames() {
        let core_manifest = r#"
            [package]
            name = "demo"
            version = "0.1.0"
            [dependencies]
            aither-core = "0.31"
        "#;
        assert_eq!(
            find_runtime_crate_in_manifest(core_manifest).as_deref(),
            Some("::aither_core")
        );

        let facade_manifest = r#"
            [package]
            name = "demo"
            version = "0.1.0"
            [dependencies]
            aither = "0.31"
        "#;
        assert_eq!(
            find_runtime_crate_in_manifest(facade_manifest).as_deref(),
            Some("::aither")
        );

        let renamed_manifest = r#"
            [package]
            name = "demo"
            version = "0.1.0"
            [dependencies]
            core_rt = { package = "aither-core", version = "0.31" }
        "#;
        assert_eq!(
            find_runtime_crate_in_manifest(renamed_manifest).as_deref(),
            Some("::core_rt")
        );
    }
}
