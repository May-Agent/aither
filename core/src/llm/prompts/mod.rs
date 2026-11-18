use alloc::string::String;
use askama::Template;

#[derive(Template)]
#[template(path = "structured_generate.txt", escape = "none")]
struct StructuredGenerateTemplate<'a> {
    schema: &'a str,
}

pub fn generate(schema: &str) -> String {
    StructuredGenerateTemplate { schema }
        .render()
        .expect("structured generate template must render")
}
