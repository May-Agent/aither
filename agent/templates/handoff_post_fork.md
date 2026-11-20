{%- for entry in entries -%}
{%- match entry -%}
{%- when PostForkEntry::User with { content, attachment_count } -%}
- **User**{% if *attachment_count > 0 %} ({{ attachment_count }} attachment(s) retained){% endif %}:
{{ content }}
{% when PostForkEntry::Assistant with { content, calls } -%}
{%- if !content.is_empty() -%}
- **Assistant**:
{{ content }}
{% endif -%}
{%- for call in calls -%}
- **Tool call** `{{ call.name }}` (`{{ call.id }}`): `{{ call.arguments }}`
{% endfor -%}
{%- when PostForkEntry::Tool with { call_id, content } -%}
- **Tool result** (`{{ call_id }}`):
{{ content }}
{% endmatch -%}
{%- endfor -%}
