These files and skills were named by the handoff above and have been restored for you. Do not read them again unless you need a part that was truncated.
{% if !files.is_empty() %}

{% include "external_notice.txt" %}
{% endif %}
{%- for file in files -%}
{{ file }}
{%- endfor -%}
{%- for skill in skills -%}
{{ skill }}
{%- endfor -%}
{% if !unavailable.is_empty() %}

### Named but not restored

The handoff also named these entries; they were not inlined. Read or re-request them on demand.
{% for entry in unavailable %}
- `{{ entry.name }}`: {{ entry.reason }}
{% endfor %}
{% endif %}