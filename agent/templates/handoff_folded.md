{{ summary }}
{%- if !user_messages.is_empty() %}

## User Messages
Verbatim user messages from earlier in the session, in chronological order:
{% for message in user_messages %}
### User {{ loop.index }}
{{ message }}
{% endfor -%}
{%- endif -%}
{%- if let Some(activity) = post_fork %}

## Activity During Compaction
These turns and tool calls ran in chronological order while the handoff summary above was being generated:

{{ activity }}
{%- endif %}
