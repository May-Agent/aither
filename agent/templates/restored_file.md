

### {{ path }}{% if truncated %} (truncated){% endif %}
{% if let Some(why) = why %}
_{{ why }}_
{% endif %}

{{ bounded_content }}