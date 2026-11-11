PATH={{ bin_dir }}"${PATH:+:$PATH}"
export PATH
{% if !exports.is_empty() %}{% for (name, value) in exports %}{{ name }}={{ value }}
{% endfor %}export{% for (name, _) in exports %} {{ name }}{% endfor %}
{% endif %}{{ script }}
