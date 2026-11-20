{% if !exports.is_empty() %}{% for (name, value) in exports %}{{ name }}={{ value }}
{% endfor %}export{% for (name, _) in exports %} {{ name }}{% endfor %}
{% endif %}{% if !pre_path.is_empty() %}{{ pre_path }}
{% endif %}PATH={{ bin_dir }}"${PATH:+:$PATH}"
export PATH
{{ script }}