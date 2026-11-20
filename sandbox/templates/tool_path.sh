{% if !exports.is_empty() %}{% for (name, value) in exports %}{{ name }}={{ value }}
{% endfor %}export{% for (name, _) in exports %} {{ name }}{% endfor %}
{% endif %}{{ pre_path }}PATH={{ bin_dir }}"${PATH:+:$PATH}"
export PATH
{{ script }}