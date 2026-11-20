The conversation up to this point was compacted into `<handoff>` above{% if had_post_fork %}, including the activity that ran while compaction was in flight{% endif %}. Earlier file contents and command outputs are no longer in the window; read any file not shown below before relying on its details.
{%- if let Some(restored) = restored_context %}

<restored_context>
{{ restored }}
</restored_context>
{%- endif %}
