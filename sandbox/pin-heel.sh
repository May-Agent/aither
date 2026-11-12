#!/usr/bin/env bash
# pin-heel.sh — pin the heel executable of one release for aither-sandbox's
# tests (aither #664).
#
# Usage:
#   sandbox/pin-heel.sh <version>
#
# <version> is a heel release version such as 0.9.1, published as the release
# `heel-v<version>` of May-Agent/heel. Every platform asset of the release is
# downloaded into a fresh directory and hashed; the hash must equal the digest
# GitHub recorded when the asset was uploaded, or nothing is written. The
# `host_programs` in sandbox/test-heel.json are then replaced by one entry per
# asset, with its URL, SHA-256 and size.
# `gh` authenticates through GH_TOKEN.
#
# The pinned version must be the version of the `heel` library in Cargo.lock:
# aither-sandbox's tests run the pinned heel through `HeelBinary::verify`, so
# `terminal::builder::tests::the_shell_is_built_on_the_sandboxs_directory_and_live_grants`
# fails otherwise, and a heel upgrade moves the library and this pin together.
set -euo pipefail

version="${1:?usage: pin-heel.sh <version>}"
repo="May-Agent/heel"
tag="heel-v${version}"
pin_file="$(cd "$(dirname "$0")" && pwd)/test-heel.json"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

gh release view "$tag" -R "$repo" --json assets \
    --jq '[.assets[] | select(.name | startswith("heel-")) | {name, url, size, digest}]' \
    > "$work/assets.json"
[ "$(jq length "$work/assets.json")" -gt 0 ] || { echo "no heel assets in $tag" >&2; exit 1; }

mkdir "$work/download"
entries='[]'
while IFS= read -r asset; do
    name="$(jq -r '.name' <<<"$asset")"
    gh release download "$tag" -R "$repo" -p "$name" -D "$work/download"
    sha256="$(shasum -a 256 "$work/download/$name" | cut -d' ' -f1)"
    size="$(wc -c < "$work/download/$name" | tr -d ' ')"
    [ "sha256:$sha256" = "$(jq -r '.digest' <<<"$asset")" ] \
        || { echo "$name: downloaded sha256:$sha256 differs from the release digest" >&2; exit 1; }
    [ "$size" = "$(jq -r '.size' <<<"$asset")" ] \
        || { echo "$name: downloaded $size bytes differ from the release size" >&2; exit 1; }
    entries="$(jq -cn --argjson all "$entries" --argjson asset "$asset" \
        --arg version "$version" --arg sha256 "$sha256" '
        # The platform a Rust target triple names: arch is the triple'"'"'s first
        # component, os is derived from the rest.
        def os_of: if endswith("-apple-darwin") then "macos"
                   elif test("linux") then "linux"
                   elif test("windows") then "windows"
                   else error("cannot map target to an os") end;
        def platform_of: {os: os_of, arch: split("-")[0]};
        # heel-<target triple>[.exe]
        ($asset.name | sub("^heel-"; "") | sub("\\.exe$"; "")) as $triple
        | $triple | platform_of as $platform
        | $all + [{
            tool: "heel",
            version: $version,
            target: $platform,
            url: $asset.url,
            sha256: $sha256,
            size: $asset.size,
            encoding: {format: "executable"},
            executable_sha256: $sha256
        }]')"
done < <(jq -c '.[]' "$work/assets.json")

jq -n --argjson heel "$entries" '
    {host_programs: ($heel | sort_by(.target.os, .target.arch))}
' > "$work/pinned.json"
cp "$work/pinned.json" "$pin_file"
echo "pinned heel ${version} for $(jq length <<<"$entries") platforms" >&2
