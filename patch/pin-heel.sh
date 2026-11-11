#!/usr/bin/env bash
# pin-heel.sh — pin the heel executable of one release in the tool
# distribution (aither #664).
#
# Usage:
#   patch/pin-heel.sh <version>
#
# <version> is a heel release version such as 0.9.1, published as the release
# `heel-v<version>` of May-Agent/heel. Every platform asset of the release is
# downloaded into a fresh directory and hashed; the hash must equal the digest
# GitHub recorded when the asset was uploaded, or nothing is written. The
# `heel` entries of the `host_programs` in patch/third-party-tools.json are
# then replaced by one entry per asset, with its URL, SHA-256 and size.
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
root="$(cd "$(dirname "$0")/.." && pwd)"
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
    entries="$(jq -cn -L "$root/patch" --argjson all "$entries" --argjson asset "$asset" \
        --arg version "$version" --arg sha256 "$sha256" '
        include "targets";
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

jq --argjson heel "$entries" '
    .host_programs = ([.host_programs[] | select(.tool != "heel")]
        + ($heel | sort_by(.target.os, .target.arch)))
' "$root/patch/third-party-tools.json" > "$work/pinned.json"
cp "$work/pinned.json" "$root/patch/third-party-tools.json"
echo "pinned heel ${version} for $(jq length <<<"$entries") platforms" >&2
