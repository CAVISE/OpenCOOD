#!/usr/bin/env bash

set -euo pipefail

readonly artifact_root="${OPENCOOD_ARTIFACT_ROOT:-/opt/opencood-artifacts}"
readonly workspace="${OPENCOOD_WORKSPACE:-${HOME}/cavise/opencood}"
readonly native_components="${OPENCOOD_NATIVE_COMPONENTS:-}"
readonly cuda_destination="${workspace}/opencood/pcdet_utils"

sync_cuda_artifacts() {
    local component_root="${artifact_root}/cuda"
    local manifest="${component_root}/cuda-artifacts.manifest"
    local relative_path
    local destination_path
    local -a artifact_paths
    local -a installed_artifacts

    if [[ ! -s "${manifest}" ]]; then
        echo "Missing or empty CUDA artifact manifest: ${manifest}" >&2
        exit 1
    fi

    mapfile -t artifact_paths < <(sed "/^[[:space:]]*$/d" "${manifest}")
    mapfile -d "" installed_artifacts < <(
        find "${component_root}" -type f -name "*_cuda*.so" -print0
    )

    if [[ "${#artifact_paths[@]}" -eq 0 || \
          "${#artifact_paths[@]}" -ne "${#installed_artifacts[@]}" ]]; then
        echo "CUDA artifact count does not match its manifest" >&2
        exit 1
    fi

    if [[ -d "${cuda_destination}" ]]; then
        find "${cuda_destination}" -type f -name "*_cuda*.so" -delete
    fi

    for relative_path in "${artifact_paths[@]}"; do
        if [[ "${relative_path}" == /* || "${relative_path}" == ".." || \
              "${relative_path}" == ../* || "${relative_path}" == */../* || \
              "${relative_path}" == */.. || \
              ! -f "${component_root}/${relative_path}" ]]; then
            echo "Invalid or missing CUDA artifact: ${relative_path}" >&2
            exit 1
        fi

        destination_path="${relative_path#opencood/}"
        mkdir -p "${workspace}/${destination_path%/*}"
        cp "${component_root}/${relative_path}" "${workspace}/${destination_path}"
    done
}

declare -a components=()
if [[ -n "${native_components}" ]]; then
    read -r -a components <<< "${native_components}"
fi

for component in "${components[@]}"; do
    if [[ "${component}" != "cuda" ]]; then
        echo "Unsupported OpenCOOD native component: ${component}" >&2
        exit 1
    fi
    sync_cuda_artifacts
done

exec "$@"
