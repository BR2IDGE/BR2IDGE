set -uo pipefail
cd "$(dirname "$0")" || exit 1

PY="${BR2IDGE_PYTHON:-python}"
LOG_DIR="logs"
WORK_DIR=".run_configs"
DATASET="beir_nfcorpus"

ALL_MODELS=(ItemKNNModel LightFMModel DeepFMModel LightGCNModel Bert4REC)
ALL_STRATEGIES=(query retrieval)

declare -A CONFIG=(
    [query]="beir_all_recs_query_as_user.json"
    [retrieval]="beir_all_recs_retrieval_as_user.json"
)
declare -A EXPERIMENT=(
    [query]="BEIR_NFCorpus_AllRecs_QueryAsUser"
    [retrieval]="BEIR_NFCorpus_AllRecs_RetrievalAsUser"
)

MODELS=("${ALL_MODELS[@]}")
STRATEGIES=("${ALL_STRATEGIES[@]}")
SKIP_CHECK=0
FRESH=0
COLLECT_ONLY=0

fail() { printf '\n[erro] %s\n' "$1" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-check)  SKIP_CHECK=1 ;;
        --fresh)       FRESH=1 ;;
        --collect-only) COLLECT_ONLY=1 ;;
        --models)
            shift; [ $# -gt 0 ] || fail "--models exige uma lista"
            IFS=, read -r -a MODELS <<< "$1" ;;
        --strategies)
            shift; [ $# -gt 0 ] || fail "--strategies exige uma lista"
            IFS=, read -r -a STRATEGIES <<< "$1" ;;
        *) fail "argumento desconhecido: $1" ;;
    esac
    shift
done

for m in "${MODELS[@]}"; do
    found=0
    for a in "${ALL_MODELS[@]}"; do [ "$m" = "$a" ] && found=1; done
    [ "$found" -eq 1 ] || fail "modelo desconhecido: '$m' (validos: ${ALL_MODELS[*]})"
done
for s in "${STRATEGIES[@]}"; do
    [ -n "${CONFIG[$s]:-}" ] || fail "estrategia desconhecida: '$s' (validas: ${ALL_STRATEGIES[*]})"
    [ -f "${CONFIG[$s]}" ] || fail "config nao encontrado: ${CONFIG[$s]}"
done

command -v "$PY" >/dev/null 2>&1 || fail "python nao encontrado: '$PY'. Ative o ambiente ou defina BR2IDGE_PYTHON."

if [ "$SKIP_CHECK" -eq 0 ] && [ "$COLLECT_ONLY" -eq 0 ]; then
    echo "[check] Verificando ambiente com $($PY -V 2>&1)... (importar tensorflow leva alguns segundos)"
    missing=$("$PY" -c '
need = ["numpy", "pandas", "scipy", "sklearn", "tensorflow", "lightfm", "libreco", "recbole", "torch"]
bad = []
for m in need:
    try:
        __import__(m)
    except Exception:
        bad.append(m)
print(" ".join(bad))
' 2>/dev/null)
    if [ -n "${missing:-}" ]; then
        fail "dependencias ausentes: ${missing}
      Ative o ambiente do projeto, ou use --skip-check para ignorar."
    fi
    echo "[check] Dependencias OK."

    "$PY" -c '
import torch
if not torch.cuda.is_available():
    print("[check] CUDA indisponivel: Bert4REC e LightGCN vao rodar em CPU.")
else:
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    archs = torch.cuda.get_arch_list()
    sm = f"sm_{cap[0]}{cap[1]}"
    print(f"[check] GPU: {name} ({sm}) | build do torch: {archs}")
    if sm not in archs:
        print(f"[check] AVISO: {sm} nao esta na build do torch. O Bert4REC vai")
        print( "        quebrar no treino mesmo com cuda.is_available()==True.")
        print( "        Contorne com: CUDA_VISIBLE_DEVICES=\"\" ./run_beir_all.sh")
' 2>/dev/null || echo "[check] nao foi possivel inspecionar a GPU."
fi

if [ "$FRESH" -eq 1 ]; then
    for s in "${STRATEGIES[@]}"; do
        exp="${EXPERIMENT[$s]}"
        d="artifacts/${DATASET}/${exp}"
        [ -d "$d" ] && { echo "[fresh] removendo $d"; rm -rf "$d"; }
        find "experimental_results/${DATASET}" -maxdepth 2 -name "${exp}*" 2>/dev/null |
            while read -r f; do echo "[fresh] removendo $f"; rm -rf "$f"; done
    done
    for m in "${MODELS[@]}"; do
        d="artifacts/${DATASET}/${m}"
        [ -d "$d" ] && { echo "[fresh] removendo $d"; rm -rf "$d"; }
    done
    echo "[fresh] saidas anteriores removidas."
fi

mkdir -p "$LOG_DIR" "$WORK_DIR"

declare -a STATUS_LINES
overall=0
started_at=$(date +%s)
total=$(( ${#STRATEGIES[@]} * ${#MODELS[@]} ))
step=0

if [ "$COLLECT_ONLY" -eq 0 ]; then
    n_runs=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['execution']['n_runs'])" "${CONFIG[${STRATEGIES[0]}]}")
    printf '\n%s\n' "======================================================================"
    printf 'BEIR/NFCorpus | %d estrategia(s) x %d modelo(s) x %s execucoes = %d invocacoes\n' \
        "${#STRATEGIES[@]}" "${#MODELS[@]}" "$n_runs" "$total"
    printf 'modelos: %s\n' "${MODELS[*]}"
    printf '%s\n' "======================================================================"

    for s in "${STRATEGIES[@]}"; do
        base="${CONFIG[$s]}"
        exp="${EXPERIMENT[$s]}"

        for m in "${MODELS[@]}"; do
            step=$((step + 1))
            derived="${WORK_DIR}/$(basename "$base" .json)__${m}.json"
            "$PY" - "$base" "$m" "$derived" <<'PYEOF'
import json, sys
src, model, dst = sys.argv[1:4]
cfg = json.load(open(src, encoding="utf-8"))
cfg["model"] = [{"name": model}]
with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
PYEOF
            [ -f "$derived" ] || { STATUS_LINES+=("  FALHOU  ${s}/${m}  (config derivado nao gerado)"); overall=1; continue; }

            stamp=$(date +%Y%m%d_%H%M%S)
            log="$LOG_DIR/${exp}__${m}_${stamp}.log"

            printf '\n%s\n' "----------------------------------------------------------------------"
            printf '[%d/%d] %s | %s\n' "$step" "$total" "$s" "$m"
            printf 'log: %s\n' "$log"
            printf '%s\n\n' "----------------------------------------------------------------------"

            t0=$(date +%s)
            "$PY" framework.py --config "$derived" 2>&1 | tee "$log"
            rc=${PIPESTATUS[0]}
            elapsed=$(( $(date +%s) - t0 ))

            if [ "$rc" -eq 0 ]; then
                STATUS_LINES+=("  OK      ${s}/${m}  (${elapsed}s)")
                printf '\n[%s/%s] concluido em %ds\n' "$s" "$m" "$elapsed"
            else
                STATUS_LINES+=("  FALHOU  ${s}/${m}  (${elapsed}s, exit ${rc})  ${log}")
                printf '\n[%s/%s] FALHOU apos %ds (exit %d) - seguindo para o proximo\n' "$s" "$m" "$elapsed" "$rc"
                overall=1
            fi
        done
    done
fi

printf '\n%s\n' "======================================================================"
printf 'AGREGANDO\n'
printf '%s\n' "======================================================================"
for s in "${STRATEGIES[@]}"; do
    printf '\n[%s] %s\n' "$s" "${EXPERIMENT[$s]}"
    "$PY" collect_runs.py "${EXPERIMENT[$s]}" --dataset "$DATASET" --stats || overall=1
done

printf '\n%s\n' "======================================================================"
printf 'RESUMO (total: %ds)\n' "$(( $(date +%s) - started_at ))"
printf '%s\n' "======================================================================"
if [ "${#STATUS_LINES[@]}" -gt 0 ]; then
    printf '%s\n' "${STATUS_LINES[@]}"
fi
printf '\nTabelas:   experimental_results/%s/recs/\n' "$DATASET"
printf 'Artefatos: artifacts/%s/<modelo>/<hash>/runs/<run_id>/\n' "$DATASET"
printf 'Logs:      %s/\n' "$LOG_DIR"
if [ "$overall" -ne 0 ]; then
    printf '\nAlgum modelo falhou. Para re-rodar so ele:\n'
    printf '  ./run_beir_all.sh --models <Modelo> --strategies <query|retrieval>\n'
fi

exit "$overall"
