set -uo pipefail  
cd "$(dirname "$0")" || exit 1

PY="${BR2IDGE_PYTHON:-python}"
LOG_DIR="logs"
SKIP_CHECK=0
[ "${1:-}" = "--skip-check" ] && SKIP_CHECK=1

CONFIGS=(
    "beir_all_recs_query_as_user.json"
    "beir_all_recs_retrieval_as_user.json"
)


fail() { printf '\n[erro] %s\n' "$1" >&2; exit 1; }

for cfg in "${CONFIGS[@]}"; do
    [ -f "$cfg" ] || fail "config não encontrado: $cfg (rode a partir da raiz do repositório)"
done

command -v "$PY" >/dev/null 2>&1 || fail "python não encontrado: '$PY'. Ative o ambiente ou defina BR2IDGE_PYTHON."

if [ "$SKIP_CHECK" -eq 0 ]; then
    echo "[check] Verificando ambiente com $($PY -V 2>&1)... (importar tensorflow leva alguns segundos)"

    missing=$("$PY" -c '
import sys
need = ["numpy", "pandas", "sklearn", "tensorflow", "lightfm", "libreco", "recbole"]
bad = []
for m in need:
    try:
        __import__(m)
    except Exception:
        bad.append(m)
print(" ".join(bad))
' 2>/dev/null)

    if [ -n "${missing:-}" ]; then
        fail "dependências ausentes: ${missing}
      Crie o ambiente antes de rodar:
        conda create -n br2idge python=3.10 && conda activate br2idge
        python -m pip install \"pip<=25.2\" && pip install -r requirements.txt
      Para ignorar esta checagem: ./run_beir_cross_task.sh --skip-check"
    fi
    echo "[check] Dependências OK."
fi

mkdir -p "$LOG_DIR"


declare -a STATUS_LINES
overall=0
started_at=$(date +%s)

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    name=$(basename "$cfg" .json)
    stamp=$(date +%Y%m%d_%H%M%S)
    log="$LOG_DIR/${name}_${stamp}.log"

    printf '\n%s\n' "======================================================================"
    printf '[%d/%d] %s\n' "$((i + 1))" "${#CONFIGS[@]}" "$cfg"
    printf 'log: %s\n' "$log"
    printf '%s\n\n' "======================================================================"

    t0=$(date +%s)
    "$PY" framework.py --config "$cfg" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    elapsed=$(( $(date +%s) - t0 ))

    if [ "$rc" -eq 0 ]; then
        STATUS_LINES+=("  OK      ${name}  (${elapsed}s)  ${log}")
        printf '\n[%s] concluído em %ds\n' "$name" "$elapsed"
    else
        STATUS_LINES+=("  FALHOU  ${name}  (${elapsed}s, exit ${rc})  ${log}")
        printf '\n[%s] FALHOU após %ds (exit %d) — seguindo para o próximo\n' "$name" "$elapsed" "$rc"
        overall=1
    fi
done


printf '\n%s\n' "======================================================================"
printf 'RESUMO (total: %ds)\n' "$(( $(date +%s) - started_at ))"
printf '%s\n' "======================================================================"
printf '%s\n' "${STATUS_LINES[@]}"
printf '\nResultados: experimental_results/beir_nfcorpus/recs/\n'
printf 'Artefatos:  artifacts/beir_nfcorpus/<modelo>/<hash>/runs/<run_id>/\n'

exit "$overall"
