#!/usr/bin/env bash
set -u
set -o pipefail

# ---------------- Configuration ----------------
EVENTS_PER_JOB=500
N_JOBS=2
START_INDEX=2
MAX_RETRIES=3
SLEEP_BETWEEN=5
# ------------------------------------------------

END_INDEX=$(( START_INDEX + N_JOBS - 1 ))

echo "Planned jobs: $N_JOBS (indexes $START_INDEX..$END_INDEX)"
echo "Events per job: $EVENTS_PER_JOB"

for (( I=START_INDEX; I<=END_INDEX; I++ )); do
  OUT="run${I}.h5"

  if [[ -f "$OUT" ]]; then
    echo "Skipping job $I: '$OUT' already exists."
    continue
  fi

  echo "Starting job $I -> $OUT"

  for (( ATTEMPT=1; ATTEMPT<=MAX_RETRIES; ATTEMPT++ )); do
    python one_pmt_trigger.py \
      --max-events "$EVENTS_PER_JOB" \
      --max-time 100000 \
      --output "$OUT"

    STATUS=$?
    if [[ $STATUS -eq 0 ]]; then
      echo "Job $I finished successfully."
      break
    fi

    echo "Job $I failed with exit status $STATUS (attempt $ATTEMPT/$MAX_RETRIES)."
    echo "Removing partial output '$OUT' (if any) and preparing to retry..."
    rm -f -- "$OUT"

    if (( ATTEMPT == MAX_RETRIES )); then
      echo "Giving up on job $I after $ATTEMPT attempts."
      break
    fi

    echo "Retrying job $I in ${SLEEP_BETWEEN}s..."
    sleep "$SLEEP_BETWEEN"
  done
done

