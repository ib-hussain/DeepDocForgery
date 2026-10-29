#!/bin/bash
# =============================================================================
# max_performance.sh  (v2 — fixes state-dir perms, adds GPU load, verifies fans)
# =============================================================================

# Deliberately NOT using `set -e` — best-effort.

# --- State dir: per-user, always writable -------------------------------------
STATE_DIR="$HOME/.cache/maxperf_state"
mkdir -p "$STATE_DIR"
chmod 700 "$STATE_DIR"
# Drop a sudo-owned marker so restore can chown back if needed
touch "$STATE_DIR/.init" 2>/dev/null || true

echo "=================================================================="
echo " MAX PERFORMANCE SCRIPT"
echo " state dir: $STATE_DIR"
echo "=================================================================="

# -----------------------------------------------------------------------------
# 1. CPU — governor + C-states
# -----------------------------------------------------------------------------
echo "[1/6] CPU: performance governor + disabling C-states..."

for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    cpu=$(basename "$(dirname "$(dirname "$gov")")")
    cat "$gov" 2>/dev/null | tee "$STATE_DIR/gov_${cpu}" > /dev/null
    echo "performance" | sudo tee "$gov" > /dev/null 2>&1 || true
done

for state in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
    key=$(echo "$state" | tr '/' '_')
    cat "$state" 2>/dev/null | tee "$STATE_DIR/state_${key}" > /dev/null
    echo "1" | sudo tee "$state" > /dev/null 2>&1 || true
done

command -v cpupower &>/dev/null && {
    sudo cpupower idle-set -D 0                 >/dev/null 2>&1 || true
    sudo cpupower frequency-set -g performance  >/dev/null 2>&1 || true
}
echo "      done."

# -----------------------------------------------------------------------------
# 2. Disks — disable write cache
# -----------------------------------------------------------------------------
echo "[2/6] Disks: disabling write cache..."
for dev in /dev/sd[a-z] /dev/nvme[0-9]n[0-9]; do
    [ -b "$dev" ] || continue
    hdparm -W "$dev" 2>/dev/null | grep -E "write-caching" \
        > "$STATE_DIR/hdparm_$(basename "$dev")" 2>/dev/null
    sudo hdparm -W 0 "$dev" > /dev/null 2>&1 || true
    echo "      $(basename "$dev") write-cache off"
done

# -----------------------------------------------------------------------------
# 3. GPU — power, persistence, clocks
# -----------------------------------------------------------------------------
echo "[3/6] GPU: power limit + persistence + clocks..."

sudo nvidia-smi -pm 1 > /dev/null 2>&1 || true

MAX_POWER=$(nvidia-smi -q -d POWER 2>/dev/null \
            | grep "Max Power Limit" | head -1 | awk '{print $(NF-1)}')
CUR_POWER=$(nvidia-smi -q -d POWER 2>/dev/null \
            | grep "Current Power Limit" | head -1 | awk '{print $(NF-1)}')

echo "current_power=$CUR_POWER" > "$STATE_DIR/gpu_power"

if [ -n "$MAX_POWER" ] && [ "$MAX_POWER" != "N/A" ]; then
    sudo nvidia-smi -pl "$MAX_POWER" > /dev/null 2>&1 \
        && echo "      GPU power limit -> ${MAX_POWER} W" \
        || echo "      GPU already at max power (${MAX_POWER} W)"
else
    echo "      WARNING: no max power value found"
fi

sudo nvidia-smi -lgc 0,999999 > /dev/null 2>&1 \
    && echo "      GPU clocks locked to max" \
    || echo "      NOTE: -lgc unsupported on this GPU (typical for laptop Ada)"

# GPU fans — best effort via nvidia-settings, expected to fail on laptop Ada
NS_BIN=""
for cand in /usr/bin/nvidia-settings /usr/lib/nvidia/nvidia-settings; do
    [ -x "$cand" ] && NS_BIN="$cand" && break
done
if [ -n "$NS_BIN" ] && [ -n "$DISPLAY" ]; then
    if env -u LD_LIBRARY_PATH -u LD_PRELOAD "$NS_BIN" -q fans >/dev/null 2>&1; then
        env -u LD_LIBRARY_PATH -u LD_PRELOAD "$NS_BIN" \
            -a "[gpu:0]/GPUFanControlState=1" \
            -a "[fan:0]/GPUTargetFanSpeed=100" >/dev/null 2>&1 \
            && echo "      GPU fan forced to 100%" \
            || echo "      GPU fan set rejected"
    else
        echo "      NOTE: nvidia-settings cannot control fans on laptop Ada."
        echo "            GPU fan curve is firmware-managed."
    fi
fi

# -----------------------------------------------------------------------------
# 4. Dell EC fans via dell-smm-hwmon
# -----------------------------------------------------------------------------
echo "[4/6] Dell EC fans: forcing to 100%..."

# Load module if not already
if ! grep -q "^dell_smm" /proc/modules 2>/dev/null; then
    sudo modprobe dell-smm-hwmon restricted=0 force=1 2>/dev/null || true
    sleep 1
fi

DELL_HWMON=""
for d in /sys/class/hwmon/hwmon*; do
    [ -f "$d/name" ] || continue
    if grep -q "dell_smm" "$d/name" 2>/dev/null; then
        DELL_HWMON="$d"; break
    fi
done

FANS_FORCED=0
if [ -n "$DELL_HWMON" ] && [ -e "$DELL_HWMON/pwm1" ]; then
    # Save original enable mode
    cat "$DELL_HWMON/pwm1_enable" 2>/dev/null \
        | tee "$STATE_DIR/dell_pwm_enable" > /dev/null 2>&1

    # Take manual control (mode 1) and drive 100% (255)
    echo 1   | sudo tee "$DELL_HWMON/pwm1_enable" > /dev/null 2>&1 || true
    echo 255 | sudo tee "$DELL_HWMON/pwm1"        > /dev/null 2>&1 || true
    for i in 2 3 4 5 6 7 8; do
        [ -e "$DELL_HWMON/pwm$i" ]            && echo 255 | sudo tee "$DELL_HWMON/pwm$i"        >/dev/null 2>&1
        [ -e "$DELL_HWMON/pwm${i}_enable" ]   && echo 1   | sudo tee "$DELL_HWMON/pwm${i}_enable" >/dev/null 2>&1
    done
    FANS_FORCED=1

    # Verify fans are actually spinning up (wait, then sample)
    echo "      waiting 8s for fans to ramp, then sampling RPM..."
    sleep 8
    echo "      --- RPM after 8s ---"
    sensors 2>/dev/null | grep -iE "fan[0-9]|Fan" | sed 's/^/      /'
else
    echo "      NOTE: no dell_smm PWM interface found."
fi

# -----------------------------------------------------------------------------
# 5. Stress — CPU, VM, I/O, GPU
# -----------------------------------------------------------------------------
# echo "[5/6] Launching stress workloads..."

# if ! command -v stress-ng &>/dev/null; then
#     sudo apt-get install -y stress-ng > /dev/null 2>&1 || true
# fi

# # ---- CPU / memory / I/O ----
# nohup stress-ng \
#     --cpu 0   --cpu-method all \
#     --vm  0   --vm-bytes 90% --vm-method all \
#     --io  4 \
#     --hdd 4   --hdd-bytes 2G \
#     --timeout 0 \
#     > "$STATE_DIR/stress.log" 2>&1 &
# STRESS_PID=$!
# echo "$STRESS_PID" > "$STATE_DIR/stress.pid"
# echo "      stress-ng PID = $STRESS_PID  (log: $STATE_DIR/stress.log)"

# # ---- GPU load: three-way fallback ----
# GPU_STRESS_PID=""

# # (a) gpu_burn binary
# if command -v gpu_burn &>/dev/null; then
#     nohup gpu_burn 0 > "$STATE_DIR/gpuburn.log" 2>&1 &
#     GPU_STRESS_PID=$!
#     echo "      gpu_burn launched (PID $GPU_STRESS_PID)"

# # (b) stress-ng --cuda
# elif stress-ng --help 2>&1 | grep -q -- "--cuda"; then
#     nohup stress-ng --cuda 1 --timeout 0 > "$STATE_DIR/gpu_stress.log" 2>&1 &
#     GPU_STRESS_PID=$!
#     echo "      stress-ng --cuda launched (PID $GPU_STRESS_PID)"

# # (c) PyTorch matmul loop
# elif python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
#     cat > "$STATE_DIR/gpu_hammer.py" <<'PYEOF'
# import torch, time
# a = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)
# b = torch.randn(8192, 8192, device='cuda', dtype=torch.float16)
# torch.cuda.synchronize()
# while True:
#     for _ in range(50):
#         a = a @ b
#     torch.cuda.synchronize()
# PYEOF
#     nohup python3 "$STATE_DIR/gpu_hammer.py" > "$STATE_DIR/gpu_stress.log" 2>&1 &
#     GPU_STRESS_PID=$!
#     echo "      PyTorch GPU hammer launched (PID $GPU_STRESS_PID)"

# else
#     echo "      WARNING: no GPU stress method available."
#     echo "               Install gpu-burn: https://github.com/wilicc/gpu-burn"
#     echo "               Or in this venv: pip install torch"
# fi

# [ -n "$GPU_STRESS_PID" ] && echo "$GPU_STRESS_PID" > "$STATE_DIR/gpu_stress.pid"

# -----------------------------------------------------------------------------
# 6. RAPL — remove package power caps
# -----------------------------------------------------------------------------
echo "[6/6] Intel RAPL package power limits..."
for rapl in /sys/class/powercap/intel-rapl/intel-rapl:*/constraint_*_limit; do
    [ -f "$rapl" ] || continue
    cat "$rapl" 2>/dev/null \
        | tee "$STATE_DIR/rapl_$(echo "$rapl" | tr '/' '_')" > /dev/null
    maxfile="${rapl%_limit}_max_power_uw"
    if [ -f "$maxfile" ]; then
        MAX_UW=$(cat "$maxfile" 2>/dev/null || echo 0)
        [ "${MAX_UW:-0}" -gt 0 ] 2>/dev/null \
            && echo "$MAX_UW" | sudo tee "$rapl" >/dev/null 2>&1 || true
    fi
done
echo "      done."

# -----------------------------------------------------------------------------
cat <<EOF

==================================================================
 MAX PERFORMANCE ACTIVE
------------------------------------------------------------------
 stress-ng PID : $STRESS_PID
 GPU stress PID: ${GPU_STRESS_PID:-<none>}
 state file    : $STATE_DIR/
 monitor       : watch -n 2 'sensors; nvidia-smi'
------------------------------------------------------------------
 Verify:
   nvidia-smi                     # GPU-Util should climb
   top -bn1 | head -20            # CPU should be ~100% each core
   sensors | grep -i fan          # fans should ramp
------------------------------------------------------------------
 Stop everything:
   bash temp/restore_defaults.sh
==================================================================
EOF
