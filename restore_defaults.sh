#!/bin/bash
# =============================================================================
# restore_defaults.sh  (v2 — matches new state dir, kills GPU stress)
# =============================================================================

STATE_DIR="$HOME/.cache/maxperf_state"

echo "=================================================================="
echo " RESTORE DEFAULTS"
echo " state dir: $STATE_DIR"
echo "=================================================================="

# -----------------------------------------------------------------------------
# 1. Kill stress workloads
# -----------------------------------------------------------------------------
echo "[1/6] Stopping stress workloads..."
sudo pkill -f stress-ng    >/dev/null 2>&1 || true
sudo pkill -f gpu_burn     >/dev/null 2>&1 || true
sudo pkill -f gpu_hammer.py >/dev/null 2>&1 || true

for f in "$STATE_DIR/stress.pid" "$STATE_DIR/gpuburn.pid" \
         "$STATE_DIR/gpu_stress.pid"; do
    [ -f "$f" ] && sudo kill -TERM "$(cat "$f")" >/dev/null 2>&1 || true
done
sleep 2
echo "      done."

# -----------------------------------------------------------------------------
# 2. CPU governor + C-states
# -----------------------------------------------------------------------------
echo "[2/6] Restoring CPU governor + C-states..."

DEFAULT_GOV="schedutil"
grep -q "powersave" /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors 2>/dev/null \
    && DEFAULT_GOV="powersave"

for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    cpu=$(basename "$(dirname "$(dirname "$gov")")")
    saved="$STATE_DIR/gov_${cpu}"
    val="$DEFAULT_GOV"
    [ -f "$saved" ] && val=$(cat "$saved")
    echo "$val" | sudo tee "$gov" >/dev/null 2>&1 || true
done

for state in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
    key=$(echo "$state" | tr '/' '_')
    saved="$STATE_DIR/state_${key}"
    val="0"
    [ -f "$saved" ] && val=$(cat "$saved")
    echo "$val" | sudo tee "$state" >/dev/null 2>&1 || true
done

command -v cpupower &>/dev/null && sudo cpupower idle-set -E >/dev/null 2>&1 || true
echo "      done."

# -----------------------------------------------------------------------------
# 3. Disks
# -----------------------------------------------------------------------------
echo "[3/6] Re-enabling disk write caches..."
for dev in /dev/sd[a-z] /dev/nvme[0-9]n[0-9]; do
    [ -b "$dev" ] || continue
    sudo hdparm -W 1 "$dev" >/dev/null 2>&1 || true
    echo "      $(basename "$dev") write-cache on"
done
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true

# -----------------------------------------------------------------------------
# 4. GPU
# -----------------------------------------------------------------------------
echo "[4/6] Restoring GPU defaults..."
sudo nvidia-smi -rgc >/dev/null 2>&1 || true
sudo nvidia-smi -rac >/dev/null 2>&1 || true

RESTORE_PL=""
[ -f "$STATE_DIR/gpu_power" ] && \
    RESTORE_PL=$(grep -oE '[0-9]+(\.[0-9]+)?' "$STATE_DIR/gpu_power" | head -1)
if [ -z "$RESTORE_PL" ] || [ "$RESTORE_PL" = "N/A" ]; then
    RESTORE_PL=$(nvidia-smi -q -d POWER 2>/dev/null \
                 | grep "Default Power Limit" | head -1 | awk '{print $(NF-1)}')
fi
if [ -n "$RESTORE_PL" ] && [ "$RESTORE_PL" != "N/A" ]; then
    sudo nvidia-smi -pl "$RESTORE_PL" >/dev/null 2>&1 \
        && echo "      power limit -> ${RESTORE_PL} W"
fi

sudo nvidia-smi -pm 0 >/dev/null 2>&1 || true

NS_BIN=""
for cand in /usr/bin/nvidia-settings /usr/lib/nvidia/nvidia-settings; do
    [ -x "$cand" ] && NS_BIN="$cand" && break
done
[ -n "$NS_BIN" ] && [ -n "$DISPLAY" ] && \
    env -u LD_LIBRARY_PATH -u LD_PRELOAD "$NS_BIN" \
        -a "[gpu:0]/GPUFanControlState=0" >/dev/null 2>&1 || true
echo "      done."

# -----------------------------------------------------------------------------
# 5. Dell EC fans
# -----------------------------------------------------------------------------
echo "[5/6] Restoring Dell EC fan control..."
for d in /sys/class/hwmon/hwmon*; do
    [ -f "$d/name" ] || continue
    if grep -q "dell_smm" "$d/name" 2>/dev/null; then
        if [ -e "$d/pwm1_enable" ]; then
            echo 2 | sudo tee "$d/pwm1_enable" >/dev/null 2>&1 || true
            for i in 2 3 4 5 6 7 8; do
                [ -e "$d/pwm${i}_enable" ] && \
                    echo 2 | sudo tee "$d/pwm${i}_enable" >/dev/null 2>&1
            done
            echo "      Dell fans -> automatic (pwm1_enable=2)"
        fi
        break
    fi
done

# -----------------------------------------------------------------------------
# 6. RAPL
# -----------------------------------------------------------------------------
echo "[6/6] Restoring Intel RAPL limits..."
for rapl in /sys/class/powercap/intel-rapl/intel-rapl:*/constraint_*_limit; do
    [ -f "$rapl" ] || continue
    saved="$STATE_DIR/rapl_$(echo "$rapl" | tr '/' '_')"
    [ -f "$saved" ] && cat "$saved" | sudo tee "$rapl" >/dev/null 2>&1
done
echo "      done."

cat <<EOF

==================================================================
 RESTORE COMPLETE
------------------------------------------------------------------
 Reboot recommended to fully reset RAPL + driver state.
 Clean up saved state: rm -rf $STATE_DIR
==================================================================
EOF
