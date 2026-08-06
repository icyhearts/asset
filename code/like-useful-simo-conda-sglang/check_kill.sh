#!/usr/bin/env bash

# Read-only readiness check for tracing senders of native 64-bit kill(2) calls.

ARCH_FILTER="b64"
OVERALL_FAIL=0
OVERALL_UNKNOWN=0
ROOT_MODE=0
NEED_LIVE_RULE=0
NEED_PERSISTENT_RULE=0

AUDITD_BIN=""
AUDITCTL_BIN=""
AUSEARCH_BIN=""
SYSTEMCTL_BIN=""
JOURNALCTL_BIN=""

pass() {
    printf '[PASS] %s\n' "$*"
}

warn() {
    printf '[WARN] %s\n' "$*"
}

fail() {
    printf '[FAIL] %s\n' "$*"
}

unknown() {
    printf '[UNKNOWN] %s\n' "$*"
}

info() {
    printf '[INFO] %s\n' "$*"
}

record_fail() {
    fail "$@"
    OVERALL_FAIL=1
}

record_unknown() {
    unknown "$@"
    OVERALL_UNKNOWN=1
}

trim_value() {
    local value=$1

    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    TRIMMED_VALUE=$value
}

locate_command() {
    local candidate
    local name=$1

    candidate=$(command -v -- "$name" 2>/dev/null) || candidate=""
    if [[ -n $candidate && -x $candidate ]]; then
        printf '%s\n' "$candidate"
        return 0
    fi

    for candidate in \
        "/usr/sbin/$name" \
        "/sbin/$name" \
        "/usr/bin/$name" \
        "/bin/$name"; do
        if [[ -x $candidate ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

usage() {
    info "Usage: ${0##*/} [-h|--help]"
    info "Read-only check for native b64 kill(2) syscall audit readiness."
    info "The script does not install packages, change audit rules, or invoke sudo."
    info "Exit codes: 0=READY, 1=NOT_READY/PARTIAL, 2=UNKNOWN, 64=usage error."
}

# Sets RULE_CLASS to full, partial, or none. A key labels an event but does not
# restrict matching, so -k and -F key= are allowed on a full-coverage rule.
classify_kill_rule() {
    local rule=$1
    local -a tokens=()
    local -a syscalls=()
    local -a extras=()
    local action_value=""
    local filter_value=""
    local syscall_value=""
    local token=""
    local syscall=""
    local lower=""
    local i=0
    local has_always_exit=0
    local has_b64=0
    local has_kill=0

    RULE_CLASS="none"
    RULE_EXTRA_DESCRIPTION=""
    read -r -a tokens <<< "$rule"

    while ((i < ${#tokens[@]})); do
        token=${tokens[$i]}
        case "$token" in
            -a|-A)
                if ((i + 1 < ${#tokens[@]})); then
                    i=$((i + 1))
                    action_value=${tokens[$i],,}
                    if [[ $action_value == "always,exit" || $action_value == "exit,always" ]]; then
                        has_always_exit=1
                    fi
                else
                    extras+=("$token (missing value)")
                fi
                ;;
            -a*|-A*)
                action_value=${token:2}
                action_value=${action_value,,}
                if [[ $action_value == "always,exit" || $action_value == "exit,always" ]]; then
                    has_always_exit=1
                else
                    extras+=("$token")
                fi
                ;;
            -F)
                if ((i + 1 < ${#tokens[@]})); then
                    i=$((i + 1))
                    filter_value=${tokens[$i]}
                    lower=${filter_value,,}
                    if [[ $lower == "arch=$ARCH_FILTER" ]]; then
                        has_b64=1
                    elif [[ $lower != key=* ]]; then
                        extras+=("-F $filter_value")
                    fi
                else
                    extras+=("-F (missing value)")
                fi
                ;;
            -F*)
                filter_value=${token:2}
                lower=${filter_value,,}
                if [[ $lower == "arch=$ARCH_FILTER" ]]; then
                    has_b64=1
                elif [[ $lower != key=* ]]; then
                    extras+=("$token")
                fi
                ;;
            -S)
                if ((i + 1 < ${#tokens[@]})); then
                    i=$((i + 1))
                    syscall_value=${tokens[$i],,}
                    IFS=',' read -r -a syscalls <<< "$syscall_value"
                    for syscall in "${syscalls[@]}"; do
                        if [[ $syscall == "kill" || $syscall == "all" ]]; then
                            has_kill=1
                        fi
                    done
                else
                    extras+=("-S (missing value)")
                fi
                ;;
            -S*)
                syscall_value=${token:2}
                syscall_value=${syscall_value,,}
                IFS=',' read -r -a syscalls <<< "$syscall_value"
                for syscall in "${syscalls[@]}"; do
                    if [[ $syscall == "kill" || $syscall == "all" ]]; then
                        has_kill=1
                    fi
                done
                ;;
            -k)
                if ((i + 1 < ${#tokens[@]})); then
                    i=$((i + 1))
                else
                    extras+=("-k (missing value)")
                fi
                ;;
            -k*)
                ;;
            -C)
                if ((i + 1 < ${#tokens[@]})); then
                    i=$((i + 1))
                    extras+=("-C ${tokens[$i]}")
                else
                    extras+=("-C (missing value)")
                fi
                ;;
            \#*)
                break
                ;;
            *)
                extras+=("$token")
                ;;
        esac
        i=$((i + 1))
    done

    if ((has_always_exit == 0 || has_b64 == 0 || has_kill == 0)); then
        return 0
    fi

    if ((${#extras[@]} == 0)); then
        RULE_CLASS="full"
        return 0
    fi

    RULE_CLASS="partial"
    for token in "${extras[@]}"; do
        if [[ -n $RULE_EXTRA_DESCRIPTION ]]; then
            RULE_EXTRA_DESCRIPTION+=", "
        fi
        RULE_EXTRA_DESCRIPTION+=$token
    done
}

# Sets NEVER_TASK_CLASS to global, filtered, or none.
classify_never_task_rule() {
    local rule=$1
    local -a tokens=()
    local action_value=""
    local token=""
    local lower=""
    local i=0
    local has_never_task=0
    local has_constraint=0

    NEVER_TASK_CLASS="none"
    read -r -a tokens <<< "$rule"

    while ((i < ${#tokens[@]})); do
        token=${tokens[$i]}
        case "$token" in
            -a|-A)
                if ((i + 1 < ${#tokens[@]})); then
                    i=$((i + 1))
                    action_value=${tokens[$i],,}
                    if [[ $action_value == "never,task" || $action_value == "task,never" ]]; then
                        has_never_task=1
                    fi
                fi
                ;;
            -a*|-A*)
                action_value=${token:2}
                action_value=${action_value,,}
                if [[ $action_value == "never,task" || $action_value == "task,never" ]]; then
                    has_never_task=1
                fi
                ;;
            -F)
                if ((i + 1 < ${#tokens[@]})); then
                    i=$((i + 1))
                    lower=${tokens[$i],,}
                    if [[ $lower != key=* ]]; then
                        has_constraint=1
                    fi
                else
                    has_constraint=1
                fi
                ;;
            -F*)
                lower=${token:2}
                lower=${lower,,}
                if [[ $lower != key=* ]]; then
                    has_constraint=1
                fi
                ;;
            -k)
                if ((i + 1 < ${#tokens[@]})); then
                    i=$((i + 1))
                fi
                ;;
            -k*)
                ;;
            \#*)
                break
                ;;
            *)
                has_constraint=1
                ;;
        esac
        i=$((i + 1))
    done

    if ((has_never_task == 1)); then
        if ((has_constraint == 1)); then
            NEVER_TASK_CLASS="filtered"
        else
            NEVER_TASK_CLASS="global"
        fi
    fi
}

# Populates ANALYSIS_* counters and rule arrays from newline-delimited rules.
analyze_rules_text() {
    local text=$1
    local line=""

    ANALYSIS_FULL=0
    ANALYSIS_PARTIAL=0
    ANALYSIS_GLOBAL_NEVER_TASK=0
    ANALYSIS_FILTERED_NEVER_TASK=0
    ANALYSIS_FULL_RULES=()
    ANALYSIS_PARTIAL_RULES=()
    ANALYSIS_PARTIAL_REASONS=()
    ANALYSIS_GLOBAL_NEVER_TASK_RULES=()
    ANALYSIS_FILTERED_NEVER_TASK_RULES=()

    while IFS= read -r line || [[ -n $line ]]; do
        trim_value "$line"
        line=$TRIMMED_VALUE
        [[ -z $line || $line == \#* || $line == "No rules" ]] && continue

        classify_kill_rule "$line"
        case "$RULE_CLASS" in
            full)
                ANALYSIS_FULL=$((ANALYSIS_FULL + 1))
                ANALYSIS_FULL_RULES+=("$line")
                ;;
            partial)
                ANALYSIS_PARTIAL=$((ANALYSIS_PARTIAL + 1))
                ANALYSIS_PARTIAL_RULES+=("$line")
                ANALYSIS_PARTIAL_REASONS+=("$RULE_EXTRA_DESCRIPTION")
                ;;
        esac

        classify_never_task_rule "$line"
        case "$NEVER_TASK_CLASS" in
            global)
                ANALYSIS_GLOBAL_NEVER_TASK=$((ANALYSIS_GLOBAL_NEVER_TASK + 1))
                ANALYSIS_GLOBAL_NEVER_TASK_RULES+=("$line")
                ;;
            filtered)
                ANALYSIS_FILTERED_NEVER_TASK=$((ANALYSIS_FILTERED_NEVER_TASK + 1))
                ANALYSIS_FILTERED_NEVER_TASK_RULES+=("$line")
                ;;
        esac
    done <<< "$text"
}

check_dependencies() {
    info "Checking audit components and command availability."

    AUDITD_BIN=$(locate_command auditd) || AUDITD_BIN=""
    AUDITCTL_BIN=$(locate_command auditctl) || AUDITCTL_BIN=""
    AUSEARCH_BIN=$(locate_command ausearch) || AUSEARCH_BIN=""
    SYSTEMCTL_BIN=$(locate_command systemctl) || SYSTEMCTL_BIN=""
    JOURNALCTL_BIN=$(locate_command journalctl) || JOURNALCTL_BIN=""

    if [[ -n $AUDITD_BIN ]]; then
        pass "auditd is installed: $AUDITD_BIN"
    else
        record_fail "auditd executable was not found."
    fi

    if [[ -n $AUDITCTL_BIN ]]; then
        pass "auditctl is available: $AUDITCTL_BIN"
    else
        record_unknown "auditctl is unavailable; loaded syscall rules cannot be verified."
    fi

    if [[ -n $AUSEARCH_BIN ]]; then
        pass "ausearch is available: $AUSEARCH_BIN"
    else
        record_unknown "ausearch is unavailable; audit records cannot be queried with the standard tool."
    fi
}

check_architecture() {
    local bits=""

    bits=$(getconf LONG_BIT 2>/dev/null) || bits=""
    if [[ $bits == "64" ]]; then
        pass "Current userspace is 64-bit; the required audit ABI is arch=$ARCH_FILTER."
    elif [[ -n $bits ]]; then
        record_fail "Current userspace is ${bits}-bit; this checker only establishes arch=$ARCH_FILTER coverage."
    else
        record_unknown "Could not determine userspace word size with getconf."
    fi
}

check_auditd_service() {
    local output=""
    local state=""
    local rc=0
    local ps_output=""
    local line=""
    local pid=""
    local uid=""
    local comm=""
    local args=""
    local process_count=0

    info "Checking whether auditd is enabled and running."
    if [[ -z $SYSTEMCTL_BIN ]]; then
        record_unknown "systemctl is unavailable; auditd enablement and service state cannot be verified."
    else
        output=$(LC_ALL=C "$SYSTEMCTL_BIN" is-enabled auditd.service 2>&1)
        rc=$?
        state=${output%%$'\n'*}
        trim_value "$state"
        state=$TRIMMED_VALUE
        case "$state" in
            enabled|enabled-runtime)
                pass "auditd.service is enabled ($state)."
                ;;
            disabled|masked|masked-runtime|not-found)
                record_fail "auditd.service is not enabled ($state)."
                ;;
            *)
                if ((rc == 0)); then
                    record_unknown "auditd.service enablement returned an unrecognized state: ${state:-empty}."
                else
                    record_unknown "Could not determine whether auditd.service is enabled: ${state:-no diagnostic}."
                fi
                ;;
        esac

        output=$(LC_ALL=C "$SYSTEMCTL_BIN" is-active auditd.service 2>&1)
        rc=$?
        state=${output%%$'\n'*}
        trim_value "$state"
        state=$TRIMMED_VALUE
        case "$state" in
            active)
                pass "auditd.service is active."
                ;;
            inactive|failed|deactivating|activating)
                record_fail "auditd.service is not running ($state)."
                ;;
            *)
                if ((rc == 0)); then
                    record_unknown "auditd.service returned an unrecognized active state: ${state:-empty}."
                else
                    record_unknown "Could not determine whether auditd.service is active: ${state:-no diagnostic}."
                fi
                ;;
        esac
    fi

    ps_output=$(LC_ALL=C ps -eo pid=,uid=,comm=,args= 2>&1)
    rc=$?
    if ((rc != 0)); then
        record_unknown "Could not inspect the process table for auditd: ${ps_output%%$'\n'*}"
        return
    fi

    while read -r pid uid comm args; do
        if [[ $comm == "auditd" ]]; then
            process_count=$((process_count + 1))
            info "auditd process: UID=$uid PID=$pid CMD=$args"
        fi
    done <<< "$ps_output"

    if ((process_count > 0)); then
        pass "Found $process_count running auditd process(es)."
    else
        record_fail "No running auditd process is visible."
    fi
}

parse_audit_status() {
    local status_text=$1
    local line=""
    local key=""
    local value=""

    AUDIT_STATUS_ENABLED=""
    AUDIT_STATUS_LOST=""
    while IFS= read -r line || [[ -n $line ]]; do
        line=${line//=/ }
        read -r key value _ <<< "$line"
        case "$key" in
            enabled)
                AUDIT_STATUS_ENABLED=$value
                ;;
            lost)
                AUDIT_STATUS_LOST=$value
                ;;
        esac
    done <<< "$status_text"
}

check_kernel_audit_status() {
    local output=""
    local rc=0

    info "Checking the kernel audit state and backlog loss counter."
    if [[ -z $AUDITCTL_BIN ]]; then
        record_unknown "Kernel audit state cannot be checked without auditctl."
        return
    fi

    output=$(LC_ALL=C "$AUDITCTL_BIN" -s 2>&1)
    rc=$?
    if ((rc != 0)); then
        record_unknown "auditctl -s failed as root: ${output%%$'\n'*}"
        return
    fi

    parse_audit_status "$output"
    if [[ $AUDIT_STATUS_ENABLED =~ ^[0-9]+$ ]]; then
        if ((AUDIT_STATUS_ENABLED > 0)); then
            pass "Kernel auditing is enabled (enabled=$AUDIT_STATUS_ENABLED)."
        else
            record_fail "Kernel auditing is disabled (enabled=0)."
        fi
    else
        record_unknown "auditctl -s did not report a usable enabled value."
    fi

    if [[ $AUDIT_STATUS_LOST =~ ^[0-9]+$ ]]; then
        if ((AUDIT_STATUS_LOST == 0)); then
            pass "Audit backlog lost counter is 0."
        else
            warn "Audit backlog lost counter is $AUDIT_STATUS_LOST; some historical records may be missing."
        fi
    else
        record_unknown "auditctl -s did not report a usable lost counter."
    fi
}

print_rule_analysis() {
    local source_label=$1
    local global_never_level=${2:-fail}
    local i=0

    for ((i = 0; i < ${#ANALYSIS_FULL_RULES[@]}; i = i + 1)); do
        info "$source_label full rule: ${ANALYSIS_FULL_RULES[$i]}"
    done
    for ((i = 0; i < ${#ANALYSIS_PARTIAL_RULES[@]}; i = i + 1)); do
        warn "$source_label filtered candidate (${ANALYSIS_PARTIAL_REASONS[$i]}): ${ANALYSIS_PARTIAL_RULES[$i]}"
    done
    for ((i = 0; i < ${#ANALYSIS_GLOBAL_NEVER_TASK_RULES[@]}; i = i + 1)); do
        if [[ $global_never_level == "warn" ]]; then
            warn "$source_label global never,task rule: ${ANALYSIS_GLOBAL_NEVER_TASK_RULES[$i]}"
        else
            fail "$source_label global never,task rule: ${ANALYSIS_GLOBAL_NEVER_TASK_RULES[$i]}"
        fi
    done
    for ((i = 0; i < ${#ANALYSIS_FILTERED_NEVER_TASK_RULES[@]}; i = i + 1)); do
        warn "$source_label filtered never,task rule may hide matching senders: ${ANALYSIS_FILTERED_NEVER_TASK_RULES[$i]}"
    done
}

check_loaded_rules() {
    local output=""
    local rc=0

    info "Checking currently loaded audit rules."
    if [[ -z $AUDITCTL_BIN ]]; then
        record_unknown "Loaded rules cannot be checked without auditctl."
        return
    fi

    output=$(LC_ALL=C "$AUDITCTL_BIN" -l 2>&1)
    rc=$?
    if ((rc != 0)); then
        record_unknown "auditctl -l failed as root: ${output%%$'\n'*}"
        return
    fi

    analyze_rules_text "$output"
    print_rule_analysis "Loaded"

    if ((ANALYSIS_FULL > 0)); then
        pass "Loaded rules include $ANALYSIS_FULL unrestricted always,exit arch=$ARCH_FILTER kill/all rule(s)."
    elif ((ANALYSIS_PARTIAL > 0)); then
        record_fail "Loaded rules only provide filtered/partial arch=$ARCH_FILTER kill coverage."
        NEED_LIVE_RULE=1
    else
        record_fail "No loaded unrestricted always,exit arch=$ARCH_FILTER kill/all rule was found."
        NEED_LIVE_RULE=1
    fi

    if ((ANALYSIS_GLOBAL_NEVER_TASK > 0)); then
        record_fail "A global never,task rule prevents syscall audit matching for affected processes."
    else
        pass "No global never,task rule was found in the loaded rules."
    fi
}

parse_auditd_config() {
    local config=$1
    local line=""
    local key=""
    local value=""

    AUDIT_WRITE_LOGS="yes"
    AUDIT_LOG_FILE="/var/log/audit/audit.log"
    AUDIT_LOCAL_EVENTS="yes"
    AUDIT_LOG_FORMAT="enriched"
    AUDIT_WRITE_LOGS_DEFAULT=1
    AUDIT_LOG_FILE_DEFAULT=1
    AUDIT_LOCAL_EVENTS_DEFAULT=1
    AUDIT_LOG_FORMAT_DEFAULT=1

    while IFS= read -r line || [[ -n $line ]]; do
        trim_value "$line"
        line=$TRIMMED_VALUE
        [[ -z $line || $line == \#* || $line != *=* ]] && continue
        key=${line%%=*}
        value=${line#*=}
        trim_value "$key"
        key=${TRIMMED_VALUE,,}
        value=${value%%#*}
        trim_value "$value"
        value=$TRIMMED_VALUE
        case "$key" in
            write_logs)
                AUDIT_WRITE_LOGS=${value,,}
                AUDIT_WRITE_LOGS_DEFAULT=0
                ;;
            log_file)
                AUDIT_LOG_FILE=$value
                AUDIT_LOG_FILE_DEFAULT=0
                ;;
            local_events)
                AUDIT_LOCAL_EVENTS=${value,,}
                AUDIT_LOCAL_EVENTS_DEFAULT=0
                ;;
            log_format)
                AUDIT_LOG_FORMAT=${value,,}
                AUDIT_LOG_FORMAT_DEFAULT=0
                ;;
        esac
    done < "$config"
}

check_persistent_logging() {
    local config="/etc/audit/auditd.conf"
    local qualifier="configured"

    info "Checking auditd persistent log configuration."
    if [[ ! -e $config ]]; then
        record_fail "$config does not exist; persistent audit logging cannot be established."
        return
    fi
    if [[ ! -r $config ]]; then
        record_unknown "$config is not readable, even as root."
        return
    fi

    parse_auditd_config "$config"
    if ((AUDIT_WRITE_LOGS_DEFAULT == 1)); then
        qualifier="defaulted"
    fi
    if [[ $AUDIT_WRITE_LOGS == "yes" ]]; then
        pass "auditd persistent log writing is enabled ($qualifier write_logs=yes)."
    else
        record_fail "auditd persistent log writing is not enabled (write_logs=${AUDIT_WRITE_LOGS:-empty})."
    fi

    qualifier="configured"
    if ((AUDIT_LOCAL_EVENTS_DEFAULT == 1)); then
        qualifier="defaulted"
    fi
    if [[ $AUDIT_LOCAL_EVENTS == "yes" ]]; then
        pass "auditd accepts local events ($qualifier local_events=yes)."
    else
        record_fail "auditd does not accept local events (local_events=${AUDIT_LOCAL_EVENTS:-empty})."
    fi

    qualifier="configured"
    if ((AUDIT_LOG_FORMAT_DEFAULT == 1)); then
        qualifier="defaulted"
    fi
    case "$AUDIT_LOG_FORMAT" in
        raw|enriched)
            pass "auditd log format writes records ($qualifier log_format=$AUDIT_LOG_FORMAT)."
            ;;
        nolog)
            record_fail "auditd log_format=NOLOG overrides write_logs and disables disk records."
            ;;
        *)
            record_fail "auditd log format is not recognized as writable (log_format=${AUDIT_LOG_FORMAT:-empty})."
            ;;
    esac

    qualifier="configured"
    if ((AUDIT_LOG_FILE_DEFAULT == 1)); then
        qualifier="defaulted"
    fi
    info "auditd log path is $AUDIT_LOG_FILE ($qualifier)."
    if [[ $AUDIT_LOG_FILE == /* && -f $AUDIT_LOG_FILE ]]; then
        pass "Audit log path is a regular file: $AUDIT_LOG_FILE"
    else
        record_fail "Audit log path is not an existing regular file with an absolute path: $AUDIT_LOG_FILE"
    fi
}

check_persistent_rules() {
    local rules_dir="/etc/audit/rules.d"
    local -a rule_files=()
    local file=""
    local line=""
    local text=""
    local unreadable=0

    info "Checking for a persistent arch=$ARCH_FILTER kill rule under $rules_dir."
    if [[ ! -d $rules_dir ]]; then
        warn "$rules_dir does not exist; the loaded kill rule cannot be restored from rules.d."
        NEED_PERSISTENT_RULE=1
        return
    fi

    shopt -s nullglob
    rule_files=("$rules_dir"/*.rules)
    shopt -u nullglob
    if ((${#rule_files[@]} == 0)); then
        warn "No *.rules files exist under $rules_dir."
        NEED_PERSISTENT_RULE=1
        return
    fi

    for file in "${rule_files[@]}"; do
        if [[ ! -r $file ]]; then
            unreadable=$((unreadable + 1))
            warn "Persistent rule file is unreadable: $file"
            continue
        fi
        while IFS= read -r line || [[ -n $line ]]; do
            text+="$line"$'\n'
        done < "$file"
    done

    analyze_rules_text "$text"
    print_rule_analysis "Persistent" "warn"
    if ((ANALYSIS_FULL > 0)); then
        pass "Found $ANALYSIS_FULL persistent unrestricted arch=$ARCH_FILTER kill/all rule(s) in rules.d."
    elif ((unreadable > 0)); then
        unknown "No readable full rule was found, but $unreadable rule file(s) could not be inspected."
        NEED_PERSISTENT_RULE=1
    elif ((ANALYSIS_PARTIAL > 0)); then
        warn "rules.d only contains filtered/partial arch=$ARCH_FILTER kill coverage."
        NEED_PERSISTENT_RULE=1
    else
        warn "No persistent unrestricted arch=$ARCH_FILTER kill/all rule was found in rules.d."
        NEED_PERSISTENT_RULE=1
    fi

    if ((ANALYSIS_GLOBAL_NEVER_TASK > 0)); then
        warn "A persistent global never,task rule will restore the syscall-audit blocker when rules reload."
    fi
}

check_kgp_processes() {
    local output=""
    local rc=0
    local line=""
    local uid=""
    local pid=""
    local ppid=""
    local comm=""
    local args=""
    local argv0=""
    local argv0_base=""
    local kgp_count=0
    local wrapper_count=0

    info "Checking visible kgp and run-kgp.sh processes."
    output=$(LC_ALL=C ps -eo uid=,pid=,ppid=,comm=,args= 2>&1)
    rc=$?
    if ((rc != 0)); then
        unknown "Could not inspect processes for kgp: ${output%%$'\n'*}"
        return
    fi

    while read -r uid pid ppid comm args; do
        argv0=${args%%[[:space:]]*}
        argv0_base=${argv0##*/}
        if [[ $comm == "kgp" || $argv0_base == "kgp" ]]; then
            kgp_count=$((kgp_count + 1))
            info "kgp process: UID=$uid PID=$pid PPID=$ppid CMD=$args"
        elif [[ $args =~ (^|[[:space:]])([^[:space:]]*/)?run-kgp\.sh([[:space:]]|$) ]]; then
            wrapper_count=$((wrapper_count + 1))
            info "run-kgp.sh process: UID=$uid PID=$pid PPID=$ppid CMD=$args"
        fi
    done <<< "$output"

    if ((kgp_count == 0)); then
        info "No current kgp process is visible."
    else
        info "Found $kgp_count current kgp process(es); process metadata is indirect evidence only."
    fi
    if ((wrapper_count > 0)); then
        info "Found $wrapper_count current run-kgp.sh wrapper process(es)."
    fi
}

check_journal_evidence() {
    local output=""
    local matches=""
    local rc=0
    local count=0

    info "Checking visible journal entries for kgp, run-kgp.sh, or [KILL]."
    if [[ -z $JOURNALCTL_BIN ]]; then
        unknown "journalctl is unavailable; journal evidence was not checked."
        return
    fi

    output=$(LC_ALL=C "$JOURNALCTL_BIN" --quiet --no-pager --lines=20 \
        --grep='kgp|run-kgp\.sh|\[KILL\]' 2>&1)
    rc=$?
    if ((rc == 1)) && [[ -z $output || $output == *"-- No entries --"* ]]; then
        output=""
    elif ((rc != 0)) && [[ $output == *"unrecognized option"* || $output == *"unknown option"* ]]; then
        output=$(LC_ALL=C "$JOURNALCTL_BIN" --quiet --no-pager --lines=10000 2>&1)
        rc=$?
        if ((rc != 0)); then
            unknown "journalctl query failed: ${output%%$'\n'*}"
            return
        fi
        matches=$(printf '%s\n' "$output" | grep -E 'kgp|run-kgp\.sh|\[KILL\]' || true)
        output=$matches
        info "journalctl lacks a usable --grep query; searched the latest 10000 visible entries instead."
    elif ((rc != 0)); then
        unknown "journalctl query failed: ${output%%$'\n'*}"
        return
    fi

    count=$(printf '%s\n' "$output" | awk '
        NF && $0 !~ /^-- (No entries|Journal begins)/ { count++ }
        END { print count + 0 }
    ')
    if ((count > 0)); then
        info "Found $count matching visible journal line(s); journal text is indirect evidence only."
    else
        info "No matching entries were found in the visible journal."
    fi
}

print_recommendations() {
    if ((NEED_LIVE_RULE == 1)); then
        info "Suggested live rule (not executed): sudo auditctl -a always,exit -F arch=$ARCH_FILTER -S kill -k trace_kill"
    fi
    if ((NEED_PERSISTENT_RULE == 1)); then
        info "Suggested persistent file: /etc/audit/rules.d/50-kill.rules"
        info "Suggested file content: -a always,exit -F arch=$ARCH_FILTER -S kill -k trace_kill"
        info "Suggested reload after review (not executed): sudo augenrules --load"
    fi
}

main() {
    local arg=""

    if (($# > 1)); then
        fail "Unexpected arguments: $*"
        usage
        return 64
    fi
    if (($# == 1)); then
        arg=$1
        case "$arg" in
            -h|--help)
                usage
                return 0
                ;;
            *)
                fail "Unknown argument: $arg"
                usage
                return 64
                ;;
        esac
    fi

    if ((EUID == 0)); then
        ROOT_MODE=1
        info "Running as root; authoritative kernel and loaded-rule checks are enabled."
    else
        ROOT_MODE=0
        info "Running as UID=$EUID without privilege escalation; all checks remain read-only."
    fi

    check_dependencies
    check_architecture
    check_auditd_service

    if ((ROOT_MODE == 1)); then
        check_kernel_audit_status
        check_loaded_rules
        check_persistent_logging
        check_persistent_rules
    else
        unknown "Loaded audit rules, kernel audit state, and audit log configuration require root access."
        info "Rerun for an authoritative result: sudo ${BASH_SOURCE[0]}"
        OVERALL_UNKNOWN=1
    fi

    check_kgp_processes
    check_journal_evidence
    print_recommendations

    if ((ROOT_MODE == 0)); then
        unknown "Overall: UNKNOWN - root access is required to establish syscall audit coverage."
        return 2
    fi
    if ((OVERALL_FAIL == 1)); then
        fail "Overall: NOT_READY/PARTIAL - one or more required conditions are missing."
        return 1
    fi
    if ((OVERALL_UNKNOWN == 1)); then
        unknown "Overall: UNKNOWN - one or more required conditions could not be verified."
        return 2
    fi

    pass "Overall: READY - auditd can persist records and an unrestricted b64 kill rule is loaded."
    return 0
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    set -u
    set -o pipefail
    main "$@"
    exit $?
fi
