#!/bin/sh
# The part of a fuzz log worth reading when a run fails, as one GitHub
# annotation (the raw log is not reachable everywhere the checks API is):
# the target that failed, the sanitizer report up to its SUMMARY with the
# first stack frames, and libFuzzer's Base64 of the input, which is enough
# to reproduce:  echo B64 | base64 -d > in; ./build-fuzz/fuzz_<t> in
log=$1 title=$2
[ -f "$log" ] || exit 0
awk '
  /^== fuzz_/                     { tgt = $0 }
  /ERROR: |runtime error:|deadly signal|out-of-memory|timeout after/ {
                                    on = 1; frames = 0 }
  on && /^ +#[0-9]+ /             { if (++frames > 14) next }
  on                              { print; if (/^SUMMARY/) on = 0 }
  /^Base64: /                     { b64 = $0 }
  /^(make|fuzz: bad|driver:)/     { last = $0 }
  END { if (tgt) print tgt; if (b64) print b64; if (last) print last }
' "$log" | grep -av '^warning' | head -n 60 > "$log.report"
[ -s "$log.report" ] || tail -n 40 "$log" | grep -av '^"' > "$log.report"
# the inputs libFuzzer kept (build-fuzz/<target>-crash-<sha1> etc.), whole
for f in build-fuzz/fuzz_*-crash-* build-fuzz/fuzz_*-leak-* \
         build-fuzz/fuzz_*-timeout-* build-fuzz/fuzz_*-oom-*; do
    [ -f "$f" ] || continue
    echo "$(basename "$f") ($(wc -c < "$f") bytes):" >> "$log.report"
    base64 -w0 "$f" | head -c 12000 >> "$log.report"
    echo >> "$log.report"
done
cat "$log.report"
printf '::error title=%s::%s\n' "$title" \
  "$(sed 's/%/%25/g' "$log.report" | awk '{printf "%s%%0A", $0}')"
