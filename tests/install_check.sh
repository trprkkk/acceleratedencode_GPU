#!/bin/sh
# install_check.sh [BUILD] - the library as a user gets it.
#
#  1. `make install` into a scratch DESTDIR (shared + static library, header,
#     pkg-config file, aegenc);
#  2. build examples/encode.c against that copy through pkg-config, once
#     linked to the shared library and once static (--static), run both, and
#     check that they write the same stream as each other and as aegenc;
#  3. the shared library exports exactly the AEG_API symbols of the header;
#  4. the header compiles as C99 (-pedantic) and as C++;
#  5. `make uninstall` leaves nothing behind.
#
# PKG_CONFIG names the pkg-config binary (default: pkg-config, then pkgconf).
# Without one, step 2 is skipped unless REQUIRE_PKG_CONFIG=1.
set -eu

BUILD=${1:-build}
CC=${CC:-cc}
CXX=${CXX:-c++}
TMP=$(mktemp -d "${TMPDIR:-/tmp}/aeg-install.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
STAGE=$TMP/stage
PREFIX=/opt/aeg
fail=0

say() { echo "install_check: $*"; }
bad() { echo "install_check: FAIL: $*"; fail=1; }

make -s BUILD="$BUILD" DESTDIR="$STAGE" PREFIX="$PREFIX" install
for f in bin/aegenc lib/libaeg.a lib/libaeg.so lib/libaeg.so.0 \
         include/aeg/aeg.h lib/pkgconfig/aeg.pc; do
    [ -e "$STAGE$PREFIX/$f" ] || bad "$f was not installed"
done
SONAME=$(readelf -d "$STAGE$PREFIX/lib/libaeg.so" | sed -n 's/.*SONAME.*\[\(.*\)\]/\1/p')
[ "$SONAME" = libaeg.so.0 ] || bad "soname '$SONAME'"
say "installed into $STAGE$PREFIX (soname $SONAME)"

# --- 3. exported symbols = the header's AEG_API declarations ---------------
# Every name declared after AEG_API (functions, and the aeg_log_level object).
sed -n 's/^AEG_API[^(;]*[ *]\(aeg_[a-z0-9_]*\) *[(;].*/\1/p' \
    "$STAGE$PREFIX/include/aeg/aeg.h" | sort -u > "$TMP/declared"
nm -D --defined-only "$STAGE$PREFIX/lib/libaeg.so" |
    awk '$2 ~ /^[TDBRVW]$/ { print $3 }' | sort -u > "$TMP/exported"
if ! cmp -s "$TMP/declared" "$TMP/exported"; then
    bad "exported symbols differ from the header (< header, > library):"
    diff "$TMP/declared" "$TMP/exported" | sed -n 's/^[<>]/  &/p'
else
    say "exports: $(wc -l < "$TMP/exported") symbols, all declared AEG_API"
fi

# --- 4. the header alone, strict C and C++ ---------------------------------
printf '#include <aeg/aeg.h>\nint main(void) { aeg_params p; aeg_params_default(&p); return p.width; }\n' \
    > "$TMP/hdr.c"
$CC -std=c99 -pedantic -Wall -Wextra -Werror -I"$STAGE$PREFIX/include" \
    -fsyntax-only "$TMP/hdr.c" || bad "the header is not clean C99"
if command -v "$CXX" > /dev/null 2>&1; then
    cp "$TMP/hdr.c" "$TMP/hdr.cpp"
    $CXX -std=c++11 -pedantic -Wall -Wextra -Werror \
        -I"$STAGE$PREFIX/include" -fsyntax-only "$TMP/hdr.cpp" ||
        bad "the header does not compile as C++"
    say "header: C99 -pedantic and C++11 clean"
else
    say "header: C99 -pedantic clean (no C++ compiler)"
fi

# --- 2. the example through pkg-config ------------------------------------
PKG=${PKG_CONFIG:-}
if [ -z "$PKG" ]; then
    for c in pkg-config pkgconf; do
        if command -v $c > /dev/null 2>&1; then PKG=$c; break; fi
    done
fi
if [ -n "$PKG" ]; then
    # the .pc file names the final PREFIX; sysroot maps it into the stage
    export PKG_CONFIG_PATH="$STAGE$PREFIX/lib/pkgconfig"
    export PKG_CONFIG_SYSROOT_DIR="$STAGE"
    CF=$($PKG --cflags aeg)
    LS=$($PKG --libs aeg)
    LST=$($PKG --static --libs aeg)
    VER=$($PKG --modversion aeg)
    say "pkg-config: aeg $VER, cflags '$CF', libs '$LS', static '$LST'"
    $CC -Wall -Wextra -Werror examples/encode.c $CF $LS -o "$TMP/ex_shared"
    # static: the archive alone (the shared library would win otherwise)
    LIBA=$(echo "$LST" | sed 's/-laeg//')
    $CC -Wall -Wextra -Werror examples/encode.c $CF \
        "$STAGE$PREFIX/lib/libaeg.a" $LIBA -o "$TMP/ex_static"
    if readelf -d "$TMP/ex_static" | grep -q 'libaeg'; then
        bad "the static example links libaeg dynamically"
    fi
    readelf -d "$TMP/ex_shared" | grep -q 'NEEDED.*libaeg.so.0' ||
        bad "the shared example does not need libaeg.so.0"

    "$BUILD/genyuv" 176 144 12 "$TMP/in.yuv" 4 > /dev/null
    LD_LIBRARY_PATH="$STAGE$PREFIX/lib" "$TMP/ex_shared" "$TMP/in.yuv" 176 144 \
        "$TMP/shared.264" 24 > "$TMP/shared.log" 2> "$TMP/shared.err" ||
        { bad "shared example failed"; cat "$TMP/shared.log" "$TMP/shared.err"; }
    "$TMP/ex_static" "$TMP/in.yuv" 176 144 "$TMP/static.264" 24 \
        > "$TMP/static.log" 2> /dev/null || bad "static example failed"
    cmp -s "$TMP/shared.264" "$TMP/static.264" ||
        bad "shared and static examples wrote different streams"
    grep -q '^\[libaeg ' "$TMP/shared.err" ||
        bad "no library message went through the example's log callback"
    grep -q '12 frames in, 12 packets out' "$TMP/shared.log" ||
        bad "the example lost pictures: $(tail -1 "$TMP/shared.log")"
    # the same encode from the CLI (packed input, CRF 24, preset 2)
    "$STAGE$PREFIX/bin/aegenc" -i "$TMP/in.yuv" -w 176 -h 144 --crf 24 -p 2 \
        -o "$TMP/cli.264" > /dev/null 2>&1 || bad "aegenc failed"
    cmp -s "$TMP/shared.264" "$TMP/cli.264" ||
        bad "the example (strided planes) and aegenc (packed) differ"
    say "example: shared and static builds run, $(wc -c < "$TMP/shared.264") bytes, identical to aegenc"
elif [ "${REQUIRE_PKG_CONFIG:-0}" = 1 ]; then
    bad "no pkg-config"
else
    say "no pkg-config: example build skipped"
fi

# --- 5. uninstall -----------------------------------------------------------
make -s BUILD="$BUILD" DESTDIR="$STAGE" PREFIX="$PREFIX" uninstall
left=$(find "$STAGE$PREFIX" -type f -o -type l | head -5)
[ -z "$left" ] || bad "uninstall left: $left"

if [ $fail != 0 ]; then
    say "FAILED"
    exit 1
fi
say "all checks passed"
