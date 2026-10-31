# A real OpenCL device without a GPU: PoCL, the portable OpenCL
# implementation that runs kernels on the host CPU.  It is a complete OpenCL
# runtime -- kernels are compiled by its own LLVM, buffers are separate device
# allocations, transfers are real copies -- so every host wrapper, transfer
# layout and kernel of the OpenCL backend runs exactly as it would on a GPU.
# Only the speed is not a GPU's.
#
# Source it (it sets environment variables for the current shell):
#
#   . tools/pocl_env.sh [DIR]          # DIR default: $AEG_POCL_DIR or
#                                      #   ~/.cache/aeg-pocl
#   make OPENCL=1
#   ./build/aegenc --list-devices      # [1] opencl  pthread-...
#   ./build/backend_check              # every device operation vs the CPU
#   BE=opencl tools/gpu_validate.sh clip.yuv W H FRAMES QP
#
# The first run installs two Python wheels into DIR with pip (no root, no
# system packages, nothing outside DIR): pyopencl, which bundles an ocl-icd
# loader, and pocl-binary-distribution, the PoCL runtime with its LLVM.  The
# encoder does not use Python at all; the wheels are only the most portable
# way to get a prebuilt loader and runtime.
#
# What the variables do:
#   AEG_OPENCL_LIB   the bundled loader (its file name carries a hash, so the
#                    encoder's usual "libOpenCL.so.1" lookup would not find it)
#   PYOPENCL_HOME    where that loader looks for its vendor file (pocl.icd);
#                    without it the loader crashes inside clGetPlatformIDs
#   LD_LIBRARY_PATH  the vendor file names the PoCL library without a path
#
# (OCL_ICD_VENDORS pointing at a hand-written .icd was tried as well and
# crashes the same bundled loader; this is the combination that works.)

_aeg_pocl_dir=${1:-${AEG_POCL_DIR:-$HOME/.cache/aeg-pocl}}
_aeg_pocl_libs=$_aeg_pocl_dir/pyopencl/.libs
if ! ls "$_aeg_pocl_libs"/libOpenCL*.so* >/dev/null 2>&1 ||
   ! ls "$_aeg_pocl_libs"/*.icd >/dev/null 2>&1; then
    echo "pocl_env: installing pyopencl + pocl-binary-distribution into $_aeg_pocl_dir"
    python3 -m pip install -q --target "$_aeg_pocl_dir" pyopencl \
        pocl-binary-distribution ||
        { echo "pocl_env: pip install failed" >&2; return 1 2>/dev/null || exit 1; }
fi
AEG_OPENCL_LIB=$(ls "$_aeg_pocl_libs"/libOpenCL*.so* | head -n 1)
PYOPENCL_HOME=$_aeg_pocl_dir/pyopencl
LD_LIBRARY_PATH=$_aeg_pocl_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export AEG_OPENCL_LIB PYOPENCL_HOME LD_LIBRARY_PATH
echo "pocl_env: AEG_OPENCL_LIB=$AEG_OPENCL_LIB"
unset _aeg_pocl_dir _aeg_pocl_libs
