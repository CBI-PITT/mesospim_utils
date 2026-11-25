# Linux x86-64 ImarisWriter runtime

This directory contains the Python wrapper and two builds of
[ImarisWriter](https://github.com/imaris/ImarisWriter) commit
`b128e6e7d1a147261e9d5caf24ebc6b5c9c63779` (Apache-2.0) with
[HDF5](https://github.com/HDFGroup/hdf5) tag `hdf5_1.14.6`
(`7bf340440909d468dbb3cf41f0ea0d87f5050cea`). LZ4 v1.10.0
(`ebb370ca83af193212df4dcbadcc5d87bc0de2f0`) is linked statically.
Their license texts are included here.

- `linux_x86_64_glibc217/` is the **default** used by the converter, built
  against a glibc 2.17 sysroot with GCC 11. It requires at most
  `GLIBC_2.17` and `GLIBCXX_3.4.30`, below the cluster compute nodes' glibc
  2.35 / GCC 11 runtime. System glibc, `libstdc++.so.6`, `libgcc_s.so.1`,
  and `libz.so.1` remain external.
- The `.so` files in this directory are the original **newer login-node build**
  (glibc 2.38 / GCC 13), retained for comparison and not selected by code.

Both writers have `$ORIGIN` RPATH/RUNPATH to their respective HDF5 libraries.
They are Linux x86-64 binaries; rebuild for other architectures.

To reproduce either build in a temporary working directory (do not install
into an existing Python or conda environment), fetch the pinned sources:

```bash
WORK=/tmp/mesospim-imaris-build
mkdir -p "$WORK"
git clone --depth 1 --branch hdf5_1.14.6 https://github.com/HDFGroup/hdf5.git "$WORK/hdf5-source"
git clone --depth 1 --branch v1.10.0 https://github.com/lz4/lz4.git "$WORK/lz4-source"
git clone --depth 1 https://github.com/imaris/ImarisWriter.git "$WORK/ImarisWriter"
git -C "$WORK/ImarisWriter" fetch --depth 1 origin b128e6e7d1a147261e9d5caf24ebc6b5c9c63779
git -C "$WORK/ImarisWriter" checkout b128e6e7d1a147261e9d5caf24ebc6b5c9c63779

cmake -S "$WORK/hdf5-source" -B "$WORK/hdf5-build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$WORK/hdf5-install" \
  -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF -DHDF5_BUILD_TOOLS=OFF \
  -DHDF5_BUILD_EXAMPLES=OFF -DHDF5_BUILD_HL_LIB=OFF \
  -DHDF5_BUILD_CPP_LIB=OFF -DHDF5_BUILD_FORTRAN=OFF \
  -DHDF5_ENABLE_SZIP_SUPPORT=OFF -DHDF5_ENABLE_Z_LIB_SUPPORT=ON \
  -DHDF5_ENABLE_PLUGIN_SUPPORT=OFF -DCMAKE_INSTALL_RPATH='$ORIGIN'
cmake --build "$WORK/hdf5-build" --parallel 12
cmake --install "$WORK/hdf5-build"

cmake -S "$WORK/lz4-source/build/cmake" -B "$WORK/lz4-build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$WORK/lz4-install" \
  -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DLZ4_BUILD_CLI=OFF
cmake --build "$WORK/lz4-build" --parallel 4
cmake --install "$WORK/lz4-build"

cmake -S "$WORK/ImarisWriter" -B "$WORK/imaris-build" \
  -DCMAKE_BUILD_TYPE=Release -DHDF5_ROOT="$WORK/hdf5-install" \
  -DZLIB_ROOT=/usr -DLZ4_ROOT="$WORK/lz4-install" -DLZ4_USE_STATIC=ON \
  -DCMAKE_BUILD_RPATH="$WORK/hdf5-install/lib"
cmake --build "$WORK/imaris-build" --target bpImarisWriter96 --parallel 8

RUNTIME=mesospim_utils/_vendor/PyImarisWriter
cp "$WORK/imaris-build/libbpImarisWriter96.so" "$RUNTIME/"
cp "$WORK/hdf5-install/lib/libhdf5.so.310.5.1" "$RUNTIME/libhdf5.so.310"
patchelf --set-rpath '$ORIGIN' "$RUNTIME/libbpImarisWriter96.so"
```

For the compute-compatible build, create an **isolated, throwaway build
toolchain**. Reuse the pinned sources cloned above:

```bash
TOOLCHAIN="$WORK/toolchain-glibc217"
conda create --prefix "$TOOLCHAIN" --override-channels -c conda-forge \
  --no-default-packages -y 'gcc_linux-64=11.2' 'gxx_linux-64=11.2' \
  'sysroot_linux-64=2.17' 'zlib=1.2.11'
SYSROOT="$TOOLCHAIN/x86_64-conda-linux-gnu/sysroot"

cmake -S "$WORK/hdf5-source" -B "$WORK/hdf5-build-old" \
  -DCMAKE_C_COMPILER="$TOOLCHAIN/bin/x86_64-conda-linux-gnu-gcc" \
  -DCMAKE_SYSROOT="$SYSROOT" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$WORK/hdf5-install-old" \
  -DBUILD_SHARED_LIBS=ON -DBUILD_TESTING=OFF -DHDF5_BUILD_TOOLS=OFF \
  -DHDF5_BUILD_EXAMPLES=OFF -DHDF5_BUILD_HL_LIB=OFF \
  -DHDF5_BUILD_CPP_LIB=OFF -DHDF5_BUILD_FORTRAN=OFF \
  -DHDF5_ENABLE_SZIP_SUPPORT=OFF -DHDF5_ENABLE_Z_LIB_SUPPORT=ON \
  -DHDF5_ENABLE_PLUGIN_SUPPORT=OFF -DCMAKE_INSTALL_RPATH='$ORIGIN' \
  -DZLIB_ROOT="$TOOLCHAIN"
LD_LIBRARY_PATH="$TOOLCHAIN/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  cmake --build "$WORK/hdf5-build-old" --parallel 12
cmake --install "$WORK/hdf5-build-old" --component libraries
cmake --install "$WORK/hdf5-build-old" --component headers
cmake --install "$WORK/hdf5-build-old" --component configinstall
cmake --install "$WORK/hdf5-build-old" --component utilsapplications

cmake -S "$WORK/lz4-source/build/cmake" -B "$WORK/lz4-build-old" \
  -DCMAKE_C_COMPILER="$TOOLCHAIN/bin/x86_64-conda-linux-gnu-gcc" \
  -DCMAKE_SYSROOT="$SYSROOT" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$WORK/lz4-install-old" \
  -DBUILD_SHARED_LIBS=OFF -DBUILD_STATIC_LIBS=ON -DLZ4_BUILD_CLI=OFF
cmake --build "$WORK/lz4-build-old" --parallel 4
cmake --install "$WORK/lz4-build-old"

cmake -S "$WORK/ImarisWriter" -B "$WORK/imaris-build-old" \
  -DCMAKE_C_COMPILER="$TOOLCHAIN/bin/x86_64-conda-linux-gnu-gcc" \
  -DCMAKE_CXX_COMPILER="$TOOLCHAIN/bin/x86_64-conda-linux-gnu-g++" \
  -DCMAKE_SYSROOT="$SYSROOT" -DCMAKE_BUILD_TYPE=Release \
  -DHDF5_ROOT="$WORK/hdf5-install-old" -DZLIB_ROOT="$TOOLCHAIN" \
  -DLZ4_ROOT="$WORK/lz4-install-old" -DLZ4_USE_STATIC=ON \
  -DCMAKE_BUILD_RPATH="$WORK/hdf5-install-old/lib"
cmake --build "$WORK/imaris-build-old" --target bpImarisWriter96 --parallel 8

RUNTIME=mesospim_utils/_vendor/PyImarisWriter/linux_x86_64_glibc217
mkdir -p "$RUNTIME"
cp "$WORK/imaris-build-old/libbpImarisWriter96.so" "$RUNTIME/"
cp "$WORK/hdf5-install-old/lib/libhdf5.so.310.5.1" "$RUNTIME/libhdf5.so.310"
patchelf --set-rpath '$ORIGIN' "$RUNTIME/libbpImarisWriter96.so"
patchelf --set-rpath '$ORIGIN' "$RUNTIME/libhdf5.so.310"
```

The three Python wrapper files come from the same ImarisWriter commit.
`PyImarisWriter.py` selects `linux_x86_64_glibc217/` in `_load_dll()` and
includes the native library path in load errors. Verify both builds with
`ldd`, check that the **default** binaries require no newer than
`GLIBC_2.17` / `GLIBCXX_3.4.30` via `readelf --version-info`, and run
`python -m pytest tests/test_omezarr_to_ims.py -q`.
