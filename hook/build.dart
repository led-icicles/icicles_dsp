import 'package:hooks/hooks.dart';
import 'package:native_toolchain_rust/native_toolchain_rust.dart';

/// Builds the Rust crate in `rust/` with the toolchain pinned in
/// `rust/rust-toolchain.toml` and bundles the resulting dynamic library as the
/// code asset behind `lib/src/icicles_dsp_bindings_generated.dart`.
void main(List<String> args) async {
  await build(args, (input, output) async {
    await RustBuilder(
      assetName: 'src/icicles_dsp_bindings_generated.dart',
      cratePath: 'rust',
    ).run(input: input, output: output);
  });
}
