# frozen_string_literal: true

# Homebrew formula for the cuda-legacy CLI (Linux x86_64 only; install with --HEAD until a release is tagged).
class CudaLegacy < Formula
  include Language::Python::Virtualenv

  desc "Inspect and adapt newer CUDA workloads for older NVIDIA GPUs"
  homepage "https://cudacompat.com/"
  license any_of: ["MIT", "Apache-2.0"]
  head "https://github.com/zachshallbetter/cuda-compatibility.git", branch: "main"

  depends_on :linux
  depends_on "python@3.12"

  def install
    virtualenv_install_with_resources
  end

  def caveats
    <<~EOS
      cuda-legacy runs your PyTorch scripts with the interpreter of the environment
      that holds the CUDA-enabled PyTorch wheel, not Homebrew's Python. Point it there:
        export CUDA_COMPAT_PYTHON=/path/to/your-env/bin/python
        cuda-legacy doctor --llvm-root /path/to/LLVM-23.1.4
      or pass --python to doctor and run. Linux x86_64 with visible sm_52 GPUs only.
    EOS
  end

  test do
    assert_match "cuda-legacy", shell_output("#{bin}/cuda-legacy --help")
    assert_match "pytorch-2.14-cu130", shell_output("#{bin}/cuda-legacy profiles list")
  end
end
