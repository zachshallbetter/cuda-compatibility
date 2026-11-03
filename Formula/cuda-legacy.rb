# frozen_string_literal: true

# Homebrew formula for the cuda-legacy CLI (Linux x86_64 only).
class CudaLegacy < Formula
  include Language::Python::Virtualenv

  desc "Inspect and adapt newer CUDA workloads for older NVIDIA GPUs"
  homepage "https://cudacompat.com/"
  url "https://github.com/zachshallbetter/cuda-compatibility/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "76928fb07b0244df14d2aa0cd84068d0e73c66930f008ba1da0e8f2d8b82d2aa"
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
