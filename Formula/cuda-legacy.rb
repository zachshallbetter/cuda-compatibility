# frozen_string_literal: true

# Homebrew formula for the cuda-legacy CLI (Linux x86_64 only).
class CudaLegacy < Formula
  include Language::Python::Virtualenv

  desc "Inspect and adapt newer CUDA workloads for older NVIDIA GPUs"
  homepage "https://cudacompat.com/"
  url "https://github.com/zachshallbetter/cuda-compatibility/archive/refs/tags/v0.2.0.tar.gz"
  sha256 "d4cc7295914349ba2dbe643db55061d3903cf8490339fc77fe20837d989fe66e"
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
      that holds the CUDA-enabled PyTorch wheel, not Homebrew's Python. Run setup once
      and point it there; it finds or downloads LLVM 23.1.4 and saves everything:
        cuda-legacy setup --python /path/to/your-env/bin/python
      Then: cuda-legacy run app.py. Linux x86_64 with visible sm_52 GPUs only.
    EOS
  end

  test do
    assert_match "cuda-legacy", shell_output("#{bin}/cuda-legacy --help")
    assert_match "pytorch-2.14-cu130", shell_output("#{bin}/cuda-legacy profiles list")
    assert_match "--remove", shell_output("#{bin}/cuda-legacy setup --help")
  end
end
