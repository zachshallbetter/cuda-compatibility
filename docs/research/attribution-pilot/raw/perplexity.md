# perplexity — attribution pilot v0.1.1

- Collector: Claude (built-in browser, tab-2), with the owner's approval
- Mode: signed out (no account; optional cookies declined at 13:13 PST)
- Start: 2026-11-03 13:08 PST  End: 2026-11-03 13:35 PST
- Location (city/region, as the product sees it): not shown

Collection notes:
- Signed-out Perplexity often refused requests with "Sign up and repeat your request." These are recorded as answers not shown. Most repeat runs and both A-004 attempts were refused.
- After about 13:13 PST a sign-up dialog covered the source panel. From then on the sources listed are the answer's inline citations (from `data-pplx-citation-url`), not the full source list; the notes give the source count the page showed.
- Answers were read from the saved answer page after reloading, because the live view was sometimes cut off mid-answer.
- Model or mode: not shown on any answer.

---

## Q-CUDA13-001 · run 1

- Time: 13:11 PST
- Model or mode shown: not shown
- Answer shown: yes
- Notes: Full source panel read before the sign-up wall appeared: 2 sources.

### Answer

It depends on how old. CUDA 13 drops support for building new applications for GPUs below compute capability 7.5 (Turing); prebuilt apps made with older CUDA toolkits can still run on hardware they supported.
developer.nvidia

So older GPUs may work with a CUDA 13 PyTorch build only if that build includes kernels for the GPU; Maxwell and Pascal support is being removed from newer PyTorch releases.
github

### Sources

1. https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/
2. https://github.com/pytorch/pytorch/issues/157517

---

## Q-CUDA13-001 · run 2

- Time: 13:11 PST
- Model or mode shown: not shown
- Answer shown: no
- Notes: Signed-out repeat of an identical prompt was refused with a sign-up request.

### Answer

Sign up and repeat your request.

### Sources

None shown.

---

## Q-CUDA13-001 · run 3

- Time: 13:12 PST
- Model or mode shown: not shown
- Answer shown: no
- Notes: Signed-out repeat of an identical prompt was refused with a sign-up request.

### Answer

Sign up and repeat your request.

### Sources

None shown.

---

## Q-CUDA13-002 · run 1

- Time: 13:12 PST
- Model or mode shown: not shown
- Answer shown: yes
- Notes: Page showed 11 sources; a sign-up wall then blocked the source panel. Inline citations only.

### Answer

No. CUDA Toolkit 13.x does not officially support Maxwell GPUs. Starting with CUDA 13.0, NVIDIA removed offline compilation and CUDA library support for Maxwell, Pascal, and Volta; supported architectures start at Turing.
docs.nvidia

What to use instead

Use CUDA Toolkit 12.x—CUDA 12.9 is the last toolkit release series before that removal—to build applications targeting Maxwell. NVIDIA explicitly states that toolkits through the 12.x series remain supported for building applications for these architectures.
docs.nvidia

This is a toolkit-support change, not a statement that existing Maxwell CUDA applications stop working. Applications built with an older compatible toolkit can continue running with a driver that supports the GPU; NVIDIA’s CUDA driver is backward-compatible with applications built using older CUDA versions. Upgrading the driver, however, does not restore Maxwell compilation or library support in CUDA 13.
docs.nvidia

### Sources

1. https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html

---

## Q-CUDA13-002 · run 2

- Time: 13:15 PST
- Model or mode shown: not shown
- Answer shown: yes
- Notes: Inline citations only (source panel behind sign-up wall). Full text re-read from the saved answer page after streaming finished.

### Answer

No. CUDA Toolkit 13 removed offline compilation and library support for NVIDIA Maxwell GPUs. This is a removal, not merely a deprecation; Pascal and Volta were dropped at the same time. CUDA 13 supports Turing and newer architectures.
docs.nvidia

What to use instead

For Maxwell, stay on CUDA Toolkit 12.x and compatible libraries. NVIDIA explicitly states that the 12.x toolkits remain supported for building applications targeting Maxwell, while newer toolkits cannot target it.
docs.nvidia

The toolkit and driver are separate: removing Maxwell support from CUDA 13 does not mean existing CUDA 12.x applications immediately stop working. They can continue running with a driver that supports the GPU and meets the application’s requirements; NVIDIA documents backward compatibility for applications built with older CUDA toolkits.
docs.nvidia

### Sources

1. https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html

---

## Q-CUDA13-002 · run 3

- Time: 13:23 PST
- Model or mode shown: not shown
- Answer shown: yes
- Notes: Page showed 10 sources; inline citations only. The repeated list at the end is in the extracted page text (rendered list plus its text copy).

### Answer

No—CUDA Toolkit 13.x does not support targeting Maxwell GPUs. NVIDIA removed offline compilation and CUDA library support for Maxwell, Pascal, and Volta starting with CUDA 13.0. Maxwell has compute capability 5.x; CUDA 13’s compiler targets compute capability 7.5 (Turing) and newer.
docs.nvidia
+1

What this means in practice

To build applications for Maxwell, use CUDA Toolkit 12.9 or an earlier compatible release. CUDA 13’s nvcc cannot compile Maxwell-targeted code.
developer.nvidia

Existing applications built with CUDA 12.9 or earlier can still run on Maxwell, provided the application includes compatible GPU code and you use a driver that supports the GPU. The toolkit change does not automatically stop older applications from working.
developer.nvidia

NVIDIA identifies the R580 driver branch as the long-term-support option for these older architectures, with support through mid-2028.
developer.nvidia

The key distinction is toolkit support versus driver compatibility: a driver released alongside CUDA 13 can still run older CUDA applications on Maxwell, but that does not make Maxwell a supported target for the CUDA 13 toolkit.
developer.nvidia

To build applications for Maxwell, use CUDA Toolkit 12.9 or an earlier compatible release. CUDA 13’s nvcc cannot compile Maxwell-targeted code.
developer.nvidia

Existing applications built with CUDA 12.9 or earlier can still run on Maxwell, provided the application includes compatible GPU code and you use a driver that supports the GPU. The toolkit change does not automatically stop older applications from working.
developer.nvidia

NVIDIA identifies the R580 driver branch as the long-term-support option for these older architectures, with support through mid-2028.
developer.nvidia

### Sources

1. https://docs.nvidia.com/cuda/cuda-features-archive/index.html
2. https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/

---

## Q-CUDA13-003 · run 1

- Time: 13:13 PST
- Model or mode shown: not shown
- Answer shown: yes
- Notes: Page showed 10 sources; source panel behind sign-up wall. Inline citations only; the '+1' chip's second URL was not exposed.

### Answer

No. The GTX 980 Ti is a Maxwell GPU (compute capability 5.2), and CUDA 13 drops Maxwell support; current default PyTorch wheels also exclude Maxwell. Use an older PyTorch/CUDA build that still supports it, or a CPU-only build.
dartmouth
+1

### Sources

1. https://rc.dartmouth.edu/hpc-cookbook/recipes/python/pytorch/

---

## Q-CUDA13-003 · run 2

- Time: 13:15 PST
- Model or mode shown: not shown
- Answer shown: no
- Notes: Signed-out repeat refused with a sign-up request.

### Answer

Sign up and repeat your request.

### Sources

None shown.

---

## Q-CUDA13-003 · run 3

- Time: 13:23 PST
- Model or mode shown: not shown
- Answer shown: no
- Notes: Signed-out repeat refused with a sign-up request.

### Answer

Sign up and repeat your request.

### Sources

None shown.

---

## Q-CUDA13-004 · run 1

- Time: 13:18 PST
- Model or mode shown: not shown
- Answer shown: yes
- Notes: Page showed 10 sources; inline citations only; the '+1' chip's second URL was not exposed.

### Answer

It means the CUDA kernel in your PyTorch build has no compatible machine code for your GPU’s architecture, so the GPU can’t run it. This often happens when the installed PyTorch build or a custom CUDA extension doesn’t support your GPU’s compute capability.
localaimaster
+1

### Sources

1. https://localaimaster.com/blog/no-kernel-image-available-rtx-50

---

## Q-CUDA13-004 · run 2

- Time: 13:24 PST
- Model or mode shown: not shown
- Answer shown: no
- Notes: Signed-out repeat refused with a sign-up request.

### Answer

Sign up and repeat your request.

### Sources

None shown.

---

## Q-CUDA13-004 · run 3

- Time: 13:30 PST
- Model or mode shown: not shown
- Answer shown: no
- Notes: Signed-out repeat refused with a sign-up request.

### Answer

Sign up and repeat your request.

### Sources

None shown.

---

## Q-CUDA13-005 · run 1

- Time: 13:16 PST
- Model or mode shown: not shown
- Answer shown: yes
- Notes: Page showed 10 sources; inline citations only. Live view was cut off by the sign-up wall; full text read from the saved answer page.

### Answer

There isn’t a general CUDA compatibility layer that makes an unsupported old GPU work with newer CUDA. NVIDIA’s forward-compatibility package addresses certain driver/toolkit mismatches on eligible GPUs; it does not add support for an unsupported GPU.
docs.nvidia

Check your GPU’s compute capability and the CUDA version your software requires, then use a toolkit and driver that support both. If the GPU is too old for that software, use an older supported software/CUDA release or run it on a newer GPU.
docs.nvidia

### Sources

1. https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html
2. https://docs.nvidia.com/cuda/cuda-installation-guide-linux/

---

## Q-CUDA13-005 · run 2

- Time: 13:24 PST
- Model or mode shown: not shown
- Answer shown: no
- Notes: Signed-out repeat refused with a sign-up request.

### Answer

Sign up and repeat your request.

### Sources

None shown.

---

## Q-CUDA13-005 · run 3

- Time: 13:31 PST
- Model or mode shown: not shown
- Answer shown: no
- Notes: Signed-out repeat refused with a sign-up request.

### Answer

Sign up and repeat your request.

### Sources

None shown.

---

## Q-CUDA13-006 · run 1

- Time: 13:19 PST
- Model or mode shown: not shown
- Answer shown: yes
- Notes: Page showed 10 sources; inline citations only.

### Answer

None, in general. CUDA 13 drops support for Maxwell GPUs, including sm_52, so a standard CUDA 13 PyTorch build cannot run CUDA kernels on it. Use a compatible CUDA 12.x build instead.
journal.qualiteg

### Sources

1. https://journal.qualiteg.com/pytorch_and_supported_gpu_version/

---

## Q-CUDA13-006 · run 2

- Time: 13:25 PST
- Model or mode shown: not shown
- Answer shown: no
- Notes: Signed-out repeat refused with a sign-up request.

### Answer

Sign up and repeat your request.

### Sources

None shown.

---

## Q-CUDA13-006 · run 3

- Time: 13:31 PST
- Model or mode shown: not shown
- Answer shown: no
- Notes: Signed-out repeat refused with a sign-up request.

### Answer

Sign up and repeat your request.

### Sources

None shown.

---

## Q-CUDA13-007 · run 1

- Time: 13:19 PST
- Model or mode shown: not shown
- Answer shown: yes
- Notes: Page showed 10 sources; inline citations only. The answer interpreted 'this tool' as an unrelated project.

### Answer

If by “Maxwell” you mean the max_training project, yes—but it’s limited: compiled training supports plain SGD, while eager mode supports SGD, SGD momentum, and prototype RMSprop; autograd has prototype backward rules for only a small set of tensor operations. It also supports single-device tensors and lacks features such as convolution and DataLoader.
github

### Sources

1. https://github.com/YichengDWu/max_training

---

## Q-CUDA13-007 · run 2

- Time: 13:26 PST
- Model or mode shown: not shown
- Answer shown: no
- Notes: Signed-out repeat refused with a sign-up request.

### Answer

Sign up and repeat your request.

### Sources

None shown.

---

## Q-CUDA13-007 · run 3

- Time: 13:32 PST
- Model or mode shown: not shown
- Answer shown: no
- Notes: Signed-out repeat refused with a sign-up request.

### Answer

Sign up and repeat your request.

### Sources

None shown.

---

## Q-CUDA13-008 · run 1

- Time: 13:20 PST
- Model or mode shown: not shown
- Answer shown: yes
- Notes: Page showed 10 sources; inline citations only; the '+1' chip's second URL was not exposed.

### Answer

Usually not. GPUs are typically faster for large, parallel workloads, while CPUs can match or beat them on small or latency-sensitive tasks; CPU fallback may also add overhead if work is split between CPU and GPU.
system-design
+1

### Sources

1. https://system-design.space/en/chapter/cpu-gpu-overview/

---

## Q-CUDA13-008 · run 2

- Time: 13:27 PST
- Model or mode shown: not shown
- Answer shown: no
- Notes: Signed-out repeat refused with a sign-up request.

### Answer

Sign up and repeat your request.

### Sources

None shown.

---

## Q-CUDA13-008 · run 3

- Time: 13:32 PST
- Model or mode shown: not shown
- Answer shown: no
- Notes: Signed-out repeat refused with a sign-up request.

### Answer

Sign up and repeat your request.

### Sources

None shown.

---

## A-001 · run 1

- Time: 13:21 PST
- Model or mode shown: not shown
- Answer shown: yes
- Notes: Page showed 24 sources; inline citations only. The answer says cudacompat.com 'failed to load in my lookup'. The duplicated list at the end is in the page text as extracted (rendered list plus its text copy).

### Answer

I couldn’t verify what cudacompat.com offers: the site failed to load in my lookup, and searches didn’t return a description of that specific domain. I therefore can’t confirm its features, ownership, or affiliation with NVIDIA.

What “CUDA compatibility” means

In NVIDIA’s documentation, CUDA compatibility describes which combinations of CUDA applications, toolkits, and NVIDIA drivers can work together—especially when software and drivers are upgraded on different schedules.
nvidia

It covers three main relationships:

Backward compatibility: newer NVIDIA drivers can run applications built with older CUDA toolkits.
docs.nvidia

Minor-version compatibility: starting with CUDA 11, applications within the same major CUDA release family can run on a sufficiently recent driver, subject to feature limitations.
nvidia

Forward compatibility: on supported systems, a cuda-compat-<major>-<minor> package supplies newer user-mode driver components so newer CUDA software can run with an older base driver, within documented limits.
docs.nvidia

These concepts explain the name, but they don’t establish what cudacompat.com itself does. NVIDIA’s verified reference is its CUDA Compatibility documentation on docs.nvidia.com.
nvidia

If you paste the site’s text or a screenshot, I can explain the specific service rather than just the underlying term.

Backward compatibility: newer NVIDIA drivers can run applications built with older CUDA toolkits.
docs.nvidia

Minor-version compatibility: starting with CUDA 11, applications within the same major CUDA release family can run on a sufficiently recent driver, subject to feature limitations.
nvidia

Forward compatibility: on supported systems, a cuda-compat-<major>-<minor> package supplies newer user-mode driver components so newer CUDA software can run with an older base driver, within documented limits.
docs.nvidia

### Sources

1. https://docs.nvidia.com/deploy/cuda-compatibility/index.html
2. https://docs.nvidia.com/deploy/cuda-compatibility/latest/frequently-asked-questions.html
3. https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html

---

## A-001 · run 2

- Time: 13:27 PST
- Model or mode shown: not shown
- Answer shown: yes
- Notes: Page showed 25 sources; inline citations only. Again reports that cudacompat.com 'failed to load in my browsing tool'. The repeated list at the end is in the extracted page text.

### Answer

I couldn’t verify what “CUDA Compatibility” at cudacompat.com offers: the site failed to load in my browsing tool, and searches did not surface a page from that domain. That means I can’t reliably describe its features, ownership, or affiliation with NVIDIA.

What the name refers to

In NVIDIA’s official documentation, CUDA compatibility describes how CUDA applications and toolkit components can run with different NVIDIA driver versions. It helps developers upgrade CUDA software without necessarily upgrading the system’s GPU driver at the same time, within documented limits.
nvidia

NVIDIA distinguishes two important mechanisms:

Minor-version compatibility: applications built within the same major CUDA release family can run on a sufficiently recent driver, with some feature limitations. This is available starting with CUDA 11.
nvidia

Forward compatibility: a cuda-compat-<major>-<minor> package lets newer CUDA software run on certain older base drivers, subject to GPU and platform restrictions.
nvidia

That explains the technical term, but it does not establish what cudacompat.com specifically provides. Paste the site’s description or a screenshot, and I can explain that particular service accurately.

Minor-version compatibility: applications built within the same major CUDA release family can run on a sufficiently recent driver, with some feature limitations. This is available starting with CUDA 11.
nvidia

Forward compatibility: a cuda-compat-<major>-<minor> package lets newer CUDA software run on certain older base drivers, subject to GPU and platform restrictions.
nvidia

### Sources

1. https://docs.nvidia.com/deploy/cuda-compatibility/index.html

---

## A-001 · run 3

- Time: 13:33 PST
- Model or mode shown: not shown
- Answer shown: yes
- Notes: Page showed 25 sources; inline citations only. Cites https://www.cudacompat.com/ (the www host, which 308-redirects to the apex). The answer says it used the site's search-indexed description because the site 'failed to load through my retrieval tool'.

### Answer

CUDA Compatibility at cudacompat.com is described as an open-source project for running selected CUDA 13 PyTorch operations on older NVIDIA GPUs, specifically including Maxwell GPUs with compute capability sm_52. Its approach combines native GPU kernels with CPU fallbacks, rather than promising full support for every CUDA 13 workload.
cudacompat

What that means

The project aims to keep older hardware usable for some newer PyTorch workloads. Its indexed description explicitly says “selected” operations, so it should not be read as a guarantee that an arbitrary CUDA 13 application or PyTorch model will run unchanged or entirely on the GPU.
cudacompat

How it differs from NVIDIA’s CUDA Compatibility

The similarly named NVIDIA documentation addresses a different problem: compatibility between CUDA software and NVIDIA driver versions. NVIDIA describes minor-version compatibility within a CUDA major-release family and forward compatibility through cuda-compat-<major>-<minor> packages, subject to platform and GPU restrictions. That is distinct from cudacompat.com’s stated focus on selected operations for older GPU hardware.
nvidia
+1

I could retrieve the site’s search-indexed description, but the website itself failed to load through my retrieval tool. Consequently, I can’t verify its current installation instructions, full hardware and operation coverage, maintainers, or performance claims.

### Sources

1. https://www.cudacompat.com/
2. https://docs.nvidia.com/deploy/cuda-compatibility/latest/

---

## A-002 · run 1

- Time: 13:08 PST
- Model or mode shown: not shown
- Answer shown: yes
- Notes: Method test run in the parent session (tab-1). Full source panel: 10 sources; none is cudacompat.com.

### Answer

I couldn’t verify who created the project from the available information.

### Sources

1. https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html
2. https://docs.nvidia.com/deploy/cuda-compatibility/latest/pdf/CUDA_Compatibility.pdf
3. https://docs.nvidia.com/deploy/cuda-compatibility/610/forward-compatibility.html
4. https://en.wikipedia.org/wiki/CUDA
5. https://github.com/NVIDIA/nvidia-container-toolkit/blob/main/cmd/nvidia-cdi-hook/cudacompat/cudacompat.go
6. https://github.com/conda-forge/cuda-compat-feedstock
7. https://github.com/chuckatkins/cuda-autocompat
8. https://github.com/NVIDIA/cuda-python/pull/1412
9. https://gpusmith.com/articles/en/cuda-compatibility-matrix-driver-toolkit-container-gpu
10. https://www.scribd.com/document/576550415/CUDA-Compatibility

---

## A-002 · run 2

- Time: 13:28 PST
- Model or mode shown: not shown
- Answer shown: no
- Notes: Signed-out repeat refused with a sign-up request.

### Answer

Sign up and repeat your request.

### Sources

None shown.

---

## A-002 · run 3

- Time: 13:34 PST
- Model or mode shown: not shown
- Answer shown: no
- Notes: Signed-out repeat refused with a sign-up request.

### Answer

Sign up and repeat your request.

### Sources

None shown.

---

## A-003 · run 1

- Time: 13:21 PST
- Model or mode shown: not shown
- Answer shown: yes
- Notes: Page showed 25 sources; inline citations only. The answer says the site 'failed to load through my browsing tool'.

### Answer

I couldn’t verify that cudacompat.com is an official NVIDIA project. Treat its NVIDIA affiliation as unverified, rather than assuming it is official.

I attempted to retrieve the website, but it failed to load through my browsing tool. Searches for the exact domain also did not surface evidence establishing NVIDIA ownership or endorsement. That isn’t proof that it’s unofficial; it means I can’t confidently authenticate it.

Official NVIDIA resources

NVIDIA publishes its CUDA compatibility documentation on docs.nvidia.com. It explains driver/toolkit compatibility and the official cuda-compat-<major>-<minor> forward-compatibility packages. Those packages should not be confused with ownership of the similarly named domain.
docs.nvidia

For authoritative information, use:

NVIDIA’s CUDA Compatibility documentation for driver and toolkit compatibility.
docs.nvidia

NVIDIA’s CUDA GPU Compute Capability page for GPU architecture support.
developer.nvidia

Until there is a verifiable NVIDIA statement linking to cudacompat.com or identifying it as an NVIDIA-operated service, I would not describe it as official.

NVIDIA’s CUDA Compatibility documentation for driver and toolkit compatibility.
docs.nvidia

NVIDIA’s CUDA GPU Compute Capability page for GPU architecture support.
developer.nvidia

### Sources

1. https://docs.nvidia.com/deploy/cuda-compatibility/latest/index.html
2. https://developer.nvidia.com/cuda/gpus

---

## A-003 · run 2

- Time: 13:29 PST
- Model or mode shown: not shown
- Answer shown: yes
- Notes: Page showed 25 sources; inline citations only. Says it 'couldn’t retrieve the website'. The repeated list at the end is in the extracted page text.

### Answer

I can’t verify that cudacompat.com is an official NVIDIA project. Its affiliation is unverified: I couldn’t retrieve the website, and searches for the exact domain did not surface evidence establishing NVIDIA ownership or endorsement. That does not prove it is unofficial, but I would not present it as an NVIDIA resource.

NVIDIA’s official resources for the related information are:

CUDA Compatibility documentation
, covering compatibility between CUDA applications, toolkit components, and NVIDIA driver versions.
docs.nvidia

CUDA GPU Compute Capability tables
, listing GPU architectures and their supported compute capabilities.
developer.nvidia

Use those sources to verify compatibility claims from cudacompat.com rather than treating the domain’s CUDA-related name as evidence of affiliation.

CUDA Compatibility documentation
, covering compatibility between CUDA applications, toolkit components, and NVIDIA driver versions.
docs.nvidia

CUDA GPU Compute Capability tables
, listing GPU architectures and their supported compute capabilities.
developer.nvidia

### Sources

1. https://docs.nvidia.com/deploy/cuda-compatibility/latest/index.html
2. https://developer.nvidia.com/cuda/gpus

---

## A-003 · run 3

- Time: 13:34 PST
- Model or mode shown: not shown
- Answer shown: yes
- Notes: Page showed 25 sources; inline citations only. Says it 'couldn’t retrieve the site'.

### Answer

I couldn’t verify that cudacompat.com is an official NVIDIA project. I couldn’t retrieve the site, and the search results did not establish NVIDIA ownership or endorsement. That leaves its status unconfirmed—not proof that it is unofficial.

NVIDIA’s official CUDA Compatibility documentation is hosted at 
docs.nvidia.com
. It explains how CUDA applications and toolkit components work across NVIDIA driver versions.
nvidia

There is also an official NVIDIA package named cuda-compat, distributed through NVIDIA’s CUDA repositories and installers. That package’s name should not be taken as evidence that the similarly named cudacompat.com website is affiliated with NVIDIA.
nvidia

### Sources

1. https://docs.nvidia.com/deploy/cuda-compatibility/latest/
2. https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html

---

## A-004 · run 1

- Time: 13:22 PST
- Model or mode shown: not shown
- Answer shown: no
- Notes: First run of this prompt refused with a sign-up request (signed-out limit).

### Answer

Sign up and repeat your request.

### Sources

None shown.

---

## A-004 · run 2

- Time: 13:29 PST
- Model or mode shown: not shown
- Answer shown: no
- Notes: Signed-out request refused with a sign-up request.

### Answer

Sign up and repeat your request.

### Sources

None shown.

---

## A-004 · run 3

- Time: 13:35 PST
- Model or mode shown: not shown
- Answer shown: no
- Notes: Signed-out request refused with a sign-up request.

### Answer

Sign up and repeat your request.

### Sources

None shown.
