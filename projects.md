---
layout: default
title: Projects
---
<h1>Projects</h1>

<h2><a href="https://github.com/deciding/handclaw" target="_blank">HandClaw</a></h2>
<p><em>One Slack = Multiple AI Coding Agents</em></p>
<p>Connect Claude Code, Codex, and OpenCode to Slack. Code anywhere from anything. Work from your phone. Monitor agent progress. Switch agents by renaming channels.</p>

<h3>Key Features</h3>
<ul>
  <li><strong>One workspace = Multiple agents</strong></li>
  <li><strong>Code from phone</strong>, even from your watch</li>
  <li><strong>Walk away, let agents work</strong></li>
  <li>Supports persistent plan/build mode switching</li>
  <li>Notifies users when coding tasks are completed</li>
</ul>

<h3>HandClaw vs OpenClaw</h3>
<table>
  <tr>
    <th>Feature</th>
    <th>HandClaw</th>
    <th>OpenClaw</th>
  </tr>
  <tr>
    <td>Switch plan/build mode</td>
    <td>✓ (`!code switch plan/build`)</td>
    <td>✗</td>
  </tr>
  <tr>
    <td>Early stop code CLI</td>
    <td>✓ (`!stop`)</td>
    <td>✗</td>
  </tr>
  <tr>
    <td>Project management via channels</td>
    <td>✓ (just rename channel)</td>
    <td>✗ (need install acpx + complex config)</td>
  </tr>
  <tr>
    <td>Support ACP</td>
    <td>✓ Easy (rename channel)</td>
    <td>✗ Complex</td>
  </tr>
</table>

<hr>

<h2><a href="https://github.com/deciding/txl" target="_blank">TeraXLang</a></h2>
<p><em>Triton Extension for LLM. As fast as FlashAttention</em></p>
<p>A CUDA kernel-specific DSL built on top of Triton that achieves SOTA GPU kernel performance on both Hopper (H100) and Blackwell (B200) architectures.</p>

<h3>Why TeraXLang?</h3>
<ul>
  <li>What optimizations has Triton done?</li>
  <li>Why do many DSLs claim they can easily outperform Triton?</li>
  <li>What if we add a few more APIs that might harm Triton's generality, but bring superior performance in exchange?</li>
</ul>

<h3>Key Features</h3>
<ul>
  <li><strong>Minimal Extensions:</strong> Adds only essential methods to Triton (smem, tmem, mbar, TMA operations)</li>
  <li><strong>Warp-level Primitives:</strong> Efficient warpgroup synchronization and reduction</li>
  <li><strong>TMA Support:</strong> Hardware-accelerated tensor memory operations</li>
  <li><strong>Multi-Architecture:</strong> Optimized for both Hopper and Blackwell GPUs</li>
</ul>

<h3>Performance</h3>

<h4>Matmul (H100 80GB HBM3)</h4>
<p><em>M=8192, N=8192, K=1024</em></p>
<table>
  <tr>
    <th>Kernel</th>
    <th>TFLOPS</th>
  </tr>
  <tr>
    <td>cuBLAS</td>
    <td>710.4</td>
  </tr>
  <tr>
    <td>TXL (hopper_txl_ws_persistent)</td>
    <td>697.7 (~2% slower)</td>
  </tr>
</table>

<h4>Flash Attention (H100 80GB HBM3)</h4>
<p><em>batch=16, heads=32, seq_len=16384, head_dim=128</em></p>
<table>
  <tr>
    <th>Kernel</th>
    <th>TFLOPS</th>
  </tr>
  <tr>
    <td>FlashAttention3</td>
    <td>640</td>
  </tr>
  <tr>
    <td>TXL (hopper_txl_ws_fa3)</td>
    <td>676.26 (~6% faster)</td>
  </tr>
</table>

<h4>MLA Decoding (H100 80GB HBM3)</h4>
<table>
  <tr>
    <th>Kernel</th>
    <th>Time (ms)</th>
    <th>TFLOPS</th>
  </tr>
  <tr>
    <td>HuggingFace MLA</td>
    <td>2.03</td>
    <td>592</td>
  </tr>
  <tr>
    <td>TXL MLA</td>
    <td>2.22</td>
    <td>754</td>
  </tr>
</table>

<h4>NSA Prefill (H100 80GB HBM3)</h4>
<table>
  <tr>
    <th>Kernel</th>
    <th>Time (us)</th>
    <th>TFLOPS</th>
  </tr>
  <tr>
    <td>FlashNSA</td>
    <td>235</td>
    <td>2248.4</td>
  </tr>
  <tr>
    <td>TXL NSA</td>
    <td>219</td>
    <td>266.4</td>
  </tr>
</table>
