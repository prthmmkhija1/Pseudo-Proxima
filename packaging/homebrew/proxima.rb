# frozen_string_literal: true

# Homebrew formula for ProximA - Quantum Simulation Framework
# https://github.com/your-org/proxima

class Proxima < Formula
  include Language::Python::Virtualenv

  desc "High-performance quantum circuit simulation framework"
  homepage "https://github.com/your-org/proxima"
  url "https://github.com/your-org/proxima/archive/refs/tags/v0.3.0.tar.gz"
  sha256 "PLACEHOLDER_SHA256_HASH_REPLACE_WITH_ACTUAL_HASH"
  license "MIT"
  head "https://github.com/your-org/proxima.git", branch: "main"

  depends_on "python@3.11"
  depends_on "numpy"

  option "with-qiskit", "Install with Qiskit Aer backend support"
  option "with-cirq", "Install with Cirq backend support"
  option "with-quest", "Install with QuEST backend support"

  resource "pydantic" do
    url "https://files.pythonhosted.org/packages/pydantic/pydantic-2.5.0.tar.gz"
    sha256 "PLACEHOLDER_PYDANTIC_SHA256"
  end

  resource "click" do
    url "https://files.pythonhosted.org/packages/click/click-8.1.7.tar.gz"
    sha256 "PLACEHOLDER_CLICK_SHA256"
  end

  resource "rich" do
    url "https://files.pythonhosted.org/packages/rich/rich-13.7.0.tar.gz"
    sha256 "PLACEHOLDER_RICH_SHA256"
  end

  resource "fastapi" do
    url "https://files.pythonhosted.org/packages/fastapi/fastapi-0.109.0.tar.gz"
    sha256 "PLACEHOLDER_FASTAPI_SHA256"
  end

  resource "uvicorn" do
    url "https://files.pythonhosted.org/packages/uvicorn/uvicorn-0.27.0.tar.gz"
    sha256 "PLACEHOLDER_UVICORN_SHA256"
  end

  def install
    venv = virtualenv_create(libexec, "python3.11")
    venv.pip_install resources
    venv.pip_install_and_link buildpath
    if build.with?("qiskit")
      venv.pip_install "qiskit>=1.0.0"
      venv.pip_install "qiskit-aer>=0.13.0"
    end
    if build.with?("cirq")
      venv.pip_install "cirq>=1.3.0"
    end
  end

  def post_install
    (var/"proxima").mkpath
  end

  def caveats
    "ProximA installed! Run: proxima --help"
  end

  test do
    assert_match "proxima", shell_output("#{bin}/proxima --version")
    system libexec/"bin/python", "-c", "import proxima"
  end
end
