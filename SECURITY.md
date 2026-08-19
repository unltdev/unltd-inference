# Security Policy

## Supported Versions

UNLTD Inference currently provides security updates for the latest stable release.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

Please do **not** open a public GitHub issue for security vulnerabilities.

If you believe you have found a security issue in UNLTD Inference, report it privately by contacting:

**security@unltd.com.ar**

Please include, when possible:

- a clear description of the vulnerability;
- affected version;
- steps to reproduce;
- operating system and environment;
- expected vs. actual behavior;
- proof-of-concept code or logs, if relevant;
- any suggested mitigation or fix.

We will make a reasonable effort to acknowledge valid reports within **7 days** and provide status updates as the issue is investigated.

If the vulnerability is confirmed, we will:

1. assess its severity and affected versions;
2. prepare and test a fix;
3. publish a patched release when appropriate;
4. document the issue in the release notes or security advisory when disclosure is safe.

If the report is determined not to be a security vulnerability, we will explain the reasoning and may suggest opening a regular GitHub issue instead.

## Scope

Security reports may include issues such as:

- malformed or malicious GGUF files causing memory safety problems or unexpected execution;
- denial-of-service conditions caused by crafted model metadata or tensor layouts;
- unsafe memory handling;
- integer overflows or bounds-checking failures;
- vulnerabilities in parsing, memory mapping, tokenization, or inference logic;
- dependency vulnerabilities that directly affect UNLTD Inference.

Performance issues, numerical differences from other runtimes, unsupported models, and general bugs should be reported through regular GitHub Issues unless they have a clear security impact.

## Responsible Disclosure

Please allow reasonable time for investigation and remediation before publicly disclosing a vulnerability.

We appreciate responsible security research and reports that help improve UNLTD Inference.
