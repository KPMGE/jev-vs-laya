# Laya: install and serve locally on Apple Silicon

Type: research
Status: open
Blocked by:

## Question

How do we install and run Laya on an Apple M5 (16 GB, macOS) so it answers decisions through a Jev-compatible HTTP API? Pin down: the Python package name and version, the Hugging Face checkpoint(s) and their sizes, the serving command (official server, Railway template's image, or MLX port), whether it runs on MPS/MLX or CPU only, and the request/response shape compared to Jev's (endpoint paths, question types such as Choice / Score / Noul, how confidence is returned). Can one client talk to both?
