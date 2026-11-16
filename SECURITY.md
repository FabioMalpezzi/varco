# Security

varco opens a channel from a container to the Mac's GPU: a program in a
container sends its array operations to a process on the Mac, which runs
them on the GPU as your user. [docs/security.md](docs/security.md) says what
the Mac end checks and what is left to whoever starts it. In short:

- Each bridge runs only the operations on its list: 114 MLX primitives,
  each with a rule on its inputs, and 1,972 of PyTorch 2.15.0's 3,120
  operators. Anything else is refused.
- On a TCP port (Docker Desktop) every container of the engine and every
  local process can connect to a server. With Kiri's channel, and no server
  on a TCP port, only a container started with `--gpus` does.
- The MLX server reads files only under the directories it is started with.
- Custom Metal kernels are refused unless the server is started with
  `--custom-kernels`. With the option on, the containers on that server
  must trust each other
  ([docs/use-cases/custom-kernels.md](docs/use-cases/custom-kernels.md)).
- Containers that share one MLX server are not separated from each other.
  The PyTorch bridge gives every program its own process.

The code was reviewed adversarially twice before publication, and no
outside party has audited it. Give the channel to containers you run
yourself, or trust as you would a program you start on the Mac.

## Reporting a problem

Please report security problems privately, through the "Report a
vulnerability" form of the repository's Security tab, rather than in a public
issue. Say what you ran, on which engine (Docker Desktop or Kiri), and what a
container was able to do. There is no bounty and no guaranteed response time:
this is a research prototype maintained on a best-effort basis.

## Supported versions

Only the latest release.
