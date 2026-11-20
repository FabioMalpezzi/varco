"""Opt-in, single-Mac MLX-LM server with bounded short-prompt priority."""
import logging
from importlib.metadata import version

from mlx_lm import server
from mlx_lm.generate import BatchGenerator


class PriorityBatch:
    """Two generators share one model, one stream and one HTTP request ID space."""

    def __init__(self, model, **options):
        long_options = dict(options, prefill_step_size=min(512, options["prefill_step_size"]))
        self.long = BatchGenerator(model, **long_options)
        self.short = BatchGenerator(model, **options)
        self.long_ids = set()
        self.short_ids = set()
        self.next_uid = 0
        self.short_turns = 0

    def insert_segments(self, **request):
        # MLX-LM 0.32.2 admits one HTTP request at a time, after prefix-cache lookup.
        segments, = request["segments"]
        if sum(map(len, segments)) <= 64:
            generator, pending = self.short, self.short_ids
        else:
            generator, pending = self.long, self.long_ids
        generator._uid_count = self.next_uid
        uid, = generator.insert_segments(**request)
        self.next_uid = uid + 1
        pending.add(uid)
        return [uid]

    def next(self):
        if self.short_ids and (not self.long_ids or self.short_turns < 4):
            generator, pending = self.short, self.short_ids
            self.short_turns = self.short_turns + 1 if self.long_ids else 0
        else:
            generator, pending = self.long, self.long_ids
            self.short_turns = 0
        progress, tokens = generator.next()
        for result in tokens:
            if result.finish_reason is not None:
                pending.discard(result.uid)
        return progress, tokens

    def extract_cache(self, uids):
        caches = {}
        for generator, pending in ((self.long, self.long_ids), (self.short, self.short_ids)):
            selected = pending.intersection(uids)
            if selected:
                caches.update(generator.extract_cache(selected))
        return caches

    @property
    def prompt_cache_nbytes(self):
        return self.long.prompt_cache_nbytes + self.short.prompt_cache_nbytes

    def remove(self, uids):
        for generator, pending in ((self.long, self.long_ids), (self.short, self.short_ids)):
            selected = pending.intersection(uids)
            if selected:
                generator.remove(selected)
                pending.difference_update(selected)
                logging.info("Removed %d cancelled priority-server request(s)", len(selected))

    def close(self):
        # Each generator restores the previous Metal wired limit on close.
        self.short.close()
        self.long.close()


class PriorityWorker(server.ResponseGenerator):
    def __init__(self, *args, **kwargs):
        self.admissions = 0
        super().__init__(*args, **kwargs)

    def _generate(self):
        # Return to the HTTP admission queue after each generation step.
        self._time_budget = range(1)
        super()._generate()

    def _next_request(self, timeout=None):
        # When a batch exists, continuous HTTP arrivals must also yield GPU turns.
        if timeout is None and self.admissions >= 8:
            self.admissions = 0
            return None
        request = super()._next_request(timeout)
        self.admissions = self.admissions + 1 if request is not None else 0
        return request


def main():
    for package, required in (("mlx-lm", "0.32.2"), ("mlx", "0.32.4")):
        if version(package) != required:
            raise SystemExit(f"Varco priority server requires {package} {required}. Run mlx/setup-mac.sh.")
    # These hooks belong to this opt-in process. MLX-LM owns the HTTP handler,
    # model loading, prefix caches, disconnect handling and sequential fallback.
    server.BatchGenerator = PriorityBatch
    server.ResponseGenerator = PriorityWorker
    if server.mx.distributed.init().size() != 1:
        raise SystemExit("The priority server requires a single Mac process.")
    server.main()


if __name__ == "__main__":
    main()
