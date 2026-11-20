"""Regression checks for MLX-LM request IDs, cancellation and fairness."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import priority_server as priority


class PriorityPolicyTest(unittest.TestCase):
    def test_request_lifecycle_and_fair_turns(self):
        with patch.object(priority, "BatchGenerator") as factory:
            long, short = Mock(), Mock()
            factory.side_effect = (long, short)
            batch = priority.PriorityBatch(object(), prefill_step_size=2048)
            self.assertEqual(factory.call_args_list[0].kwargs["prefill_step_size"], 512)
            long.insert_segments.side_effect = lambda **kw: [long._uid_count]
            short.insert_segments.side_effect = lambda **kw: [short._uid_count]
            self.assertEqual(batch.insert_segments(segments=[[[1] * 65]]), [0])
            self.assertEqual(batch.insert_segments(segments=[[[2] * 63, [3]]]), [1])
            self.assertEqual(batch.insert_segments(segments=[[[4]]]), [2])
            long.next.return_value = (["long"], [])
            short.next.return_value = (["short"], [])
            self.assertEqual([batch.next()[0][0] for _ in range(10)], ["short"] * 4 + ["long"] + ["short"] * 4 + ["long"])
            short.extract_cache.return_value = {2: "prefix"}
            self.assertEqual(batch.extract_cache([2]), {2: "prefix"})
            long.extract_cache.assert_not_called()
            long.prompt_cache_nbytes, short.prompt_cache_nbytes = 100, 20
            self.assertEqual(batch.prompt_cache_nbytes, 120)
            batch.remove([1, 0, 999])
            long.remove.assert_called_once_with({0})
            short.remove.assert_called_once_with({1})
            short.next.return_value = ([], [SimpleNamespace(uid=2, finish_reason="stop")])
            batch.next()
            self.assertFalse(batch.short_ids or batch.long_ids)
            batch.close()
            short.close.assert_called_once()
            long.close.assert_called_once()

    def test_continuous_admission_yields_to_generation(self):
        worker = priority.PriorityWorker.__new__(priority.PriorityWorker)
        worker.admissions = 0
        with patch.object(priority.server.ResponseGenerator, "_next_request", return_value="request"):
            self.assertEqual([worker._next_request() for _ in range(10)], ["request"] * 8 + [None, "request"])


if __name__ == "__main__":
    unittest.main()
