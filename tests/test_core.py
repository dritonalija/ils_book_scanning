import os
import tempfile
import unittest

from models.local_search import LocalSearch
from models.parser import Parser
from models.solution import Solution
from validator.validator import validate_solution


class CoreAlgorithmTests(unittest.TestCase):
    def _write_instance(self, directory, content):
        path = os.path.join(directory, "instance.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return path

    def _portfolio_instance(self, directory):
        return self._write_instance(
            directory,
            "\n".join([
                "6 3 7",
                "1 2 3 6 8 10",
                "4 2 2",
                "0 1 2 3",
                "3 1 2",
                "3 4 5",
                "2 2 1",
                "2 5",
                "",
            ]),
        )

    def _order_sensitive_instance(self, directory):
        return self._write_instance(
            directory,
            "\n".join([
                "4 2 2",
                "100 90 20 10",
                "1 1 1",
                "2",
                "2 1 2",
                "0 1",
                "",
            ]),
        )

    def test_fast_evaluate_matches_solution_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = self._portfolio_instance(tmp)
            data = Parser(input_path).parse()
            order = [0, 1, 2]

            solution = Solution.from_order(order, data)
            self.assertEqual(solution.fitness_score, data.fast_evaluate(order))

            data.assignment_mode = "sequential"
            sequential_solution = Solution.from_order(order, data)
            self.assertEqual(
                sequential_solution.fitness_score,
                data.fast_evaluate(order),
            )

    def test_exported_solution_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = self._portfolio_instance(tmp)
            data = Parser(input_path).parse()
            solution = Solution.from_order([0, 1, 2], data)
            output_path = os.path.join(tmp, "solution.txt")
            solution.export(output_path)

            self.assertEqual(
                validate_solution(input_path, output_path, isConsoleApplication=True),
                "Valid",
            )
            details = validate_solution(input_path, output_path)
            self.assertIn("Solution is valid!", details)
            self.assertIn(f"Total score: {solution.fitness_score}", details)
            self.assertNotIn("Fitness score:", details)

    def test_validator_rejects_duplicate_scanned_books(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = self._portfolio_instance(tmp)
            output_path = os.path.join(tmp, "bad_solution.txt")
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write("\n".join([
                    "2",
                    "0 1",
                    "3",
                    "1 1",
                    "3",
                    "",
                ]))

            self.assertEqual(
                validate_solution(input_path, output_path, isConsoleApplication=True),
                "Invalid",
            )
            self.assertIn(
                "repeats already scanned books",
                validate_solution(input_path, output_path),
            )

    def test_exact_no_proxy_move_accepts_strict_improvement(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = self._order_sensitive_instance(tmp)
            data = Parser(input_path).parse()
            current = Solution.from_order([0, 1], data)
            improved_order = [1, 0]
            stats = {"exact_checked": 0}

            candidate = LocalSearch._try_exact_move(
                improved_order,
                data,
                current,
                stats,
            )

            self.assertIsNotNone(candidate)
            self.assertGreater(candidate.fitness_score, current.fitness_score)
            self.assertEqual(stats["exact_checked"], 1)


if __name__ == "__main__":
    unittest.main()
