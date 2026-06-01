import tempfile
import unittest
from pathlib import Path

from models.tweaks import Tweaks
from run_experiments import (
    build_jobs,
    build_parser,
    collect_instances,
    parse_seed_list,
    validate_args,
)


class ExperimentRunnerTests(unittest.TestCase):
    def _build_args(self, tmp, extra_args):
        instance = Path(tmp) / "instance.txt"
        instance.write_text(
            "\n".join([
                "1 1 1",
                "5",
                "1 1 1",
                "0",
                "",
            ]),
            encoding="utf-8",
        )
        instances_file = Path(tmp) / "instances.txt"
        instances_file.write_text(f"{instance.as_posix()}\n", encoding="utf-8")

        parser = build_parser()
        args = parser.parse_args([
            "--instances-file",
            instances_file.as_posix(),
            "--seeds",
            "1",
            "--output-dir",
            (Path(tmp) / "output").as_posix(),
            *extra_args,
        ])
        validate_args(args)
        return args, collect_instances(args), parse_seed_list(args)

    def test_operator_group_ablation_includes_full_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, instances, seeds = self._build_args(
                tmp,
                ["--operator-group-ablations", "all"],
            )

            jobs = build_jobs(args, instances, seeds, "commit", "false")
            run_labels = {job["run_label"] for job in jobs}

            self.assertEqual(len(jobs), 4)
            self.assertIn("full", run_labels)
            self.assertIn("drop_order_operators", run_labels)
            self.assertIn("drop_insert_operators", run_labels)
            self.assertIn("drop_strategic_operators", run_labels)

    def test_operator_group_ablation_can_skip_full(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, instances, seeds = self._build_args(
                tmp,
                ["--operator-group-ablations", "all", "--skip-full"],
            )

            jobs = build_jobs(args, instances, seeds, "commit", "false")
            run_labels = {job["run_label"] for job in jobs}

            self.assertEqual(len(jobs), 3)
            self.assertNotIn("full", run_labels)
            self.assertEqual(
                run_labels,
                {
                    "drop_order_operators",
                    "drop_insert_operators",
                    "drop_strategic_operators",
                },
            )

    def test_specific_operator_ablation_can_skip_full(self):
        operator = sorted(Tweaks.operator_labels())[0]
        with tempfile.TemporaryDirectory() as tmp:
            args, instances, seeds = self._build_args(
                tmp,
                ["--operator-ablations", operator, "--skip-full"],
            )

            jobs = build_jobs(args, instances, seeds, "commit", "false")

            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]["run_label"], f"drop_{operator}")
            self.assertEqual(jobs[0]["component"], "operator_ablation")
            self.assertEqual(jobs[0]["omitted_operator"], operator)
            self.assertNotIn(operator, jobs[0]["operators"])


if __name__ == "__main__":
    unittest.main()
