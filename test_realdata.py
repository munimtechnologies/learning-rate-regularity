# SPDX-License-Identifier: Apache-2.0
# Unit tests for realdata.py (standard library only; no R, no download). Made with AI agents working in MV2's name.
import io
import os
import tempfile
import unittest

import realdata as rd


class Parsing(unittest.TestCase):
    def test_parse_kdd(self):
        lines = [
            "Row\tAnon Student Id\tProblem Name\tCorrect First Attempt\tKC(Default)\tOpportunity(Default)\n",
            "1\tS1\tP\"q\"\t1\tA\t1\n",
            "2\tS1\tP\t0\tA~~B\t2~~1\n",
            "3\tS1\tP\t0\tA\t3\r\n",
            "4\tS2\tP\t1\t\t\n",
            "5\tS2\tP\t1\tB\t1\n",
            "6\tS2\n",
        ]
        self.assertEqual(list(rd.parse_kdd(lines)), [("S1", "A", 0, 1), ("S1", "A", 2, 0), ("S2", "B", 0, 1)])

    def test_parse_assist(self):
        text = ("order_id,user_id,original,correct,skill_id\n"
                "10,u1,1,0,5\n"
                "11,u1,1,1,5\n"
                "12,u1,1,1,5\n"
                "12,u1,1,1,7\n"
                "13,u1,0,1,5\n"
                "9,u2,1,1,7\n"
                "14,u2,1,0,\n"
                "15,u1,1,1,7\n")
        got = list(rd.parse_assist(io.StringIO(text, newline="")))
        self.assertEqual(got, [("u2", "7", 0, 1), ("u1", "5", 0, 0), ("u1", "5", 1, 1), ("u1", "7", 0, 1)])

    def test_parse_assist_bom(self):
        text = chr(0xEF) + chr(0xBB) + chr(0xBF) + "order_id,user_id,correct,skill_id\n1,u,1,3\n"
        self.assertEqual(list(rd.parse_assist(io.StringIO(text, newline=""))), [("u", "3", 0, 1)])


RECS = [("S1", "A", 0, 1), ("S1", "A", 1, 0), ("S1", "A", 2, 1), ("S1", "B", 0, 0),
        ("S2", "A", 0, 1), ("S2", "C", 0, 1), ("S2", "C", 30, 1),
        ("S3", "A", 0, 0), ("S3", "B", 1, 1)]


class Selection(unittest.TestCase):
    def test_select(self):
        sel, kcs, students = rd.select(RECS, 2, 10, 25, 1)
        self.assertEqual(kcs, ["A", "B"])
        self.assertEqual(students, ["S1", "S2", "S3"])
        self.assertEqual(len(sel), 7)
        self.assertTrue(all(r[1] in ("A", "B") and r[2] < 25 for r in sel))

    def test_select_caps_students(self):
        sel, kcs, students = rd.select(RECS, 3, 2, 25, 1)
        self.assertEqual(len(students), 2)
        self.assertTrue(set(students) <= {"S1", "S2", "S3"})
        self.assertTrue(all(r[0] in students for r in sel))

    def test_describe(self):
        sel, kcs, students = rd.select(RECS, 2, 10, 25, 1)
        d = rd.describe(sel, kcs, students)
        self.assertEqual((d["rows"], d["students"], d["kcs"], d["sequences"]), (7, 3, 2, 5))
        self.assertEqual(d["share_sequences_reaching_t5"], 0.0)

    def test_write_window_csv_bootstrap(self):
        sel, kcs, students = rd.select(RECS, 2, 10, 25, 1)
        by_student = rd.build_index(sel)
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            rows = rd.write_window_csv(by_student, ["S3", "S1", "S3"], {"A": 0, "B": 1}, 1, path)
            self.assertEqual(rows, 4)
            with open(path) as fh:
                lines = fh.read().splitlines()
            self.assertEqual(lines, ["student,kc,t,y", "0,0,0,0", "1,0,0,1", "1,1,0,0", "2,0,0,0"])
        finally:
            os.remove(path)


class Stats(unittest.TestCase):
    def test_pct_and_r_est(self):
        self.assertAlmostEqual(rd.pct([4, 1, 3, 2], 0.5), 2.5)
        self.assertEqual(rd.pct([7.0], 0.95), 7.0)
        self.assertIsNone(rd.pct([], 0.5))
        self.assertAlmostEqual(rd.r_est(0.1, 0.02), 1.3119, places=3)
        self.assertIsNone(rd.r_est(0.1, 0.2))

    def test_num(self):
        self.assertIsNone(rd._num("NA"))
        self.assertIs(rd._num("FALSE"), False)
        self.assertEqual(rd._num("12.5"), 12.5)


def fit(sd, aic=None):
    return {"sd_delta": sd, "gbar": 0.1, "sd_theta": 1.0, "aic": aic, "error": 0.0, "singular": False, "rows": 9}


def build(n_boot, log_short=0.06, skip_boot_for=()):
    results = {}
    for d in ("d1", "d2"):
        for b in range(n_boot + 1):
            if b > 0 and d in skip_boot_for:
                continue
            results[(d, "first5", "linear", b)] = fit(0.10)
            results[(d, "first25", "linear", b)] = fit(0.05, 1000.0)
            results[(d, "first5", "log", b)] = fit(log_short if d == "d2" else 0.06)
            results[(d, "first25", "log", b)] = fit(0.05, 990.0)
    return results


class Verdicts(unittest.TestCase):
    def test_all_held(self):
        v = rd.verdicts(build(20), ["d1", "d2"], 20)
        self.assertEqual(v["Q1"]["verdict"], "held")
        self.assertAlmostEqual(v["Q1"]["wdi_linear"]["d1"], 2.0)
        self.assertEqual(v["Q2"]["verdict"], "held")
        self.assertEqual(v["Q2"]["d2"]["boot_pairs"], 20)
        self.assertEqual(v["Q2"]["d2"]["share_log_lower"], 1.0)
        self.assertEqual(v["Q3"]["verdict"], "held")

    def test_q2_fails(self):
        v = rd.verdicts(build(20, log_short=0.12, skip_boot_for=("d1",)), ["d1", "d2"], 20)
        self.assertEqual(v["Q2"]["verdict"], "did not hold")
        self.assertEqual(v["Q2"]["d1"]["boot_pairs"], 0)

    def test_inconclusive(self):
        v = rd.verdicts(build(10), ["d1", "d2"], 10)
        self.assertTrue(v["Q2"]["verdict"].startswith("inconclusive"))
        self.assertEqual(v["Q1"]["verdict"], "held")
        r = build(0)
        r[("d1", "first25", "log", 0)]["error"] = 1.0
        v = rd.verdicts(r, ["d1", "d2"], 0)
        self.assertTrue(v["Q3"]["verdict"].startswith("inconclusive"))
        self.assertIsNone(v["Q3"]["d1"])

    def test_cells_and_markdown(self):
        r = build(20)
        table = rd.cells(r, ["d1", "d2"], 20)
        self.assertEqual(len(table), 2 * 3 * 2)
        first = table[0]
        self.assertEqual((first["dataset"], first["window"], first["model"]), ("d1", "first5", "linear"))
        self.assertEqual(first["boot_ok"], 20)
        missing = [c for c in table if c["window"] == "first10"]
        self.assertTrue(all(not c["ok"] and c["boot_ok"] == 0 for c in missing))
        fd, path = tempfile.mkstemp(suffix=".md")
        os.close(fd)
        try:
            rd.write_md(table, rd.verdicts(r, ["d1", "d2"], 20), {"boots_done": 20}, path)
            with open(path) as fh:
                text = fh.read()
            self.assertIn("| Q2 |", text)
            self.assertIn("| d1 | first5 | linear | 9 | 1.0000 | 0.1000 | [0.1000, 0.1000] |", text)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
