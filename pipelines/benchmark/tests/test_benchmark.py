import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parents[1]
REPO=HERE.parents[1]
spec=importlib.util.spec_from_file_location('benchmark_tables',HERE/'reproduce_tables.py')
tables=importlib.util.module_from_spec(spec);spec.loader.exec_module(tables)


class DetectionEndpoints(unittest.TestCase):
    def test_localisation_requires_overlap(self):
        rows=[dict(reference_positive=1,reference_pixels=2,predicted_pixels=3,intersection_pixels=1,score=.9,overlap_score=.8),
              dict(reference_positive=1,reference_pixels=1,predicted_pixels=1,intersection_pixels=0,score=.7,overlap_score=.3),
              dict(reference_positive=0,reference_pixels=0,predicted_pixels=2,intersection_pixels=0,score=.6,overlap_score=''),
              dict(reference_positive=0,reference_pixels=0,predicted_pixels=0,intersection_pixels=0,score=.1,overlap_score='')]
        s=tables.summary(rows)
        self.assertEqual([s[x] for x in ['tp','fp','tn','fn']],[1,1,1,1])
        self.assertAlmostEqual(s['dice_positive_mean'],.2)
        self.assertAlmostEqual(s['dice_tp_mean'],.4)
        self.assertAlmostEqual(s['overlap_auroc'],.75)
        self.assertAlmostEqual(s['overlap_auprc_ap'],5/6)

    def test_tied_scores_receive_half_auc(self):
        rows=[dict(reference_positive=1,reference_pixels=1,predicted_pixels=1,intersection_pixels=1,score=.5,overlap_score=.5),
              dict(reference_positive=0,reference_pixels=0,predicted_pixels=1,intersection_pixels=0,score=.5,overlap_score='')]
        self.assertEqual(tables.summary(rows)['overlap_auroc'],.5)

    def test_single_seed_has_no_sample_sd(self):
        self.assertIsNone(tables.sd([.5]))
        self.assertIsNone(tables.avg([None]))


class RecordedInputs(unittest.TestCase):
    def test_experiment_matrix_is_complete(self):
        rows=json.loads((REPO/'configs/benchmark/model_recipes.json').read_text())
        self.assertEqual(len(rows),135)
        self.assertEqual(len({(r['group'],r['seed'],r['fold']) for r in rows}),135)
        for group in {r['group'] for r in rows}:
            self.assertEqual({(r['seed'],r['fold']) for r in rows if r['group']==group},{(s,f) for s in [17,42,2026] for f in range(5)})

    def test_all_reported_benchmark_tables_reproduce(self):
        target=REPO/'runs/benchmark_test_outputs';target.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target) as work,contextlib.redirect_stdout(io.StringIO()):
            result=tables.run(REPO,Path(work))
            self.assertEqual(result['status'],'PASS')
            self.assertEqual(result['benchmark_summary_checks'],1980)
            self.assertEqual(result['additional_seed_summaries'],9)
            self.assertEqual(result['additional_regional_records'],11638)
            self.assertTrue((Path(work)/'reproduced_tables/Table_B24.csv').is_file())


if __name__=='__main__':unittest.main()
