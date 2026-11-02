import unittest
import pandas as pd
from src.prepare_public_pool import prepare


class PoolPreparationTests(unittest.TestCase):
    def test_overlap_excludes_whole_group_and_small_groups_are_reported(self):
        rows = []
        texts = {
            'keep': ['alpha red violet', 'beta orange green', 'gamma blue black'],
            'overlap': ['official exact response', 'other one answer', 'other two answer'],
            'tiny': ['short yellow answer', 'short purple answer'],
        }
        for prompt, answers in texts.items():
            for model, text in zip('abc', answers):
                rows.append(dict(id=f'{prompt}-{model}', prompt_id=prompt, model=model,
                                 text=text, source='test', prompt_key=prompt))
        official = pd.DataFrame({'id': ['o1', 'o2', 'o3'], 'prompt_id': ['official'] * 3,
                                 'model': list('abc'),
                                 'text': ['official exact response', 'official second response', 'official third response']})
        pool, reasons, audit = prepare(pd.DataFrame(rows), official)
        self.assertEqual(set(pool.prompt_id), {'keep'})
        self.assertEqual(len(reasons), 5)
        self.assertEqual(audit['official_answer_matches'], 1)
        self.assertEqual(reasons['overlap-b'], 'official_answer_or_id_overlap_group')
        self.assertEqual(reasons['tiny-a'], 'fewer_than_three_authors')

    def test_no_repeated_prompt_author_removal(self):
        pool = pd.DataFrame({'id': list('abcd'), 'prompt_id': ['p'] * 4,
                             'model': list('aabc'), 'text': ['red', 'yellow', 'green', 'blue'],
                             'source': ['test'] * 4, 'prompt_key': ['p'] * 4})
        official = pd.DataFrame({'id': ['o1','o2','o3'], 'prompt_id': ['o']*3,
                                 'model': list('abc'), 'text': ['cat','dog','mouse']})
        result, _, _ = prepare(pool, official)
        self.assertEqual(len(result), 4)


if __name__ == '__main__':
    unittest.main()
