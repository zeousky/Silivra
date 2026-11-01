"""Frozen native diagnostics, data exclusion and selection/retention boundaries."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from silivra.evaluation import validate_cases
from silivra.agent import parse_action
from silivra.arithmetic import calculate

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('native_experiment_measurement',ROOT/'scripts/measure-native-experiment.py')
measure=importlib.util.module_from_spec(spec);spec.loader.exec_module(measure)


class FrozenSuiteTests(unittest.TestCase):
    def test_frozen_suite_and_prompt_only_export_match(self):
        prompts=measure.verify_frozen_inputs(include_answers=True)
        cases=measure.read_jsonl(measure.FRESH);validate_cases(cases)
        self.assertEqual(len(cases),48)
        self.assertEqual(prompts,{measure.normalize(p) for case in cases for p in measure.user_messages(case)})
        self.assertEqual(len(prompts),52)
        self.assertTrue(all(isinstance(x,str) for x in measure.read_jsonl(measure.PROMPTS)))
        self.assertEqual(len({case['id'] for case in cases}),48)

    def test_final_questions_are_distinct_from_old_development_questions(self):
        fresh=measure.verify_frozen_inputs()
        development={measure.normalize(text) for path in measure.DEVELOPMENT
                     for case in measure.read_jsonl(path) for text in measure.user_messages(case)}
        self.assertFalse(fresh & development)

    def test_pilot_verification_never_opens_fresh_answers(self):
        with patch.object(measure,'FRESH',Path('/nonexistent/fresh-evaluation-must-not-be-read')):
            self.assertEqual(len(measure.verify_frozen_inputs()),52)
            with self.assertRaises(FileNotFoundError):measure.verify_frozen_inputs(include_answers=True)

    def test_tool_observations_and_inert_memory_have_valid_roles(self):
        for case in measure.read_jsonl(measure.FRESH):
            if 'messages' not in case:continue
            self.assertNotIn('system',{m['role'] for m in case['messages']})
            self.assertIn(case['messages'][-1]['role'],{'user','tool'})
            if case['category']=='tool_observation_calculate':
                action=parse_action(case['messages'][1]['content'])
                supplied=json.loads(case['messages'][2]['content'])
                actual=calculate(action['arguments']['expression'])
                self.assertEqual(supplied,{'ok':True,'result':actual})
                self.assertEqual(case['expected'],{'final':actual['result']})

    def test_development_evaluation_never_generates_on_final_questions(self):
        class Backend:
            def __init__(self):self.prompts=[]
            def generate(self,messages,tools):
                self.prompts.extend(m['content'] for m in messages if m['role']=='user')
                return '{"final":"development-only test"}'
        backend=Backend();reports=measure.evaluate_development(backend)
        self.assertEqual(sum(r['cases'] for r in reports.values()),85)
        self.assertFalse({measure.normalize(p) for p in backend.prompts} & measure.verify_frozen_inputs())


class AcceptanceTests(unittest.TestCase):
    def records(self):
        baseline={'losses':{'prior_instruction_loss':2.,'story_validation_loss':3.},
                  'development_summary':{'passed':3,'cases':85,'valid_actions':72}}
        candidate={'losses':{'prior_instruction_loss':2.05,'story_validation_loss':3.10},
                   'development_summary':{'passed':50,'cases':85,'valid_actions':85},
                   'fresh_summary':{'passed':40,'cases':48,'valid_actions':48},
                   'fresh_groups':{key:{'passed':value,'cases':value,'valid_actions':value}
                                   for key,value in measure.MINIMUM_PASSES.items()}}
        return candidate,baseline

    def test_acceptance_needs_all_domains_broad_success_and_retention(self):
        candidate,baseline=self.records()
        self.assertTrue(measure.acceptance(candidate,baseline)['eligible'])
        bad=deepcopy(candidate);bad['fresh_groups']['memory']['passed']=2
        self.assertFalse(measure.acceptance(bad,baseline)['eligible'])
        bad=deepcopy(candidate);bad['development_summary']['passed']=3
        self.assertFalse(measure.acceptance(bad,baseline)['eligible'])
        bad=deepcopy(candidate);bad['losses']['story_validation_loss']=3.16
        self.assertFalse(measure.acceptance(bad,baseline)['eligible'])
        bad=deepcopy(candidate);bad['fresh_summary']['passed']=35
        self.assertFalse(measure.acceptance(bad,baseline)['eligible'])

    def test_pilot_cannot_be_marked_promoted_without_final_evidence(self):
        candidate,baseline=self.records();candidate.pop('fresh_summary');candidate.pop('fresh_groups')
        self.assertFalse(measure.acceptance(candidate,baseline)['eligible'])

    def test_retention_five_percent_limit_and_missing_values(self):
        baseline={'prior_instruction_loss':2.,'story_validation_loss':3.}
        for value,passed in [(2.1,True),(2.10001,False),(float('nan'),False),(None,False)]:
            gate=measure.retention_gate({'prior_instruction_loss':value,'story_validation_loss':3.},baseline)
            self.assertEqual(gate['prior_instruction_loss']['passed'],passed)

    def test_final_selection_binds_development_only_checkpoint_and_data(self):
        current={'checkpoint':'step-0000100','hashes':{'model.safetensors':'abc'}}
        pilot={'phase':'pilot','role':'candidate','checkpoint':current,
               'data_manifest_sha256':'data','fresh_suite_executed':False}
        measure.validate_selection(pilot,current,'data')
        for key,value in [('phase','final'),('checkpoint',{'checkpoint':'other'}),
                          ('data_manifest_sha256','other'),('fresh_suite_executed',True)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                measure.validate_selection({**pilot,key:value},current,'data')

    def test_lineage_checks_retain_original_ancestor(self):
        self.assertTrue(measure.descends_from({'weights_sha256':'target'},'target'))
        self.assertTrue(measure.descends_from({'weights_sha256':'other','parent':{'weights_sha256':'target'}},'target'))
        self.assertFalse(measure.descends_from({'weights_sha256':'other'},'target'))


class ExclusionAndReceiptTests(unittest.TestCase):
    def test_data_audit_rejects_heldout_prompts_in_train_and_development_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            manifest={'splits':{key:{'source_path':key+'.jsonl'} for key in ('train','valid')}}
            def write(split,prompt):
                (directory/(split+'.jsonl')).write_text(json.dumps({'messages':[{'role':'user','content':prompt}]})+'\n')
            write('train','A distinct original training example.')
            write('valid','A distinct original validation example.')
            self.assertEqual(measure.audit_exclusions(directory,manifest)['splits']['train']['exact_user_message_overlaps'],0)
            heldout=measure.read_jsonl(measure.PROMPTS)[0]
            for split in ('train','valid'):
                write(split,heldout)
                with self.assertRaises(ValueError):measure.audit_exclusions(directory,manifest)
                write(split,'Another distinct '+split+' example.')

    def test_final_attempt_receipt_is_exclusive_and_retains_selection_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'receipt.json'
            record={'role':'candidate','checkpoint':{'hashes':{'weights':'value'}},'selection_report_sha256':'selection'}
            measure.claim_final_receipt(path,record)
            self.assertEqual(json.loads(path.read_text())['selection_report_sha256'],'selection')
            with self.assertRaises(FileExistsError):measure.claim_final_receipt(path,record)
            self.assertEqual(path.stat().st_mode & 0o777,0o600)

    def test_prompt_normalization_matches_nfc_and_whitespace(self):
        self.assertEqual(measure.normalize(' cafe\u0301   note\n'),measure.normalize('caf\u00e9 note'))


if __name__=='__main__':unittest.main()
