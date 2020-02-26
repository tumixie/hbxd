# coding: utf-8

"""
job_pboc_parse.py

Usage:
  job_pboc_parse.py <work_dir> <report_dir> <bom_dir> <log_dir> <run_date>
  job_pboc_parse.py -h | --helpa
  job_pboc_parse.py --version

Options:
  -h --help              Show this screen.
  --version              Show version.
"""

import sys
import json
import os
import shutil
import logging
import pathlib
import traceback

from docopt import docopt

sys.path.append(pathlib.Path(__file__).absolute().parent.parent.as_posix())

# sys.path.append('/Users/tumixie/project/ffd/ds/root/project/job/huabei_loan_pboc')
# import tojson, pboc

# from scripts import tojson, pboc


def parse_pboc(work_dir, word_file: str, out_dir: str = None, log_dir=None):
    import traceback
    p = word_file.split(os.path.sep)
    d, f = '{0}'.format(os.path.sep).join(p[:-1]), p[-1]
    if out_dir is not None:
        d = out_dir
    if pathlib.Path(d, f).as_posix() != f:
        shutil.copy(word_file, pathlib.Path(log_dir, f).as_posix())
    json_file = pathlib.Path(log_dir, '{0}.json'.format(f)).as_posix()
    bom_file = pathlib.Path(d, '{0}.bom.txt'.format(f)).as_posix()
    his_bom_all_dir = os.path.join(work_dir, 'all_var_bom_his')
    os.makedirs(his_bom_all_dir, exist_ok=True)
    all_var_bom_file = pathlib.Path(his_bom_all_dir, '{0}.bom.txt'.format(f)).as_posix()
    logger.info('to json: {0}'.format(word_file))
    tojson.to_json(word_file, json_file)
    with open(json_file, encoding='utf-8') as f:
        obj = json.load(f)
        logger.info('run pboc bom: {0}'.format(json_file))
        bom = pboc.pboc_bom(obj)
        logger.info('bom to file: {0}'.format(bom_file))
        with open(bom_file, 'w', encoding='utf-8') as of:
            all_var_bom = {}
            export_vars = ['pboc_debt_loan', 'pboc_lc_ucl_pct_lf', 'pboc_lc_uclj6_pct_lf', 'pboc_hs_coffiecient_level1'
                , 'pboc_hs_coffiecient_level2', 'pboc_hs_credit_limit_level1', 'pboc_hs_credit_limit_level2'
                , 'pboc_hs_repay_monthly_coffiecient_level1', 'pboc_hs_repay_monthly_coffiecient_level2']
            # all_var_bom['pboc_debt_loan'] = bom.get('pboc_debt_loan_004', 'C')
            for v in export_vars:
                all_var_bom[v] = bom.get(v, 'C')
            json.dump(all_var_bom, of, ensure_ascii=False)
        with open(all_var_bom_file, 'w', encoding='utf-8') as of:
            json.dump(bom, of, ensure_ascii=False)


def get_pboc_word_files(from_dir, bom_dir):
    # f = [
    #     # r'F:\rongsai\ds\root\project\dtils\tests\etl\pboc\7256_fanshaohua.docx',
    #     '/home/taiping/pboc_jobs/scripts/7431_chenwenhong.docx',
    # ]
    dones = [os.path.basename(fl).split('.')[0] for fl in os.listdir(bom_dir)]
    lst = []
    for fl in os.listdir(from_dir):
        if os.path.basename(fl).split('.')[0] in dones:
            continue
        lst.append(os.path.join(from_dir, fl))
    return lst


def run_job(work_dir, report_dir, bom_dir, log_dir):
    his_bom_all_dir = os.path.join(work_dir, 'bom_his')
    os.makedirs(his_bom_all_dir, exist_ok=True)
    if not os.path.exists(report_dir) or len(os.listdir(report_dir)) <= 0:
        logger.info('无报告')
        os.removedirs(bom_dir)
        return
    word_files = get_pboc_word_files(report_dir, his_bom_all_dir)
    # out_dir = r'F:\rongsai\ds\root\project\dtils\tests\etl\temp'
    # out_dir = '/home/taiping/pboc_jobs/log'
    # shutil.move(bom_dir, '{0}__bak'.format(bom_dir))
    # logger.info('删除上次结果')
    # if os.path.exists(bom_dir):
    #    for fl in os.listdir(bom_dir):
    #        os.remove(os.path.join(bom_dir, fl))
    #    os.rmdir(bom_dir)
    if len(word_files) == 0:
        logger.info('无新文件')
        os.removedirs(bom_dir)
        return
    os.makedirs(bom_dir, exist_ok=True)
    for fl in word_files:
        try:
            logger.info('start {0}'.format(fl))
            parse_pboc(work_dir, fl, bom_dir, log_dir)
        except Exception as e:
            logger.error(traceback.format_exc())
    for fl in os.listdir(bom_dir):
        shutil.copy(pathlib.Path(bom_dir, fl).as_posix(), his_bom_all_dir)


def log_(log_file_name=None, name=__name__, stdout_on=True):
    # logger_ = logging.getLogger(name)
    # fmt = '[%(asctime)s.%(msecs)d][%(name)s][%(levelname)s]%(msg)s'
    # date_fmt = '%Y-%m-%d %H:%M:%S'
    # logging.basicConfig(filename=log_file_name, datefmt=date_fmt, format=fmt)
    # logger_.setLevel(logging.DEBUG)
    log_file_name = log_file_name if log_file_name is not None else (__name__ + '.log')

    logger_ = logging.getLogger(name)
    fmt = '[%(asctime)s.%(msecs)d][%(name)s][process:%(process)d][%(levelname)s]%(msg)s'
    date_fmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)

    if stdout_on:
        stout_handler = logging.StreamHandler(sys.stdout)
        stout_handler.setFormatter(formatter)
        logger_.addHandler(stout_handler)

    file_handler = logging.FileHandler(log_file_name, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger_.addHandler(file_handler)

    logger_.setLevel(logging.DEBUG)

    return logger_


if __name__ == '__main__':
    # work_dir = '/Users/tumixie/project/ffd/ds/root/project/job/huabei_loan_pboc/test'
    # report_dir = '/Users/tumixie/project/ffd/ds/root/project/job/huabei_loan_pboc/test/doc'
    # bom_dir = '/Users/tumixie/project/ffd/ds/root/project/job/huabei_loan_pboc/test/bom'
    # log_dir = '/Users/tumixie/project/ffd/ds/root/project/job/huabei_loan_pboc/test/log'
    # run_date = '20200226'
    # sys.argv = [sys.argv[0], work_dir, report_dir, bom_dir, log_dir, run_date]
    args = docopt(__doc__)
    print(args)

    work_dir = args['<work_dir>']
    run_date = args['<run_date>']
    report_dir = args['<report_dir>']
    log_dir = args['<log_dir>']
    bom_dir = args['<bom_dir>']

    from datetime import datetime

    # log_dir = os.path.join(work_dir, 'log', datetime.now().strftime('%Y%m%d'))
    try:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'job_pboc_parse_{0}'.format(datetime.now().strftime('%Y%m%d%H%M%S')))
        logger = log_(log_file, name='scripts', stdout_on=True)
        logger.info('开始解析')
        logger.info('-' * 30)
        run_job(work_dir, report_dir, bom_dir, log_dir)
    except Exception as e:
        logger.error(traceback.format_exc())
        shutil.move(log_file, '{1}/ERROR_{0}'.format(os.path.basename(log_file), os.path.dirname(log_file)))
    finally:
        if os.path.exists(bom_dir):
            if len(os.listdir(bom_dir)) == 0:
                os.removedirs(bom_dir)
        with open(log_file, encoding='utf-8') as f:
            log_str = f.read()
            if 'ERROR' in log_str or 'error' in log_str:
                shutil.move(log_file, '{1}/ERROR_{0}'.format(os.path.basename(log_file), os.path.dirname(log_file)))
