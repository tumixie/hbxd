# coding: utf-8

import re
import json
import logging
from datetime import datetime, date
from dateutil import relativedelta, parser
from typing import List, Optional
from collections import defaultdict

import pandas as pd
import numpy as np
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from scipy.linalg import norm

START_TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
TIME_WINDOW = {'j1m': 30, 'j3m': 90, 'j6m': 180, 'j12m': 360, 'j24m': 720, 'lf': 99999}
TIME_WINDOW_V2 = {'j3m': 3, 'j6m': 6, 'j12m': 12, 'j24m': 24, 'lf': 99999}

logger = logging.getLogger(__name__)


def pboc_bom(obj, version=None):
    """
    输入原始的征信报文
    :param obj:
    :param version
    :return:
    """
    features = {}
    pboc = PBOCEntity(obj, _type=1)

    features['education_level'] = transfer_education_level(pboc.basic_info.get('eduLevel', None))
    features['pboc_lc_ucl_pct_lf'] = cal_used_credit_limit_percent(pboc.raw_data)
    features['pboc_lc_uclj6_pct_lf'] = cal_used_credit_limit_percent_j6m(pboc.raw_data)
    features.update(hbxd_house_loan_feature(pboc))
    features.update(summary_bom(pboc))
    features.update(query_info_bom(pboc.query_info))
    features.update(loan_info_bom(pboc.loan_detail))
    features.update(loan_card_bom(pboc.credit_card_detail))
    features.update(standard_loan_card_bom(pboc.standard_credit_card_detail))
    features.update(rule_direct_variables(pboc, obj))
    features.update(debt_variables(pboc))
    features['credit_limit'] = calculate_credit_limit(features)
    features = clean(features)
    features = mapping(features)

    # features = filter_feature(features)

    return features


def calculate_credit_limit(features):
    """计算可贷额度"""
    return (0.75 * 1000000 - features['pboc_debt_loan_004']) / 0.00424


def mapping(features):
    """
    字段映射
    CUMBERLAND代表数字0-9，输出结果以百分比为单位，结果取整，若数字是0或空，则输出为0(即C）。例：数值为3%输出为M，数值为46%输出为BR，输出结果以字母表示
    金额<=10000以内输出结果以百元为单位，100元记为1，金额在(10000，100000)元的按千元为单位，金额在100000以上的按万元为单位，金额为0的输出为0， 输出结果取整
    CUMBERLAND代表数字0-9，结果取整，若数字是0或空，则输出为0(即C）。例：笔数为3输出为M，笔数为46输出为BR，输出结果以字母表示
    :param features:
    :return:
    """
    n = {}
    for k, v in features.items():
        if isinstance(v, str):
            n[k] = v
        elif isinstance(v, (float, int)):
            if 'pct' in k:
                if np.isinf(v):
                    v = 9999999
                n[k] = number_to_string(int(v * 100))
            elif 'cl' in k or 'amt' in k or 'amount' in k or 'balance' in k or k in ['pboc_debt_ln_bank_nperiod',
                                                                                     'pboc_debt_ln_bank_period',
                                                                                     'pboc_debt_ln_nbank_nperiod',
                                                                                     'pboc_debt_ln_nbank_period',
                                                                                     'pboc_debt_loan_004']:
                if features[k] <= 10000:
                    v = round(features[k], -2)
                elif features[k] <= 100000:
                    v = round(features[k], -3)
                else:
                    v = round(features[k], -4)
                n[k] = number_to_string(int(v))
            else:
                n[k] = number_to_string(int(features[k]))

    return n


def number_to_string(number):
    """
    :param number:
    :return:
    """
    out = 'CUMBERLAND'
    rs = ''
    for v in str(number):
        rs += out[int(v)]
    return rs


def transfer_education_level(education_level):
    """
    高中
    大学专科和专科学校（简称"大专"）
    初中
    大学本科（简称"大学"）
    小学
    中等专业学校或中等技术学校
    研究生
    --
    :param education_level:
    :return:
    """
    if education_level is None or education_level == '' or '--' in education_level:
        return 0
    elif '小学' in education_level:
        return 1
    elif '初中' in education_level:
        return 2
    elif '中等专业学校或中等技术学校' in education_level:
        return 3
    elif '高中' in education_level:
        return 4
    elif '大学本科' in education_level:
        return 5
    elif '大学专科' in education_level:
        return 6
    elif '研究生' in education_level:
        return 7

    return 0


def loan_debt_cal_logic(x, type_=1):
    # 贷款本金余额为0，则默认结清，不计算负债
    # 贷款“转出”的，不计负债  # TODO
    # 贷款“已结清”的，不计负债
    if x['settle_type'] == 'stl' or x['balance'] == 0:
        return 0
    if type_ == 1:
        ########################
        # 明确显示剩余还款期数、明确显示本月应还款额
        # 明确显示剩余还款期数、未显示本月应还款额（显示为0、-等）
        # 未明确显示剩余还款期数，明确显示本月应还款额的（例如：不定期归还、双周供、气球贷等）
        # 未明确显示剩余还款期数，未明确显本月应还款额的
        ########################
        if not pd.isnull(x['remainPaymentCyc']) and x['scheduledPaymentAmount'] != 0:
            """
            本金余额*2＞本月应还款额*剩余还款期数＞本金余额，则按显示的本月应还款额计算
            本月应还款额*剩余还款期数≤本金余额，则按贷款发放金额，按实际贷款期数摊算
            本月应还款额*剩余贷款期数≥本金余额*2，则按贷款发放金额，按实际贷款期数摊算
            """
            if x['balance'] * 2 > x['scheduledPaymentAmount'] * x['remainPaymentCyc'] > x['balance']:
                return x['scheduledPaymentAmount']
            else:
                return x['credit_limit'] / x['loan_terms']
        elif not pd.isnull(x['remainPaymentCyc']) and x['scheduledPaymentAmount'] == 0:
            """
            按贷款发放金额，结合实际贷款期数摊算
            """
            if not pd.isnull(x['loan_terms']):
                return x['credit_limit'] / x['loan_terms']
            else:
                return x['credit_limit'] / int((get_time(x['end_date']) - get_time(x['openDate'])).days / 30)
        elif pd.isnull(x['remainPaymentCyc']) and x['scheduledPaymentAmount'] != 0:
            """
            贷款发放金额*2＞本月应还款额*贷款期数＞贷款发放金额，则按显示的本月应还款额计算
            本月应还款额*贷款期数≤贷款发放金额，则按贷款发放金额，结合实际贷款期数摊算
            本月应还款额*贷款期数≥贷款发放金额*2，则按贷款发放金额，结合实际贷款期数摊算
            """
            if x['credit_limit'] * 2 > x['scheduledPaymentAmount'] * x['loan_terms'] > x['credit_limit']:
                return x['scheduledPaymentAmount']
            else:
                return x['credit_limit'] / x['loan_terms']
        else:
            """
            按贷款发放金额，结合贷款期限摊算，或贷款发放金额*3.5%折算
            """
            if not pd.isnull(x['loan_terms']):
                return x['credit_limit'] / x['loan_terms']
            elif not pd.isnull(x['end_date']):
                return x['credit_limit'] / int((get_time(x['end_date']) - get_time(x['openDate'])).days / 30)
            return x['credit_limit'] * 0.035
    elif type_ == 2:
        """简化版本"""
        if x['loan_type'] == '抵押担保':
            if x['scheduledPaymentAmount'] != 0:
                return x['scheduledPaymentAmount']
            else:
                return x['credit_limit'] / x['loan_terms']
        else:
            if not pd.isnull(x['remainPaymentCyc']) and x['scheduledPaymentAmount'] != 0:
                if x['scheduledPaymentAmount'] * x['remainPaymentCyc'] > x['balance']:
                    return x['scheduledPaymentAmount']
                else:
                    return x['credit_limit'] * 0.035
    elif type_ == 3:
        """
        期供类贷款：条件1或者条件2任意一条满足即可
        条件1.明确显示剩余还款期数、明确显示本月应还款额的，
        且本金余额*2＞本月应还款额*剩余还款期数＞本金余额
        条件2.未明确显示剩余还款期数，明确显示本月应还款额的，
        且贷款发放金额*2＞本月应还款额*贷款期数＞贷款发放金额
        """
        if not pd.isnull(x['remainPaymentCyc']) and x['scheduledPaymentAmount'] != 0 and x['balance'] * 2 > x[
            'scheduledPaymentAmount'] * x['remainPaymentCyc'] > x['balance']:
            return x['scheduledPaymentAmount']
        elif pd.isnull(x['remainPaymentCyc']) and x['scheduledPaymentAmount'] != 0 and x['credit_limit'] * 2 > x[
            'scheduledPaymentAmount'] * x['loan_terms'] > x['credit_limit']:
            return x['scheduledPaymentAmount']
        else:
            if '银行' in x['loan_from']:
                if pd.isnull(x['loan_terms']):
                    return x['credit_limit'] * 0.035
                if x['loan_terms'] <= 12:
                    return x['credit_limit'] * 0.09
                elif x['loan_terms'] <= 35:
                    return x['credit_limit'] * 0.049
                else:
                    return x['credit_limit'] * 0.035
            else:
                if pd.isnull(x['loan_terms']):
                    return x['credit_limit'] * 0.04
                if x['loan_terms'] <= 12:
                    return x['credit_limit'] * 0.095
                elif x['loan_terms'] <= 35:
                    return x['credit_limit'] * 0.053
                else:
                    return x['credit_limit'] * 0.04


def detail_processing(x):
    if x['settle_type'] == 'stl' or x['balance'] == 0:
        return 0, 0, 0
    if not pd.isnull(x['remainPaymentCyc']) and x['scheduledPaymentAmount'] != 0 and x['balance'] * 2 > x[
        'scheduledPaymentAmount'] * x['remainPaymentCyc'] > x['balance']:
        return 0, x['scheduledPaymentAmount'], 0
    elif pd.isnull(x['remainPaymentCyc']) and x['scheduledPaymentAmount'] != 0 and x['credit_limit'] * 2 > x[
        'scheduledPaymentAmount'] * x['loan_terms'] > x['credit_limit']:
        return 0, x['scheduledPaymentAmount'], 0
    else:
        if '银行' in x['loan_from']:
            if pd.isnull(x['loan_terms']):
                return 0, x['credit_limit'] * 0.035, 0.035
            if x['loan_terms'] <= 12:
                return 1, x['credit_limit'] * 0.09, 0.09
            elif x['loan_terms'] <= 35:
                return 1, x['credit_limit'] * 0.049, 0.049
            else:
                return 1, x['credit_limit'] * 0.035, 0.035
        else:
            if pd.isnull(x['loan_terms']):
                return 0, x['credit_limit'] * 0.04, 0.04
            if x['loan_terms'] <= 12:
                return 2, x['credit_limit'] * 0.095, 0.095
            elif x['loan_terms'] <= 35:
                return 2, x['credit_limit'] * 0.053, 0.053
            else:
                return 2, x['credit_limit'] * 0.04, 0.04


def credit_card_debt_cal_logic(x):
    """
    贷记卡 准贷记卡
    未销户贷记卡信息汇总中的已用额度,信用卡：已用额度 * 10 %
    未销户准贷记卡信息汇总中的透支余额,准贷记卡：透支余额 * 10 %
    """
    if x['accountState'] == '正常':
        return x['used_credit_limit'] * 0.1
    else:
        return 0


def pboc_debt_loan(loan_df, loan_card, standard_loan_card, type_=1):
    debt_sum = 0
    simple_debt_sum = 0
    features = dict()

    ldf = loan_df.copy()
    if len(ldf) != 0:
        ldf['debt_sum'] = ldf.apply(lambda x: loan_debt_cal_logic(x, type_=type_), axis=1)
        ldf['debt_cls'] = ldf.apply(lambda x: loan_debt_cls(x), axis=1)
        ldf['simple_debt_sum'] = ldf.apply(lambda x: loan_debt_cal_logic(x, type_=2), axis=1)
        debt_sum = np.sum(ldf['debt_sum'])
        simple_debt_sum = np.sum(ldf['simple_debt_sum'])
        # 负债分类
        rs = ldf.groupby(['debt_cls'])['debt_sum'].sum()
        for ix in rs.index:
            features['pboc_debt_ln_{0}'.format(ix)] = rs[ix]
    if len(loan_card) != 0:
        loan_card = loan_card.drop_duplicates(['account']).copy()
        rs9 = np.sum(loan_card.apply(credit_card_debt_cal_logic, axis=1))
        features['pboc_debt_loan_card'] = rs9
        debt_sum += rs9
        simple_debt_sum += rs9
    if len(standard_loan_card) != 0:
        standard_loan_card = standard_loan_card.drop_duplicates(['account']).copy()
        rs10 = np.sum(standard_loan_card.apply(credit_card_debt_cal_logic, axis=1))
        features['pboc_debt_standard_loan_card'] = rs10
        debt_sum += rs10
        simple_debt_sum += rs10

    return round(debt_sum, 2), round(simple_debt_sum, 2), ldf[['debt_sum', 'account']] if len(ldf) > 0 else '', features


def loan_debt_cls(x: pd.Series):
    if not pd.isnull(x['remainPaymentCyc']) and x['scheduledPaymentAmount'] != 0 and x['balance'] * 2 > x[
        'scheduledPaymentAmount'] * x['remainPaymentCyc'] > x['balance']:
        return 'bank_period' if '银行' in x['loan_from'] else 'nbank_period'
    elif pd.isnull(x['remainPaymentCyc']) and x['scheduledPaymentAmount'] != 0 and x['credit_limit'] * 2 > x[
        'scheduledPaymentAmount'] * x['loan_terms'] > x['credit_limit']:
        return 'bank_period' if '银行' in x['loan_from'] else 'nbank_period'
    else:
        if '银行' in x['loan_from']:
            return 'bank_nperiod'
        else:
            return 'nbank_nperiod'


def detail_process_debt(loan_df):
    if len(loan_df) == 0:
        return ''
    df = loan_df.apply(lambda x: pd.Series(detail_processing(x)), axis=1)
    se = df.groupby([0, 2])[1].sum()
    return se.to_json()


def debt_variables(pboc):
    """负债计算变量"""
    features = dict()
    loan_df = pboc.loan_detail.copy()
    loan_card = pboc.credit_card_detail.copy()
    standard_loan_card = pboc.standard_credit_card_detail.copy()
    features['pboc_debt_loan_001'], features['pboc_debt_loan_002'], _, _ = pboc_debt_loan(loan_df, loan_card,
                                                                                          standard_loan_card)
    loan_df = pboc.loan_detail.copy()
    loan_df = loan_df.drop_duplicates(['account'])
    # loan_df = loan_df[loan_df.apply(debt_type_check, axis=1)]
    ld1 = loan_df[loan_df.apply(debt_type_check, axis=1)]
    ld2 = loan_df[loan_df.apply(debt_type_check_v1, axis=1)]
    features['pboc_debt_loan_003'] = pboc_debt_loan(ld1, loan_card, standard_loan_card, type_=1)[0]
    features['pboc_debt_loan_004'], _, ldf, loan_features = pboc_debt_loan(ld2, loan_card, standard_loan_card, type_=3)
    # features['pboc_debt_loan_004_dt'] = detail_process_debt(ld2) + '||' + str(ldf).replace('\n', '|')
    features.update(loan_features)
    return features


def debt_type_check(x):
    """
    判断是否需要计算负债
    业务品种	个人住房贷款\（包括住房公积金贷款）	个人商用房（包括商住两用）贷款	个人经营性贷款	个人消费贷款	其他	个人汽车贷款
    担保方式                          
    质押（含保证金）	不计算	不计算	不计算	不计算	不计算	不计算
    抵押	不计算	不计算	不计算	不计算	不计算	不计算
    自然人保证	计算负债比	计算负债比	计算负债比	计算负债比	计算负债比	计算负债比
    信用免担保	计算负债比	计算负债比	计算负债比	计算负债比	计算负债比	计算负债比
    组合（含保证）	不计算	不计算	计算负债比	计算负债比	计算负债比	不计算
    组合（不含保证）	不计算	不计算	不计算	不计算	计算负债比	不计算
    农户联保	计算负债比	计算负债比	计算负债比	计算负债比	计算负债比	计算负债比
    其他	不计算	不计算	计算负债比	计算负债比	计算负债比	计算负债比



    发放单位	担保方式	是否计入负债
    小额信贷公司	非抵押、非质押	计算负债比
    汽车金融公司
    消费金融有限公司
    财务公司
    信托投资公司
    金融租赁公司
    其他非银行、住房公积金中心
    :return:
    """
    if '银行' not in x['loan_from'] and '住房公积金中心' not in x['loan_from']:
        if '抵押' not in x['loan_type'] and '质押' not in x['loan_type']:
            return True
    if '抵押' in x['loan_type'] or '质押' in x['loan_type']:
        return False
    elif '自然人保证' in x['loan_type'] or '保证' == x['loan_type'] or '信用免担保' in x['loan_type'] or '信用' == x[
        'loan_type'] or '免担保' == x['loan_type']:
        return True
    elif '组合含保证' in x['loan_type']:
        if '个人住房贷款' in x['type'] or '住房公积金贷款' in x['type'] or '个人商用房' in x['type']:
            return False
        elif '个人汽车贷款' in x['type']:
            return False
        else:
            return True
    elif '组合不含保证' in x['loan_type']:
        for lt in ['个人住房贷款', '住房公积金贷款', '个人商用房', '个人经营性贷款', '个人消费贷款', '个人汽车贷款']:
            if lt in x['loan_type']:
                return True
        return False
    elif '农户联保' in x['loan_type']:
        return True
    else:
        if '个人住房贷款' in x['type'] or '住房公积金贷款' in x['type'] or '个人商用房' in x['type']:
            return False
        else:
            return True


def debt_type_check_v1(x):
    """

    :return:
    """
    if '银行' in x['loan_from']:  # 银行
        if '抵押' in x['loan_type'] or '质押' in x['loan_type']:
            return False
        elif '自然人保证' in x['loan_type'] or '信用免担保' in x['loan_type']:
            return True
        elif '组合含保证' in x['loan_type']:
            if '个人住房贷款' in x['type'] or '住房公积金贷款' in x['type'] or '个人商用房' in x['type']:
                return False
            elif '个人汽车贷款' in x['type']:
                return False
            else:
                return True
        elif '组合不含保证' in x['loan_type']:
            for lt in ['个人住房贷款', '住房公积金贷款', '个人商用房', '个人经营性贷款', '个人消费贷款', '个人汽车贷款']:
                if lt in x['type']:
                    return False
            return True
        elif '农户联保' in x['loan_type']:
            return True
        else:
            if '个人住房贷款' in x['type'] or '住房公积金贷款' in x['type'] or '个人商用房' in x['type']:
                return False
            else:
                return True
    else:  # 非银行
        if '抵押' in x['loan_type'] or '质押' in x['loan_type']:
            return False
        else:
            return True


def hbxd_house_loan_type_check(x, level=1):
    """
    华北小贷房贷认定,暂时区分严格和宽松
    :param x:
    :param level:
    :return:
    """
    if level == 1:
        if '银行' not in x['loan_from'] and '公积金中心' not in x['loan_from']:
            return False
        if '抵押' in x['loan_type'] or '质押' in x['loan_type']:
            for lt in ['个人住房贷款', '住房公积金贷款', '个人商用房', '个人经营性贷款', '个人消费贷款']:
                if lt in x['type']:
                    return True
            if x['loan_terms'] is not None and x['loan_terms'] >= 96:
                return True
            else:
                return False
        elif '组合含保证' in x['loan_type']:
            if '个人住房贷款' in x['type'] or '住房公积金贷款' in x['type'] or '个人商用房' in x['type']:
                return True
            else:
                return False
        elif '组合不含保证' in x['loan_type']:
            for lt in ['个人住房贷款', '住房公积金贷款', '个人商用房', '个人经营性贷款', '个人消费贷款', '个人汽车贷款']:
                if lt in x['type']:
                    return True
            return False
        elif '保证' in x['loan_type']:
            if '住房公积金贷款' in x['type']:
                return True
            else:
                return False
        else:
            if '个人住房贷款' in x['type'] or '住房公积金贷款' in x['type'] or '个人商用房' in x['type']:
                return True
            else:
                return False
    elif level == 2:
        if '银行' not in x['loan_from'] and '公积金中心' not in x['loan_from']:
            return False
        if '保证' == x['loan_type']:
            if '住房公积金贷款' in x['type']:
                return True
            else:
                return False
        else:
            if '个人住房贷款' in x['type'] or '住房公积金贷款' in x['type'] or '个人商用房' in x['type']:
                return True
            else:
                return False


def hbxd_house_loan_admission(x):
    """

    :param x:
    :return:
    """
    if x['settle_type'] == 'stl':
        return False
    elif x['loan_terms'] <= 24:
        return False
    elif not _is_repay_monthly(x):
        return False
    return True


def _is_repay_monthly(x):
    """

    :param x:
    :return:
    """
    if not pd.isnull(x['remainPaymentCyc']) and x['scheduledPaymentAmount'] != 0 and x['balance'] * 2 > x[
        'scheduledPaymentAmount'] * x['remainPaymentCyc'] > x['balance']:
        return True
    elif pd.isnull(x['remainPaymentCyc']) and x['scheduledPaymentAmount'] != 0 and x['credit_limit'] * 2 > x[
        'scheduledPaymentAmount'] * x['loan_terms'] > x['credit_limit']:
        return True
    else:
        return False


def rule_direct_variables(pboc, obj):
    features = dict()
    lc = pboc.credit_card_detail
    slc = pboc.standard_credit_card_detail
    loan = pboc.loan_detail
    features['pboc_negative_blank_001'] = pboc_negative_blank_001(loan, lc, slc)
    features['pboc_negative_blank_002'] = pboc_negative_blank_002(lc)
    features['pboc_negative_blank_003'] = pboc_negative_blank_003(lc)
    # 近24个月有对外担保信息，但无其他当前正常使用的信贷记录
    features['pboc_negative_blank_004'] = 0  # TODO
    features['pboc_negative_blank_005'] = pboc_negative_blank_005(lc, slc)
    features['pboc_negative_blank_006'] = pboc_negative_blank_006(lc)
    #
    features['pboc_negative_black_001'] = pboc_negative_black_001(obj)
    features['pboc_negative_black_002'] = 0  # TODO
    features['pboc_negative_black_003'] = 0  # TODO

    return features


def pboc_negative_blank_001(loan, lc, slc):
    """
    近24个月无信贷记录（空白报告）
    :param loan:
    :param lc:
    :param slc:
    :return:
    """
    if len(loan) == 0 and len(lc) == 0 and len(slc) == 0:
        return 1
    elif len(loan) != 0:
        if len(loan[loan['up_to_days'] <= 365 * 2]) != 0:
            return 0
        else:
            return 1
    elif len(lc) != 0:
        if len(lc[lc['up_to_days'] <= 365 * 2]) != 0:
            return 0
        else:
            return 1
    elif len(slc) != 0:
        if len(slc[slc['up_to_days'] <= 365 * 2]) != 0:
            return 0
        else:
            return 1
    else:
        return 1


def pboc_negative_blank_002(lc):
    """
    近24个月仅有未激活的信用卡
    :param lc:
    :return:
    """

    def ck(x):
        if x['accountState'] == '未激活' and not pd.isnull(x['up_to_days']) and x['up_to_days'] <= 365 * 2:
            return 1
        elif not pd.isnull(x['up_to_days']) and x['up_to_days'] <= 365 * 2:
            return 2
        else:
            return 0

    tag = lc.apply(ck, axis=1)
    if len(tag[tag == 1]) == 0:
        return 0
    elif len(tag[tag == 2]) == 0:
        return 1
    else:
        return 0


def pboc_negative_blank_003(lc):
    """
    近24个月仅有已销户且无正常还款记录的信用卡
    :param lc:
    :return:
    """

    # 无正常还款记录的判断条件无法确认 TODO
    def ck(x):
        if x['accountState'] == '销户' and not pd.isnull(x['up_to_days']) and x['up_to_days'] <= 365 * 2:
            return 1
        elif not pd.isnull(x['up_to_days']) and x['up_to_days'] <= 365 * 2:
            return 2
        else:
            return 0

    tag = lc.apply(ck, axis=1)
    if len(tag[tag == 1]) == 0:
        return 0
    elif len(tag[tag == 2]) == 0:
        return 1
    else:
        return 0


def pboc_negative_blank_005(lc, slc):
    """
    贷记卡、准贷记卡近12个月无使用记录且授信额度为0
    :param lc:
    :param slc:
    :return:
    """

    def ck(x):
        if not pd.isnull(x['up_to_days']) and x['up_to_days'] <= 365 and x['credit_limit'] == 0:
            return True
        elif x['credit_limit'] == 0:
            return True
        else:
            return False

    tag = lc.apply(ck, axis=1)
    if len(lc) == 0 and len(slc) == 0:
        return 0
    elif len(lc) != 0 and len(lc[-tag]) != 0:
        return 0
    elif len(slc) != 0 and len(slc[-tag]) != 0:
        return 0
    else:
        return 1


def pboc_negative_blank_006(lc):
    """
    近24个月仅有已激活但未使用的信用卡
    :param lc:
    :return:
    """

    def ck(x):
        if pd.isnull(x['up_to_days']):
            return 0
        elif x['up_to_days'] <= 365 * 2 and x['accountState'] not in ['销户', '未激活'] and x['usedHighestAmount'] <= 0:
            return 1
        elif x['up_to_days'] <= 365 * 2:
            return 2
        else:
            return 0

    tag = lc.apply(ck, axis=1)
    if len(lc) == 0:
        return 0
    elif len(lc[tag == 1]) == 0:
        return 0
    elif len(lc[tag == 2]) == 0:
        return 1
    else:
        return 0


def pboc_negative_black_001(obj):
    # 呆账、核销、（冻结、止付)、担保人代偿/保证人代偿、以资抵债
    obj_str = obj['body_str']
    for kw in ['呆账', '核销', '冻结', '止付', '担保人代偿', '保证人代偿', '以资抵债']:
        if kw in obj_str:
            return 1


def pboc_negative_loan_001(loan_info):
    def ck(x):
        # 24个月内出现4
        if x['due_last_months'] > 3 and x['months'] <= 24:
            return True
        return False

    return 1 if len(loan_info[loan_info.apply(ck, axis=1)]) > 0 else 0


def pboc_negative_loan_002(loan_info):
    def ck(x):
        # 12个月内出现3
        if x['due_last_months'] > 2 and x['months'] <= 12:
            return True
        return False

    return 1 if len(loan_info[loan_info.apply(ck, axis=1)]) > 0 else 0


def pboc_negative_loan_003(loan_info):
    def ck(x):
        # 6个月内出现2
        if x['due_last_months'] > 1 and x['months'] <= 6:
            return True
        return False

    return 1 if len(loan_info[loan_info.apply(ck, axis=1)]) > 0 else 0


def pboc_negative_loan_004(loan_info):
    def ck(x):
        # 当前逾期，即最近一期出现1、2、3……
        if x['currOverdueAmount'] > 0:
            return True
        return False

    return 1 if len(loan_info[loan_info.apply(ck, axis=1)]) > 0 else 0


def pboc_negative_loan_005(loan_info):
    def ck(x):
        # 贷款五级分类：次级、可疑、损失
        if x['class5State'] in ('次级', '可疑', '损失'):
            return True
        return False

    return 1 if len(loan_info[loan_info.apply(ck, axis=1)]) > 0 else 0


def pboc_negative_loan_006(loan_info):
    def ck(x):
        # 已到期未结清（“结清”文字优先，未结清看逻辑：到期时间早于报告时间，且本金余额>0为未结清）
        if x['end_days'] is not None and x['end_days'] > 0 and x['settle_type'] == 'ustl':
            return True
        return False

    return 1 if len(loan_info[loan_info.apply(ck, axis=1)]) > 0 else 0


def pboc_negative_loan_007(loan_info):
    def ck1(x):
        # 近24个月内贷款，出现连续3个月逾期，或者累计出现6次逾期记录
        if x['due_last_months'] >= 3 and x['months'] <= 24:
            return True
        return False

    def ck2(x):
        # 近24个月内贷款，出现连续3个月逾期，或者累计出现6次逾期记录
        if x['due_last_months'] >= 1 and x['months'] <= 24:
            return True
        return False

    if len(loan_info[loan_info.apply(ck1, axis=1)]) > 0 or len(loan_info[loan_info.apply(ck2, axis=1)]) >= 6:
        return 1
    else:
        return 0


def pboc_negative_lc_001(lc):
    def ck(x):
        # “最大使用额度”＞1000，且最近24个月内有1次（含）以上连续逾期91天记录；（出现4）
        if x['accountState'] == '正常' and x['usedHighestAmount'] > 1000 and x['due_last_months'] > 3 and x[
            'months'] <= 24:
            return True
        return False

    return 1 if len(lc[lc.apply(ck, axis=1)]) > 0 else 0


def pboc_negative_lc_002(lc):
    def ck(x):
        # “最大使用额度”＞1000，且最近6个月内有1次（含）以上逾期达61天记录；（出现3）
        if x['accountState'] == '正常' and x['usedHighestAmount'] > 1000 and x['due_last_months'] > 2 and x[
            'months'] <= 6:
            return True
        return False

    return 1 if len(lc[lc.apply(ck, axis=1)]) > 0 else 0


def pboc_negative_lc_003(lc):
    def ck(x):
        # “最大使用额度”＞1000，且最近3个月内有1次逾期达31天记录；（出现2）
        if x['accountState'] == '正常' and x['usedHighestAmount'] > 1000 and x['due_last_months'] > 1 and x[
            'months'] <= 3:
            return True
        return False

    return 1 if len(lc[lc.apply(ck, axis=1)]) > 0 else 0


def pboc_negative_lc_004(lc):
    def ck(x):
        # “最大使用额度”＞1000，且最近3个月内有1次逾期达31天记录；（出现2）
        if x['accountState'] == '正常' and x['currOverdueAmount'] > 1000:
            return True
        return False

    return 1 if len(lc[lc.apply(ck, axis=1)]) > 0 else 0


def pboc_negative_lc_005(lc):
    def ck1(x):
        # 近24个月内贷记卡，出现连续3个月逾期，或者累计出现6次逾期记录
        if x['accountState'] == '正常' and x['due_last_months'] >= 3 and x['months'] <= 24:
            return True
        return False

    def ck2(x):
        # 近24个月内贷记卡，出现连续3个月逾期，或者累计出现6次逾期记录
        if x['accountState'] == '正常' and x['due_last_months'] >= 1 and x['months'] <= 24:
            return True
        return False

    if len(lc[lc.apply(ck1, axis=1)]) > 0 or len(lc[lc.apply(ck2, axis=1)]) >= 6:
        return 1
    else:
        return 0


def pboc_negative_slc_001(lc):
    def ck(x):
        # “最大透支余额”＞1000，且最近24个月内有1次（含）以上连续逾期达91天记录；
        if x['accountState'] == '正常' and x['usedHighestAmount'] > 1000 and x['due_last_months'] > 3 and x[
            'months'] <= 24:
            return True
        return False

    return 1 if len(lc[lc.apply(ck, axis=1)]) > 0 else 0


def pboc_negative_slc_002(lc):
    def ck(x):
        # “最大透支余额”＞1000，且最近12个月内有1次（含）以上逾期达61天记录；
        if x['accountState'] == '正常' and x['usedHighestAmount'] > 1000 and x['due_last_months'] > 2 and x[
            'months'] <= 12:
            return True
        return False

    return 1 if len(lc[lc.apply(ck, axis=1)]) > 0 else 0


def pboc_negative_slc_003(lc):
    def ck(x):
        # 当期有“透支余额”＞1000，且还款记录出现“3”及以上状态的准贷记卡
        if x['accountState'] == '正常' and x['used_credit_limit'] > 1000 and x['due_last_months'] > 2:
            return True
        return False

    return 1 if len(lc[lc.apply(ck, axis=1)]) > 0 else 0


def pboc_negative_slc_004(lc):
    def ck1(x):
        # 近24个月内准贷记卡，出现连续3个月逾期，或者累计出现6次逾期记录
        if x['accountState'] == '正常' and x['due_last_months'] >= 3 and x['months'] <= 24:
            return True
        return False

    def ck2(x):
        # 近24个月内准贷记卡，出现连续3个月逾期，或者累计出现6次逾期记录
        if x['accountState'] == '正常' and x['due_last_months'] >= 1 and x['months'] <= 24:
            return True
        return False

    if len(lc[lc.apply(ck1, axis=1)]) > 0 or len(lc[lc.apply(ck2, axis=1)]) >= 6:
        return 1
    else:
        return 0


def cal_used_credit_limit_percent(obj):
    """

    :param obj:
    :return:
    """
    lc = get_value('summary_info,shareAndDebt,unDestroyLoanCard', obj, {})
    credit_limit = transfer_amount(lc.get('creditLimit', '0'))
    used_credit_limit = transfer_amount(lc.get('usedCreditLimit', '0'))
    return used_credit_limit / credit_limit if credit_limit > 0 else 0


def cal_used_credit_limit_percent_j6m(obj):
    """

    :param obj:
    :return:
    """
    lc = get_value('summary_info,shareAndDebt,unDestroyLoanCard', obj, {})
    credit_limit = transfer_amount(lc.get('creditLimit', '0'))
    used_credit_limit = transfer_amount(lc.get('latest6MonthUsedAvgAmount', '0'))
    return used_credit_limit / credit_limit if credit_limit > 0 else 0


def filter_feature(features):
    """
    学历 education_level
    近1年信用卡逾期笔数 pboc_lc_due_rcrd_sum_j12m
    近2年信用卡最大逾期情况 pboc_lc_due_months_max_j24m
    最早信用记录距今天数 pboc_lc_dsst_max_lf
    信用卡额度使用率 pboc_lc_credit_limit_pct_lf
    信用卡最大额度 pboc_lc_credit_limit_max_lf
    信用卡平均额度 pboc_lc_credit_limit_avg_lf
    信用卡当前逾期总金额 pboc_lc_due_amount_sum_lf
    信用卡张数 pboc_lc_rcrd_sum_lf
    贷款笔数 pboc_ln_tot_tot_rcrd_sum_lf
    近2年贷款最大逾期情况
    房贷记录数 pboc_ln_tot_hs_rcrd_sum_lf
    未结清贷款最早账户距离申请的天数 pboc_ln_ustl_tot_dsst_max_lf
    首笔贷款发放距今月数 pboc_ln_tot_tot_msst_max_lf
    近1年贷款逾期笔数
    贷款当期逾期最大期数 pboc_ln_curr_due_cyc_max_lf
    近90天审批查询次数 pboc_qr_xs_tot_rcrd_cnt_j3m
    近90天总查询次数 pboc_qr_tot_tot_rcrd_cnt_j3m
    近90天最早一次信用卡查询距今天数 pboc_qr_tot_cd_dsst_max_j3m
    近180天审批查询次数 pboc_qr_xs_tot_rcrd_cnt_j6m
    :param features:
    :return:
    """
    keep_features = [
        'education_level',
        'pboc_lc_due_rcrd_sum_j12m',
        'pboc_lc_due_months_max_j24m',
        'pboc_lc_dsst_max_lf',
        'pboc_lc_credit_limit_pct_lf',
        'pboc_lc_credit_limit_max_lf',
        'pboc_lc_credit_limit_avg_lf',
        'pboc_lc_due_amount_sum_lf',
        'pboc_lc_rcrd_sum_lf',
        'pboc_ln_tot_tot_rcrd_sum_lf',
        'pboc_ln_tot_hs_rcrd_sum_lf',
        'pboc_ln_ustl_tot_dsst_max_lf',
        'pboc_ln_tot_tot_msst_max_lf',
        'pboc_ln_curr_due_cyc_max_lf',
        'pboc_qr_xs_tot_rcrd_cnt_j3m',
        'pboc_qr_tot_tot_rcrd_cnt_j3m',
        'pboc_qr_tot_cd_dsst_max_j3m',
        'pboc_qr_xs_tot_rcrd_cnt_j6m',
    ]
    rs_features = {}
    for k, v in features.items():
        if k in keep_features:
            rs_features[k] = v
    return rs_features


def query_info_bom(query_info):
    """
    查询
    1.
        qr 查询分析
    3.
        reason与operator合并
        reason:
            xs 信用卡审批
            dk 贷款审批
            dg 贷后管理
            bcn 本人查询（互联网个人信用信息服务平台）
            bc 本人查询（商业银行网上银行）
            ot 其他查询
    4.
        cnt,min,max,sum

    """
    reason = {'xs': u'信用卡审批', 'dg': u'贷后管理', 'bcn': u'本人查询（互联网个人信用信息服务平台）',
              'ot': u'其他查询', 'bc': u'本人查询（商业银行网上银行）', 'tot': None, 'dk': u'贷款审批', 'dq': '贷前审批'}

    # operator = {'xj': u'消费金融', 'ln': u'小额贷款', 'cd': u'信用卡', 'sb': u'国有银行',
    #             'tb': u'其他银行', 'nf': u'非银机构', 'ot': u'其他', 'own': u'本人', 'tot': None}

    source = ['pboc_qr']
    feature = dict()
    dt = query_info

    for s1 in source:
        for tw in TIME_WINDOW:
            df1 = dt[dt['days'] <= TIME_WINDOW[tw]].copy()
            for s2 in reason:
                if reason[s2] is not None:
                    df2 = df1[df1['query_reason'].apply(lambda x: s2 in x.split(','))]
                else:
                    df2 = df1
                if len(df2) == 0:
                    continue
                s1234 = s1 + '_' + s2
                feature[s1234 + '_dsst_max_' + tw] = df2['days'].max()  # 最早一次查询
                feature[s1234 + '_dsst_min_' + tw] = df2['days'].min()  # 最晚一次查询
                feature[s1234 + '_days_sum_' + tw] = len(df2['days'].drop_duplicates())  # 有查询的天数
                feature[s1234 + '_rcrd_cnt_' + tw] = len(df2['query_reason'])  # 查询次数
                feature[s1234 + '_org_nno_' + tw] = len(set(df2['querier']))  # 机构数

    def condition001(x):
        # 在不含本笔的情况下，借款人的人行报告显示近2个月内分别有5次（含）以上信用查询记录，且查询原因是“贷款审批”或“信用卡审批”的，不予接受；但确认为同一银行在一个月（自然日）内同一原因查询的，可以算作一次查询记录；
        if x['days'] <= 60 and x['query_reason'] == 'xs':
            return True
        return False

    feature['pboc_negative_query_001'] = 1 if len(
        dt[dt.apply(condition001, axis=1)].drop_duplicates(['querier', 'query_reason'])) >= 5 else 0

    return feature


def loan_info_bom(loan_info):
    """
    贷款
    2.
        ustl 贷款未结清
        ustl1
        stl 贷款已结清
    3.
        mngm 经营
        cnsm 消费
        stdt 助学
        car 汽车
        frmr 农户
        othr 其他
    4.

    5.
        min,max,sum,average,cnt
    """
    source = ['pboc_ln']
    feature = dict()
    # settle_type = ['ustl', 'ustl1', 'stl', 'tot']
    settle_type = ['ustl1', 'tot']
    # items_type = {'mngm': u'经营', 'cnsm': u'消费', 'stdt': u'助学', 'car': u'汽车', 'frmr': u'农户', 'othr': u'其他',
    #               'hs': u'住房', 'tot': None}
    items_type = {'hs': u'住房', 'tot': None, 'bank': u'银行', 'nbank': u'非银行'}
    # due_flag = {'due': 1, 'nml': 0}
    if len(loan_info) == 0:
        return feature
    for s1 in source:
        for tw in TIME_WINDOW:
            df1 = loan_info[loan_info['open_days'] <= TIME_WINDOW[tw]].copy()
            if len(df1) == 0:
                continue
            # 贷款发放机构数
            feature['pboc_ln_org_nno_{0}'.format(tw)] = len(set(df1['loan_from']))
            feature['pboc_ln_blc_org_nno_{0}'.format(tw)] = len(set(df1[df1['balance'] > 0]['loan_from']))
            # 基本信息
            for s2 in settle_type:
                if s2 == 'tot':
                    df2 = df1
                else:
                    df2 = df1[df1['settle_type'] == s2]
                for s3 in items_type:
                    if items_type[s3] is not None:
                        df3 = df2[df2['loan_item'].apply(lambda x: s3 in x.split(','))]
                    else:
                        df3 = df2
                    if len(df3) == 0:
                        continue
                    # 记录
                    feature['{0}_{1}_{2}_rcrd_sum_{3}'.format(s1, s2, s3, tw)] = len(df3)
                    # day since first record or last record (dsst)
                    feature['{0}_{1}_{2}_dsst_max_{3}'.format(s1, s2, s3, tw)] = max(df3['open_days'])
                    feature['{0}_{1}_{2}_dsst_min_{3}'.format(s1, s2, s3, tw)] = min(df3['open_days'])
                    feature['{0}_{1}_{2}_msst_max_{3}'.format(s1, s2, s3, tw)] = max(df3['open_days']) / 30
                    # 额度
                    feature['{0}_{1}_cl_sum_{2}'.format(s1, s2, tw)] = sum(df3['credit_limit'])
                    feature['{0}_{1}_cl_sum_{2}'.format(s1, s2, tw)] = sum(df3['credit_limit'])
                    feature['{0}_{1}_cl_max_{2}'.format(s1, s2, tw)] = max(df3['credit_limit'])
                    feature['{0}_{1}_cl_min_{2}'.format(s1, s2, tw)] = min(df3['credit_limit'])
                    feature['{0}_{1}_cl_avg_{2}'.format(s1, s2, tw)] = sum(df3['credit_limit']) / len(df3)
                    # 期数
                    feature['{0}_{1}_term_max_{2}'.format(s1, s2, tw)] = np.max(df3['loan_terms'])
                    feature['{0}_{1}_term_min_{2}'.format(s1, s2, tw)] = np.min(df3['loan_terms'])
                    v = df3['loan_terms'][df3['loan_terms'].apply(lambda x: not pd.isnull(x))]
                    if len(v) > 0:
                        feature['{0}_{1}_term_avg_{2}'.format(s1, s2, tw)] = np.average(v)

                    # balance
                    feature['{0}_{1}_{2}_balance_min_{3}'.format(s1, s2, s3, tw)] = np.min(df3['balance'])
                    feature['{0}_{1}_{2}_balance_max_{3}'.format(s1, s2, s3, tw)] = np.max(df3['balance'])
                    feature['{0}_{1}_{2}_balance_avg_{3}'.format(s1, s2, s3, tw)] = np.average(df3['balance'])
                    # 已还与应还的占比
                    feature['{0}_{1}_ppct_max_{2}'.format(s1, s2, tw)] = np.max(
                        (df3['actualPaymentAmount'] / df3['scheduledPaymentAmount']).apply(
                            lambda x: None if np.inf == x else x))
                    feature['{0}_{1}_ppct_min_{2}'.format(s1, s2, tw)] = np.min(
                        (df3['actualPaymentAmount'] / df3['scheduledPaymentAmount']).apply(
                            lambda x: None if np.inf == x else x))
                    feature['{0}_{1}_ppct_avg_{2}'.format(s1, s2, tw)] = np.sum(df3['actualPaymentAmount']) / np.sum(
                        df3['scheduledPaymentAmount'].apply(lambda x: None if np.inf == x else x))

        for tw in TIME_WINDOW_V2:
            df1 = loan_info[loan_info['months'] <= TIME_WINDOW_V2[tw]].copy()
            for s3 in items_type:
                if items_type[s3] is not None:
                    df3 = df1[df1['loan_item'].apply(lambda x: s3 in x.split(','))]
                else:
                    df3 = df1
                if len(df3) == 0:
                    continue
                # day since first record or last record (dsst)
                feature['{0}_due_msst_min_{1}'.format(s1, tw)] = np.min(df3['months'])
                feature['{0}_due_msst_max_{1}'.format(s1, tw)] = np.max(df3['months'])
                # 最大逾期次数
                feature['{0}_due_rcrd_max_{1}'.format(s1, tw)] = np.max(df3.groupby(['account']).apply(len))
                # 逾期次数
                se31 = df3[df3['due_last_months'] >= 4].groupby(['account'])['account'].apply(len)
                feature['{0}_due90p_rcrd_max_{1}'.format(s1, tw)] = np.max(se31)
                feature['{0}_due90p_rcrd_sum_{1}'.format(s1, tw)] = np.sum(se31)
                feature['{0}_due90p_rcrd_avg_{1}'.format(s1, tw)] = np.average(se31)
                se32 = df3[df3['due_last_months'] >= 2].groupby(['account'])['account'].apply(len)
                feature['{0}_due30p_rcrd_max_{1}'.format(s1, tw)] = np.max(se32)
                feature['{0}_due30p_rcrd_sum_{1}'.format(s1, tw)] = np.sum(se32)
                feature['{0}_due30p_rcrd_avg_{1}'.format(s1, tw)] = np.average(se32)
                feature['{0}_due_rcrd_sum_{1}'.format(s1, tw)] = len(df3.index)
                feature['{0}_due_months_max_{1}'.format(s1, tw)] = np.max(df3['due_last_months'])
                feature['{0}_due_months_avg_{1}'.format(s1, tw)] = np.average(df3['due_last_months'])
                feature['{0}_cdue_amount_sum_{1}'.format(s1, tw)] = np.max(df3['currOverdueAmount'])
                # 逾期账户数
                feature['{0}_due_rcrd_nno_{1}'.format(s1, tw)] = len(set(df3['account']))

            feature['{0}_curr_due_cyc_max_{1}'.format(s1, tw)] = np.max(df1['currOverdueCyc'])

    feature['pboc_negative_loan_001'] = pboc_negative_loan_001(loan_info)
    feature['pboc_negative_loan_002'] = pboc_negative_loan_002(loan_info)
    feature['pboc_negative_loan_003'] = pboc_negative_loan_003(loan_info)
    feature['pboc_negative_loan_004'] = pboc_negative_loan_004(loan_info)
    feature['pboc_negative_loan_005'] = pboc_negative_loan_005(loan_info)
    feature['pboc_negative_loan_006'] = pboc_negative_loan_006(loan_info)
    feature['pboc_negative_loan_007'] = pboc_negative_loan_007(loan_info)

    return feature


def loan_card_bom(credit_analyzes):
    """
    信用卡
    1.贷记卡状态: 销户,未激活,正常
    """
    source = ['pboc_lc']
    feature = dict()
    # account_state = {'nml': ['正常', '销户'], 'nml1': ['正常'], 'cl': ['未激活', '销户'], 'tot': None}
    account_state = {'nml1': ['正常'], 'tot': None}
    dt = credit_analyzes
    if len(dt) == 0:
        return feature
    # 币种
    feature['pboc_lc_tot_ncur_lf'] = len(set(dt['accountType']))
    feature['pboc_lc_nml_ncur_lf'] = len(set(dt[dt['accountState'] == '正常']['accountType']))

    for s1 in source:
        for tw in TIME_WINDOW:
            df1 = dt[dt['open_days'] <= TIME_WINDOW[tw]].copy()
            if len(df1) == 0:
                continue
            # 发卡机构数
            feature['pboc_lc_org_nno_{0}'.format(tw)] = len(set(df1['loan_from']))
            feature['pboc_lc_blc_org_nno_{0}'.format(tw)] = len(
                set(df1[df1['credit_limit'] > df1['used_credit_limit']]['loan_from']))
            for s2, v in account_state.items():
                df2 = df1[df1['accountState'].isin(v)].copy() if v is not None else df1.copy()
                df21 = df2[df2['accountType'].apply(lambda x: '人民币' in x)]
                if len(df2) == 0:
                    continue
                feature['{0}_{1}_rcrd_sum_{2}'.format(s1, s2, tw)] = len(df2)  # 贷记卡数量
                feature['{0}_{1}_rcrd_nno_{2}'.format(s1, s2, tw)] = len(set(df2['loan_from']))  # 不同银行的贷记卡数量

                if len(df21) == 0:
                    continue
                # 最近6个月平均使用额度
                feature['{0}_{1}_uclj6_max_{2}'.format(s1, s2, tw)] = np.max(df21['latest6MonthUsedAvgAmount'])
                feature['{0}_{1}_uclj6_min_{2}'.format(s1, s2, tw)] = np.min(df21['latest6MonthUsedAvgAmount'])
                feature['{0}_{1}_uclj6_avg_{2}'.format(s1, s2, tw)] = np.average(df21['latest6MonthUsedAvgAmount'])

                # 信用卡已还与应还的占比
                feature['{0}_{1}_ppct_max_{2}'.format(s1, s2, tw)] = np.max(
                    (df21['actualPaymentAmount'] / df21['scheduledPaymentAmount']).apply(
                        lambda x: None if np.inf == x else x))
                feature['{0}_{1}_ppct_min_{2}'.format(s1, s2, tw)] = np.min(
                    (df21['actualPaymentAmount'] / df21['scheduledPaymentAmount']).apply(
                        lambda x: None if np.inf == x else x))
                feature['{0}_{1}_ppct_avg_{2}'.format(s1, s2, tw)] = np.sum(df21['actualPaymentAmount']) / np.sum(
                    df21['scheduledPaymentAmount']) if np.sum(df21['scheduledPaymentAmount']) != 0 else 0

                # 额度(只考虑人民币账户)
                feature['{0}_{1}_cl_sum_{2}'.format(s1, s2, tw)] = sum(df21['credit_limit'])
                feature['{0}_{1}_cl_max_{2}'.format(s1, s2, tw)] = max(df21['credit_limit'])
                feature['{0}_{1}_cl_min_{2}'.format(s1, s2, tw)] = min(df21['credit_limit'])
                feature['{0}_{1}_cl_avg_{2}'.format(s1, s2, tw)] = sum(df21['credit_limit']) / len(df21)
                feature['{0}_{1}_cl_latest_{2}'.format(s1, s2, tw)] = dt['credit_limit'][df21['open_days'].idxmin()]
                # 已使用额度(只考虑人民币账户)
                feature['{0}_{1}_ucl_sum_{2}'.format(s1, s2, tw)] = sum(df21['used_credit_limit'])
                feature['{0}_{1}_ucl_max_{2}'.format(s1, s2, tw)] = max(df21['used_credit_limit'])
                feature['{0}_{1}_ucl_min_{2}'.format(s1, s2, tw)] = min(df21['used_credit_limit'])
                feature['{0}_{1}_ucl_avg_{2}'.format(s1, s2, tw)] = sum(df21['used_credit_limit']) / len(df21)
                # 已使用额度占比(只考虑人民币账户)
                feature['{0}_{1}_ucl_pct_{2}'.format(s1, s2, tw)] = sum(df21['used_credit_limit']) / sum(
                    df21['credit_limit']) if sum(df21['credit_limit']) != 0 else 0

                # 开卡距今天数
                feature['{0}_{1}_dsst_max_{2}'.format(s1, s2, tw)] = max(df21['open_days'])
                feature['{0}_{1}_dsst_min_{2}'.format(s1, s2, tw)] = min(df21['open_days'])
                feature['{0}_{1}_dscl_max_{2}'.format(s1, s2, tw)] = df21['open_days'][
                    df21['credit_limit'].idxmax()]  # 最高额度贷记卡距申请天数

        # 逾期情况
        df = dt[dt['months'].apply(lambda x: not pd.isnull(x))]
        feature['{0}_due_rcrd_nno_lf'.format(s1)] = len(set(df['account']))  # 逾期过的贷记卡总数
        for tw in TIME_WINDOW:
            df1 = dt[dt['months'] <= TIME_WINDOW[tw]].copy()
            if len(df1) == 0:
                continue
            # 最大逾期次数
            feature['{0}_due_rcrd_max_{1}'.format(s1, tw)] = np.max(df1.groupby(['account']).apply(len))
            # 逾期次数
            feature['{0}_due90p_rcrd_max_{1}'.format(s1, tw)] = np.max(
                df1[df1['due_last_months'] >= 4].groupby(['account'])['account'].apply(len))
            feature['{0}_due30p_rcrd_max_{1}'.format(s1, tw)] = np.max(
                df1[df1['due_last_months'] >= 2].groupby(['account'])['account'].apply(len))
            feature['{0}_due_rcrd_sum_{1}'.format(s1, tw)] = len(df1.index)
            feature['{0}_due_months_max_{1}'.format(s1, tw)] = np.max(df1['due_last_months'])
            feature['{0}_cdue_amount_sum_{1}'.format(s1, tw)] = np.max(df1['currOverdueAmount'])
            # 逾期账户数
            feature['{0}_due_rcrd_nno_{1}'.format(s1, tw)] = len(set(df1['account']))
            # 逾期距离申请日天数
            feature['{0}_due_msst_max_{1}'.format(s1, tw)] = np.max(df1['months'])
            feature['{0}_due_msst_min_{1}'.format(s1, tw)] = np.min(df1['months'])

    feature['pboc_negative_lc_001'] = pboc_negative_lc_001(dt)
    feature['pboc_negative_lc_002'] = pboc_negative_lc_002(dt)
    feature['pboc_negative_lc_003'] = pboc_negative_lc_003(dt)
    feature['pboc_negative_lc_004'] = pboc_negative_lc_004(dt)
    feature['pboc_negative_lc_005'] = pboc_negative_lc_005(dt)

    return feature


def standard_loan_card_bom(dt):
    feature = dict()
    feature['pboc_negative_slc_001'] = pboc_negative_slc_001(dt)
    feature['pboc_negative_slc_002'] = pboc_negative_slc_002(dt)
    feature['pboc_negative_slc_003'] = pboc_negative_slc_003(dt)
    feature['pboc_negative_slc_004'] = pboc_negative_slc_004(dt)

    return feature


class PBOCEntity(object):
    """征信报告实体类"""

    def __init__(self, obj, version=None, _type=0):
        self.raw_data, self.version = self.load_pboc(obj)
        self._type = _type
        if version is not None:
            assert self.version == version
        self.query_time = self.get_query_time()
        self.basic_info = self.get_basic_info()
        # self.residence = self.get_residence()

        self.query_info = self.get_query_info_detail()
        self.credit_card_detail = self.get_loan_or_credit_detail(context='loanCard')
        self.standard_credit_card_detail = self.get_loan_or_credit_detail(context='standardLoanCard')
        self.loan_detail = self.get_loan_or_credit_detail()

    def load_pboc(self, obj):
        """获取pboc报文"""
        if not isinstance(obj, dict):
            raw_data = json.loads(obj)
        else:
            raw_data = obj
        version = 1
        return raw_data, version

    def get_query_time(self):
        """
        ------------
        version 1
        "messageHeader": {
                        "queryTime": "2019.03.22 11:22:25",
                        "reportCreateTime": "2019.03.22 11:22:26",
                        "reportSN": "2014120900000614319078"
                        }
        ------------
        version 2

        :return:
        """
        if self.version == 1:
            query_time = get_value('header,messageHeader,queryTime', self.raw_data)
            if query_time is None:
                return datetime.now()
            query_time = get_time(transfer_date(query_time))
        else:
            """
            header
            headerInfos
            messageHeader
            reportSN
            reportCreateTime
            """
            query_time = get_value('header,headerInfos,messageHeader,reportCreateTime', self.raw_data)
            query_time = get_time(transfer_date(query_time))
        return query_time

    def get_basic_info(self):
        """
        基本信息
        ------------
        version 1
        "identity": {
            "gender": "男性",
            "birthday": "1966.03.13",
            "maritalState": "未婚",
            "mobile": "13904108866",
            "officeTelephoneNo": "01060242299",
            "homeTelephoneNo": "01060242299",
            "eduLevel": "硕士",
            "eduDegree": "其他",
            "postAddress": "北京市朝阳区",
            "registeredAddress": "天津市武清"
            }
        ------------
        version 2
        """
        basic_info = {}
        if self.version == 1:
            identity = get_value('personalInfo,identity', self.raw_data, dict())
            basic_info['gender'] = get_value('gender', identity)
            basic_info['birthday'] = get_value('birthday', identity)
            basic_info['maritalState'] = get_value('maritalState', identity)
            basic_info['eduLevel'] = get_value('eduLevel', identity)
        else:
            """
            Identity
            identityInfos
            generalInfo
            gender
            birthday
            eduLevel
            eduDegree
            """
            identity = get_value('Identity,identityInfos,generalInfo', self.raw_data, dict())
            basic_info['gender'] = get_value('gender', identity)
            basic_info['birthday'] = get_value('birthday', identity)
            # basic_info['maritalState'] = get_value('maritalState', identity)
            basic_info['eduLevel'] = get_value('eduLevel', identity)
        return basic_info

    def get_residence(self):
        """
        居住地址信息
        :return:
        """
        residence = get_value('personalInfo,residence', self.raw_data)
        address_parsed = [address_parse(rd.get('address')) for rd in residence]
        rs1 = []
        for ii, rd1 in enumerate(address_parsed):
            rs2 = []
            for jj, rd2 in enumerate(address_parsed):
                rs2.append(sum(address_match_score(rd1, rd2)))
            rs1.append(rs2)
        sim_score = pd.DataFrame(rs1)
        residence = address_cls(residence, sim_score)
        return pd.DataFrame(residence)

    def get_query_info_detail(self):
        """ 查询记录
        query_reason: 信用卡审批 , 贷后管理 , 贷款审批 ,本人查询
        ------------
        version 1
        {
            "queryReason": "贷款审批",
            "queryDate": "2019.03.21",
            "querier": "中国工商银行/jack_ghuser1"
        }
        ------------
        version 2
        """
        if self.version == 1:
            query_info = get_value('queryRecord,recordInfo', self.raw_data)
            query_info = pd.DataFrame(query_info, columns=["queryReason", "queryDate", 'querier'])
            query_info['query_reason'] = query_info['queryReason'].apply(transfer_query_reason)
            query_info['query_date'] = query_info['queryDate'].apply(lambda x: get_time(transfer_date(x)))
            query_info['querier'] = query_info['querier']
            query_info['days'] = query_info['query_date'].apply(lambda x: (self.query_time.date() - x.date()).days)
        else:
            """
            version 2
            selectRecord
            selectRecords
            selectDate
            selecOrgType
            selecOrg
            selecReason
            """
            query_info = get_value('selectRecord,selectRecords', self.raw_data)
            query_info = pd.DataFrame(query_info, columns=["selecReason", "selectDate", 'selecOrg'])
            query_info['query_reason'] = query_info['selecReason'].apply(transfer_query_reason)
            query_info['query_date'] = query_info['selectDate'].apply(lambda x: get_time(transfer_date(x)))
            query_info['querier'] = query_info['selecOrg']
            query_info['days'] = query_info['query_date'].apply(lambda x: (self.query_time.date() - x.date()).days)
            query_info = query_info[["selecReason", "selectDate", 'selecOrg']]

        return query_info

    def get_credit_card_detail(self, card_type='loanCard'):
        """ 信用卡记录详细信息
        ------------
        version 1
        {
        "cue": "1.2011年11月23日商业银行u201CMRu201D发放的贷记卡（人民币账户），业务号X，授信额度50,000元，共享授信额度10,000元，信用/免担保。截至2018年08月25日，",
        "currOverdue": {
          "currOverdueCyc": "0",
          "currOverdueAmount": "0",
          "overdue31To60Amount": null,
          "overdue61To90Amount": null,
          "overdue91To180Amount": null,
          "overdueOver180Amount": null
        },
        "latest24MonthPaymentState": {
          "beginMonth": "2015年09月",
          "endMonth": "2017年08月",
          "latest24State": "NNNNNNNNNNNNNNNNNNNNNNNN",
          "state": null
        },
        "latest5YearOverdueRecord": null,
        "specialTrade": null,
        "bankIlluminate": null,
        "dissentInfo": null,
        "announceInfo": null,
        "awardCreditInfo": {
          "financeOrg": "MR",
          "account": "X",
          "openDate": "2011年11月23日",
          "creditLimitAmount": "50,000",
          "guaranteeType": "信用/免担保",
          "financeType": "",
          "currency": "人民币"
        },
        "repayInfo": {
          "stateEndDate": "2018年08月25日",
          "stateEndMonth": null,
          "scheduledPaymentAmount": "15,000",
          "scheduledPaymentDate": "2017.08.05",
          "actualPaymentAmount": "10,000",
          "recentPayDate": "2017.08.05",
          "shareCreditLimitAmount": "10,000",
          "usedCreditLimitAmount": "20,000",
          "latest6MonthUsedAvgAmount": "0",
          "usedHighestAmount": "30,000"
        },
        "state": "正常"
        }
        """
        if self.version == 1:
            if self._type == 0:
                loan_card_dt = get_value('creditDetail,loancard', self.raw_data)
                credit_card_list = []
                for cc in loan_card_dt:
                    record = dict()
                    latest5YearOverdueRecord = get_value('latest5YearOverdueRecord', cc)
                    if not empty_judge(latest5YearOverdueRecord):
                        for rd in get_value('overdueRecord', latest5YearOverdueRecord):
                            record = dict()
                            record['due_month'] = transfer_month(get_value('month', rd))
                            record['due_last_months'] = get_value('lastMonths', rd)
                            record['latest5YearOverdueRecord'] = latest5YearOverdueRecord
                            record['currOverdueCyc'] = get_value('currOverdue,currOverdueCyc', cc, '0')
                            record['currOverdueAmount'] = transfer_amount(
                                get_value('currOverdue,currOverdueAmount', cc, '0'))
                            record['overdue31To60Amount'] = get_value('currOverdue,overdue31To60Amount', cc, 0)
                            record['overdue61To90Amount'] = get_value('currOverdue,overdue61To90Amount', cc, 0)
                            record['overdue91To180Amount'] = get_value('currOverdue,overdue91To180Amount', cc, 0)
                            record['overdueOver180Amount'] = get_value('currOverdue,overdueOver180Amount', cc, 0)
                            record['state'] = get_value('state', cc, '正常')
                            record['credit_limit'] = transfer_amount(
                                get_value('awardCreditInfo,creditLimitAmount', cc, '0'))
                            record['currency'] = get_value('awardCreditInfo,currency', cc, 0)
                            record['openDate'] = get_value('awardCreditInfo,openDate', cc, 0)
                            record['account'] = get_value('awardCreditInfo,account', cc)
                            record['used_credit_limit'] = transfer_amount(
                                get_value('repayInfo,usedCreditLimitAmount', cc, '0'))
                            credit_card_list.append(record)
                        continue
                    record['latest5YearOverdueRecord'] = latest5YearOverdueRecord
                    record['currOverdueCyc'] = get_value('currOverdue,currOverdueCyc', cc, '0')
                    record['currOverdueAmount'] = transfer_amount(get_value('currOverdue,currOverdueAmount', cc, '0'))
                    record['overdue31To60Amount'] = get_value('currOverdue,overdue31To60Amount', cc, 0)
                    record['overdue61To90Amount'] = get_value('currOverdue,overdue61To90Amount', cc, 0)
                    record['overdue91To180Amount'] = get_value('currOverdue,overdue91To180Amount', cc, 0)
                    record['overdueOver180Amount'] = get_value('currOverdue,overdueOver180Amount', cc, 0)
                    record['state'] = get_value('state', cc, '正常')
                    record['credit_limit'] = transfer_amount(get_value('awardCreditInfo,creditLimitAmount', cc, '0'))
                    record['currency'] = get_value('awardCreditInfo,currency', cc, 0)
                    record['openDate'] = get_value('awardCreditInfo,openDate', cc, 0)
                    record['account'] = get_value('awardCreditInfo,account', cc)
                    record['used_credit_limit'] = transfer_amount(get_value('repayInfo,usedCreditLimitAmount', cc, '0'))
                    credit_card_list.append(record)

                cc_df = pd.DataFrame(credit_card_list)
                cc_df['openDate'] = cc_df['openDate'].apply(transfer_date)
                cc_df['open_days'] = cc_df['openDate'].apply(
                    lambda x: (self.query_time.date() - get_time(x).date()).days)
                cc_df['days'] = cc_df['openDate'].apply(lambda x: (self.query_time.date() - get_time(x).date()).days)
                cc_df['months'] = cc_df['due_month'].apply(
                    lambda x: round((self.query_time - get_time(x)).days / 30) if not pd.isnull(x) else None)
            else:
                loan_card_dt = get_value('creditDetail,{0}'.format(card_type), self.raw_data)
                if loan_card_dt is None:
                    return pd.DataFrame()
                credit_card_list = []
                for ii, cc in enumerate(loan_card_dt):
                    overdueRecords = get_value('overdueRecord', cc)
                    statements = get_value('statements', cc)
                    account = re.search('业务号(?P<account>[X\d]+)', statements).group('account')
                    account = '{0}{1}'.format(account, ii)
                    account_type = re.search('（(?P<accountType>.*?)账户）', statements).group('accountType')
                    account_state = re.search('账户状态为“(?P<accountState>.*?)”', statements)
                    if account_state is None:
                        account_state = '正常'
                    else:
                        account_state = account_state.group('accountState')
                    credit_limit = re.search('授信额度(.*?)(?P<creditlimit>[\d,\.]+)', statements).group('creditlimit')
                    openDate = re.search('\d{4}年\d{1,2}月\d{1,2}日', statements).group()
                    latest24Date = get_value('latest24Date', cc)
                    latest24State = get_value('latest24State', cc)
                    latest24OverdueInfo = get_overdue_info_from_latest24State(latest24Date, latest24State)
                    if len(latest24OverdueInfo) != 0 or (overdueRecords is not None and get_value('overdueRecordDetail',
                                                                                                  overdueRecords) is not None):
                        overdueRecordDetail = get_value('overdueRecordDetail', overdueRecords, [])
                        overdueRecordDetail += latest24OverdueInfo
                        for rd in overdueRecordDetail:
                            record = dict()
                            if get_value('month', rd) == '--':
                                record['due_month'] = None
                                record['due_last_months'] = 0
                            else:
                                record['due_month'] = transfer_month(get_value('month', rd))
                                record['due_last_months'] = get_value('lastMonths', rd)
                            record['currOverdueCyc'] = get_value('currOverdueCyc', cc, '0')
                            record['currOverdueAmount'] = transfer_amount(get_value('currOverdueAmount', cc, '0'))
                            record['scheduledPaymentDate'] = transfer_date(get_value('scheduledPaymentDate', cc))
                            record['overdue31To60Amount'] = get_value('overdue31To60Amount', cc, 0)
                            record['overdue61To90Amount'] = get_value('overdue61To90Amount', cc, 0)
                            record['overdue91To180Amount'] = get_value('overdue91To180Amount', cc, 0)
                            record['overdueOver180Amount'] = get_value('overdueOver180Amount', cc, 0)
                            record['state'] = get_value('state', cc, '正常')
                            record['credit_limit'] = transfer_amount(credit_limit)
                            # record['currency'] = get_value('awardCreditInfo,currency', cc, 0)
                            record['openDate'] = openDate
                            record['account'] = account
                            record['accountType'] = account_type
                            record['accountState'] = account_state
                            record['used_credit_limit'] = transfer_amount(get_value('usedCreditLimitAmount', cc, '0'))
                            record['usedHighestAmount'] = transfer_amount(get_value('usedHighestAmount', cc, '0'))
                            credit_card_list.append(record)
                    else:
                        record = dict()
                        record['currOverdueCyc'] = get_value('currOverdueCyc', cc, '0')
                        record['currOverdueAmount'] = transfer_amount(get_value('currOverdueAmount', cc, '0'))
                        record['scheduledPaymentDate'] = transfer_date(get_value('scheduledPaymentDate', cc))
                        record['overdue31To60Amount'] = get_value('overdue31To60Amount', cc, 0)
                        record['overdue61To90Amount'] = get_value('overdue61To90Amount', cc, 0)
                        record['overdue91To180Amount'] = get_value('overdue91To180Amount', cc, 0)
                        record['overdueOver180Amount'] = get_value('overdueOver180Amount', cc, 0)
                        record['state'] = get_value('state', cc, '正常')
                        record['credit_limit'] = transfer_amount(credit_limit)
                        # record['currency'] = get_value('currency', cc, 0)
                        record['openDate'] = openDate
                        record['account'] = account
                        record['accountType'] = account_type
                        record['accountState'] = account_state
                        record['due_month'] = None
                        record['due_last_months'] = 0
                        record['used_credit_limit'] = transfer_amount(get_value('usedCreditLimitAmount', cc, '0'))
                        record['usedHighestAmount'] = transfer_amount(get_value('usedHighestAmount', cc, '0'))
                        credit_card_list.append(record)

                cc_df = pd.DataFrame(credit_card_list)
                cc_df['openDate'] = cc_df['openDate'].apply(transfer_date)
                cc_df['due_last_months'] = cc_df['due_last_months'].apply(float)
                cc_df['activate_days'] = cc_df['scheduledPaymentDate'].apply(
                    lambda x: (self.query_time.date() - get_time(x).date()).days if x is not None else None)
                cc_df['open_days'] = cc_df['openDate'].apply(
                    lambda x: (self.query_time.date() - get_time(x).date()).days)
                cc_df['days'] = cc_df['openDate'].apply(lambda x: (self.query_time.date() - get_time(x).date()).days)
                cc_df['months'] = cc_df['due_month'].apply(
                    lambda x: round((self.query_time - get_time(x)).days / 30) if not pd.isnull(x) else None)

        else:
            return
        return cc_df

    def get_loan_info_detail(self):
        """ 贷款记录详细信息
        """
        if self.version == 1:
            if self._type == 0:
                loan_info_detail = get_value('creditDetail,loan', self.raw_data)
                loan_info_list = []
                for li in loan_info_detail:
                    record = dict()
                    latest5YearOverdueRecord = get_value('latest5YearOverdueRecord', li)
                    if not empty_judge(latest5YearOverdueRecord):
                        for rd in get_value('overdueRecord', latest5YearOverdueRecord):
                            record = dict()
                            record['due_month'] = transfer_month(get_value('month', rd))
                            record['due_last_months'] = get_value('lastMonths', rd)
                            record['currOverdueCyc'] = get_value('currOverdue,currOverdueCyc', li, 0)
                            record['currOverdueAmount'] = transfer_amount(
                                get_value('currOverdue,currOverdueAmount', li, 0))
                            record['overdue31To60Amount'] = get_value('currOverdue,overdue31To60Amount', li, 0)
                            record['overdue61To90Amount'] = get_value('currOverdue,overdue61To90Amount', li, 0)
                            record['overdue91To180Amount'] = get_value('currOverdue,overdue91To180Amount', li, 0)
                            record['overdueOver180Amount'] = get_value('currOverdue,overdueOver180Amount', li, 0)
                            record['state'] = get_value('state', li, '正常')
                            record['latest5YearOverdueRecord'] = get_value('latest5YearOverdueRecord', li)
                            record['credit_limit'] = transfer_amount(
                                get_value('contractInfo,creditLimitAmount', li, '0'))
                            record['currency'] = get_value('contractInfo,currency', li, 0)
                            record['openDate'] = get_value('contractInfo,openDate', li, 0)
                            record['account'] = get_value('contractInfo,account', li, 0)
                            record['type'] = get_value('contractInfo,type', li, 0)
                            record['balance'] = transfer_amount(get_value('currAccountInfo,balance', li, '0'))
                            loan_info_list.append(record)
                        continue
                    record['currOverdueCyc'] = get_value('currOverdue,currOverdueCyc', li, 0)
                    record['currOverdueAmount'] = transfer_amount(get_value('currOverdue,currOverdueAmount', li, 0))
                    record['overdue31To60Amount'] = get_value('currOverdue,overdue31To60Amount', li, 0)
                    record['overdue61To90Amount'] = get_value('currOverdue,overdue61To90Amount', li, 0)
                    record['overdue91To180Amount'] = get_value('currOverdue,overdue91To180Amount', li, 0)
                    record['overdueOver180Amount'] = get_value('currOverdue,overdueOver180Amount', li, 0)
                    record['state'] = get_value('state', li, '正常')
                    record['latest5YearOverdueRecord'] = get_value('latest5YearOverdueRecord', li)
                    record['credit_limit'] = transfer_amount(get_value('contractInfo,creditLimitAmount', li, '0'))
                    record['currency'] = get_value('contractInfo,currency', li, 0)
                    record['openDate'] = get_value('contractInfo,openDate', li, 0)
                    record['account'] = get_value('contractInfo,account', li, 0)
                    record['type'] = get_value('contractInfo,type', li, 0)
                    record['balance'] = transfer_amount(get_value('currAccountInfo,balance', li, '0'))
                    loan_info_list.append(record)

                li_df = pd.DataFrame(loan_info_list)
                li_df['openDate'] = li_df['openDate'].apply(transfer_date)
                li_df['days'] = li_df['openDate'].apply(lambda x: (self.query_time.date() - get_time(x).date()).days)
                li_df['settle_type'] = li_df['balance'].apply(lambda x: 'ustl' if float(x) > 0 else 'stl')
                li_df['loan_item'] = li_df['type'].apply(transfer_loan_item)
                li_df['months'] = li_df['due_month'].apply(
                    lambda x: round((self.query_time - get_time(x)).days / 30) if not pd.isnull(x) else None)
                li_df['is_dued'] = li_df['latest5YearOverdueRecord'].apply(lambda x: 1 if x is not None else 0)
            else:
                loan_info_detail = get_value('creditDetail,loan', self.raw_data)
                loan_info_list = []
                if loan_info_detail is None:
                    return pd.DataFrame()
                for ii, li in enumerate(loan_info_detail):
                    statements = get_value('statements', li)
                    account_no = re.search('业务号(?P<account>[A-Z\d]+)', statements).group('account')
                    account_no = '{0}{1}'.format(account_no, ii)
                    credit_limit = re.search('发放(.*?)(?P<creditLimit>[\d,\.]+)元', statements).group('creditLimit')
                    open_date = re.search('\d{4}年\d{1,2}月\d{1,2}日', statements).group()
                    loan_from = re.search('{0}(?P<from>.*?)发放'.format(open_date), statements).group('from')
                    up2date = re.search('截至(?P<upToDate>[\d年月日]+)', statements)
                    loan_detail_type = re.search('[)）](?P<type>.*?贷款)', statements).group('type')
                    end_date = re.search('(?P<endDate>[\d年月日]+)到期', statements)
                    if end_date is not None:
                        end_date = end_date.group('endDate')
                    if up2date is not None:
                        up2date = up2date.group('upToDate')
                    loan_terms = re.search('，(?P<terms>\d+)期', statements)
                    if loan_terms is not None:
                        loan_terms = float(loan_terms.group('terms'))
                    loan_type = re.search('业务号.*?，(?P<type>.*?)，', statements).group('type')
                    loan_type = loan_type.replace('（', '').replace('）', '').replace('(', '').replace(')', '').replace(
                        '/', '')
                    settle_type = '结清' if '结清' in statements else None
                    overdue_records = get_value('overdueRecord', li)
                    latest24Date = get_value('latest24Date', li)
                    latest24State = get_value('latest24State', li)
                    latest24OverdueInfo = get_overdue_info_from_latest24State(latest24Date, latest24State)
                    if len(latest24OverdueInfo) != 0 or (
                            overdue_records is not None and get_value('overdueRecordDetail',
                                                                      overdue_records) is not None):
                        overdueRecordDetail = get_value('overdueRecordDetail', overdue_records, [])
                        overdueRecordDetail += latest24OverdueInfo
                        for rd in overdueRecordDetail:
                            if get_value('lastMonths', rd) == '--':
                                continue
                            record = dict()
                            record['due_month'] = transfer_month(get_value('month', rd))
                            record['due_last_months'] = float(get_value('lastMonths', rd))
                            record['currOverdueCyc'] = get_value('currOverdueCyc', li, '0')
                            record['currOverdueAmount'] = transfer_amount(get_value('currOverdueAmount', li, '0'))
                            record['overdue31To60Amount'] = transfer_amount(get_value('overdue31To60Amount', li, '0'))
                            record['overdue61To90Amount'] = transfer_amount(get_value('overdue61To90Amount', li, '0'))
                            record['overdue91To180Amount'] = transfer_amount(get_value('overdue91To180Amount', li, '0'))
                            record['overdueOver180Amount'] = transfer_amount(get_value('overdueOver180Amount', li, '0'))
                            record['scheduledPaymentAmount'] = transfer_amount(
                                get_value('scheduledPaymentAmount', li, '0'))
                            remain_cycle = get_value('remainPaymentCyc', li, '0')
                            record['remainPaymentCyc'] = None if remain_cycle == '--' else float(remain_cycle)
                            record['scheduledPaymentDate'] = transfer_date(get_value('scheduledPaymentDate', li))
                            record['class5State'] = get_value('class5State', li)
                            record['state'] = get_value('state', li, '正常')
                            record['type'] = loan_detail_type
                            record['loan_type'] = loan_type
                            record['loan_terms'] = loan_terms
                            record['loan_from'] = loan_from
                            record['credit_limit'] = transfer_amount(credit_limit)
                            record['openDate'] = open_date
                            record['upToDate'] = transfer_date(up2date)
                            record['end_date'] = transfer_date(end_date)
                            record['account'] = account_no
                            record['balance'] = transfer_amount(get_value('balance', li, '0'))
                            record['settle_type'] = settle_type
                            loan_info_list.append(record)
                    else:
                        record = dict()
                        record['scheduledPaymentAmount'] = transfer_amount(get_value('scheduledPaymentAmount', li, '0'))
                        remain_cycle = get_value('remainPaymentCyc', li, '0')
                        record['remainPaymentCyc'] = None if remain_cycle == '--' else float(remain_cycle)
                        record['scheduledPaymentDate'] = transfer_date(get_value('scheduledPaymentDate', li))
                        record['currOverdueCyc'] = get_value('currOverdueCyc', li, '0')
                        record['currOverdueAmount'] = transfer_amount(get_value('currOverdueAmount', li, '0'))
                        record['overdue31To60Amount'] = transfer_amount(get_value('overdue31To60Amount', li, '0'))
                        record['overdue61To90Amount'] = transfer_amount(get_value('overdue61To90Amount', li, '0'))
                        record['overdue91To180Amount'] = transfer_amount(get_value('overdue91To180Amount', li, '0'))
                        record['overdueOver180Amount'] = transfer_amount(get_value('overdueOver180Amount', li, '0'))
                        record['state'] = get_value('state', li, '正常')
                        record['class5State'] = get_value('class5State', li)
                        record['credit_limit'] = transfer_amount(credit_limit)
                        record['type'] = loan_detail_type
                        record['loan_from'] = loan_from
                        record['loan_terms'] = loan_terms
                        record['openDate'] = open_date
                        record['end_date'] = end_date
                        record['upToDate'] = transfer_date(up2date)
                        record['account'] = account_no
                        record['due_month'] = None
                        record['due_last_months'] = 0
                        record['loan_type'] = loan_type
                        record['balance'] = transfer_amount(get_value('balance', li, '0'))
                        record['settle_type'] = '结清' if '结清' in statements else None
                        loan_info_list.append(record)
                li_df = pd.DataFrame(loan_info_list)
                li_df['openDate'] = li_df['openDate'].apply(transfer_date)
                li_df['end_date'] = li_df['end_date'].apply(transfer_date)
                li_df['end_days'] = li_df.apply(lambda x: (
                        (self.query_time.date() if x['upToDate'] is None else get_time(
                            x['upToDate']).date()) - get_time(
                    x['end_date']).date()).days if x['end_date'] is not None else None, axis=1)
                li_df['open_days'] = li_df['openDate'].apply(
                    lambda x: (self.query_time.date() - get_time(x).date()).days)
                li_df['activate_days'] = li_df['scheduledPaymentDate'].apply(
                    lambda x: (self.query_time.date() - get_time(x).date()).days if x is not None else None)
                li_df['settle_type'] = li_df.apply(
                    lambda x: 'ustl' if x['settle_type'] != '结清' and x['end_days'] > 0 and x['balance'] > 0 else (
                        'stl' if x['settle_type'] == '结清' else None), axis=1)
                # li_df['days'] = li_df['openDate'].apply(lambda x: (self.query_time.date() - get_time(x).date()).days)
                li_df['loan_item'] = li_df['type'].apply(transfer_loan_item)
                li_df['months'] = li_df['due_month'].apply(
                    lambda x: round((self.query_time - get_time(x)).days / 30) if not pd.isnull(x) else None)
                li_df['is_dued'] = li_df['due_month'].apply(lambda x: 1 if x is not None else 0)
        else:
            return
        return li_df

    def get_loan_or_credit_detail(self, context='loan'):
        loan_info_detail = get_value('creditDetail,{0}'.format(context), self.raw_data)
        loan_info_list = []
        if loan_info_detail is None:
            return pd.DataFrame()
        for ii, li in enumerate(loan_info_detail):
            loan_from, loan_detail_type, end_date, loan_terms, loan_type, account_type, account_state = [None] * 7
            statements = get_value('statements', li)
            account_no = re.search('业务号(?P<account>[A-Z\d]+)', statements).group('account')
            account_no = '{0}{1}'.format(account_no, ii)
            open_date = re.search('\d{4}年\d{1,2}月\d{1,2}日', statements).group()
            up2date = re.search('截至(?P<upToDate>[\d年月日]+)', statements)
            loan_from = re.search('{0}(?P<from>.*?)发放'.format(open_date), statements).group('from')
            if context == 'loan':
                credit_limit = re.search('发放(.*?)(?P<creditLimit>[\d,\.]+)元', statements).group('creditLimit')
                loan_detail_type = re.search('[)）](?P<type>.*?贷款)', statements).group('type')
                end_date = re.search('(?P<endDate>[\d年月日]+)到期', statements)
                loan_terms = re.search('，(?P<terms>\d+)期', statements)
                loan_type = re.search('业务号.*?，(?P<type>.*?)，', statements).group('type')
                loan_type = loan_type.replace('（', '').replace('）', '').replace('(', '').replace(')', '').replace(
                    '/', '')
            else:
                credit_limit = re.search('授信额度(.*?)(?P<creditLimit>[\d,\.]+)', statements).group('creditLimit')
                account_type = re.search('（(?P<accountType>.*?)账户）', statements).group('accountType')
                account_state = re.search('账户状态为“(?P<accountState>.*?)”', statements)
                account_state = "正常" if account_state is None else account_state.group('accountState')
            if end_date is not None:
                end_date = end_date.group('endDate')
            if up2date is not None:
                up2date = up2date.group('upToDate')
            if loan_terms is not None:
                loan_terms = float(loan_terms.group('terms'))
            else:
                if '一次性归还' in statements:
                    loan_terms = 1
                else:
                    if end_date is not None:
                        loan_terms = (parser.parse(re.sub('[^\d]+', '', end_date)).date() - parser.parse(
                            re.sub('[^\d]+', '', open_date)).date()).days / 30
                        loan_terms = int(loan_terms)

            settle_type = '结清' if '结清' in statements else None
            overdue_records = get_value('overdueRecord', li, {})
            latest24_date = get_value('latest24Date', li)
            latest24_state = get_value('latest24State', li)
            latest24_overdue_info = get_overdue_info_from_latest24State(latest24_date, latest24_state)
            if len(latest24_overdue_info) != 0 or get_value('overdueRecordDetail', overdue_records) is not None:
                overdue_record_detail = get_value('overdueRecordDetail', overdue_records, [])
                overdue_record_detail += latest24_overdue_info
                for rd in overdue_record_detail:
                    if get_value('lastMonths', rd) == '--':
                        continue
                    record = dict()
                    record['due_month'] = transfer_month(get_value('month', rd))
                    record['due_last_months'] = float(get_value('lastMonths', rd))
                    record['currOverdueCyc'] = get_value('currOverdueCyc', li, '0')
                    record['currOverdueAmount'] = transfer_amount(get_value('currOverdueAmount', li, '0'))
                    record['overdue31To60Amount'] = transfer_amount(get_value('overdue31To60Amount', li, '0'))
                    record['overdue61To90Amount'] = transfer_amount(get_value('overdue61To90Amount', li, '0'))
                    record['overdue91To180Amount'] = transfer_amount(get_value('overdue91To180Amount', li, '0'))
                    record['overdueOver180Amount'] = transfer_amount(get_value('overdueOver180Amount', li, '0'))
                    record['scheduledPaymentAmount'] = transfer_amount(get_value('scheduledPaymentAmount', li, '0'))
                    record['actualPaymentAmount'] = transfer_amount(get_value('actualPaymentAmount', li, '0'))
                    remain_cycle = get_value('remainPaymentCyc', li, '0')
                    record['remainPaymentCyc'] = None if remain_cycle == '--' else float(remain_cycle)
                    record['scheduledPaymentDate'] = transfer_date(get_value('scheduledPaymentDate', li))
                    record['class5State'] = get_value('class5State', li)
                    record['state'] = get_value('state', li, '正常')
                    record['type'] = loan_detail_type
                    record['loan_type'] = loan_type
                    record['loan_terms'] = loan_terms
                    record['loan_from'] = loan_from
                    record['credit_limit'] = transfer_amount(credit_limit)
                    record['used_credit_limit'] = transfer_amount(get_value('usedCreditLimitAmount', li, '0'))
                    record['usedHighestAmount'] = transfer_amount(get_value('usedHighestAmount', li, '0'))
                    record['latest6MonthUsedAvgAmount'] = transfer_amount(
                        get_value('latest6MonthUsedAvgAmount', li, '0'))
                    record['openDate'] = open_date
                    record['upToDate'] = transfer_date(up2date)
                    record['end_date'] = transfer_date(end_date)
                    record['account'] = account_no
                    record['accountType'] = account_type
                    record['accountState'] = account_state
                    record['balance'] = transfer_amount(get_value('balance', li, '0'))
                    record['settle_type'] = settle_type
                    loan_info_list.append(record)
            else:
                record = dict()
                record['due_month'] = None
                record['due_last_months'] = 0
                record['scheduledPaymentAmount'] = transfer_amount(get_value('scheduledPaymentAmount', li, '0'))
                record['actualPaymentAmount'] = transfer_amount(get_value('actualPaymentAmount', li, '0'))
                remain_cycle = get_value('remainPaymentCyc', li, '0')
                record['remainPaymentCyc'] = None if remain_cycle == '--' else float(remain_cycle)
                record['scheduledPaymentDate'] = transfer_date(get_value('scheduledPaymentDate', li))
                record['currOverdueCyc'] = get_value('currOverdueCyc', li, '0')
                record['currOverdueAmount'] = transfer_amount(get_value('currOverdueAmount', li, '0'))
                record['overdue31To60Amount'] = transfer_amount(get_value('overdue31To60Amount', li, '0'))
                record['overdue61To90Amount'] = transfer_amount(get_value('overdue61To90Amount', li, '0'))
                record['overdue91To180Amount'] = transfer_amount(get_value('overdue91To180Amount', li, '0'))
                record['overdueOver180Amount'] = transfer_amount(get_value('overdueOver180Amount', li, '0'))
                record['state'] = get_value('state', li, '正常')
                record['class5State'] = get_value('class5State', li)
                record['credit_limit'] = transfer_amount(credit_limit)
                record['used_credit_limit'] = transfer_amount(get_value('usedCreditLimitAmount', li, '0'))
                record['usedHighestAmount'] = transfer_amount(get_value('usedHighestAmount', li, '0'))
                record['latest6MonthUsedAvgAmount'] = transfer_amount(get_value('latest6MonthUsedAvgAmount', li, '0'))
                record['type'] = loan_detail_type
                record['loan_from'] = loan_from
                record['loan_terms'] = loan_terms
                record['openDate'] = open_date
                record['end_date'] = end_date
                record['upToDate'] = transfer_date(up2date)
                record['account'] = account_no
                record['accountType'] = account_type
                record['accountState'] = account_state
                record['loan_type'] = loan_type
                record['balance'] = transfer_amount(get_value('balance', li, '0'))
                record['settle_type'] = '结清' if '结清' in statements else None
                loan_info_list.append(record)
        li_df = pd.DataFrame(loan_info_list)
        li_df['openDate'] = li_df['openDate'].apply(transfer_date)
        li_df['end_date'] = li_df['end_date'].apply(transfer_date)
        li_df['end_days'] = li_df.apply(self.get_end_days, axis=1)
        li_df['up_to_days'] = li_df['upToDate'].apply(self.get_up_to_days)
        li_df['open_days'] = li_df['openDate'].apply(
            lambda x: (self.query_time.date() - get_time(x).date()).days)
        li_df['activate_days'] = li_df['scheduledPaymentDate'].apply(
            lambda x: (self.query_time.date() - get_time(x).date()).days if x is not None else None)
        if context == 'loan':
            li_df['settle_type'] = li_df.apply(transfer_settle_type, axis=1)
        li_df['loan_item'] = li_df.apply(transfer_loan_item_v1, axis=1)
        li_df['months'] = li_df['due_month'].apply(
            lambda x: round((self.query_time - get_time(x)).days / 30) if not pd.isnull(x) else None)
        li_df['is_dued'] = li_df['due_month'].apply(lambda x: 1 if x is not None else 0)

        return li_df

    def get_end_days(self, x):
        end = self.query_time.date() if pd.isnull(x['upToDate']) else get_time(x['upToDate']).date()
        return (end - get_time(x['end_date']).date()).days if not pd.isnull(x['end_date']) else None

    def get_up_to_days(self, up2date):
        if pd.isnull(up2date):
            return 0
        else:
            return (self.query_time.date() - get_time(up2date).date()).days


def hbxd_house_loan_feature(pboc_entity: PBOCEntity):
    """

    :param pboc_entity:
    :return:
    """
    features = dict()

    ldf = pboc_entity.loan_detail.copy()
    if len(ldf) == 0:
        return features
    ldf = ldf.drop_duplicates(['account'])
    ldf['hs_type_check_level1'] = ldf.apply(lambda x: hbxd_house_loan_type_check(x, 1), axis=1)
    ldf['hs_type_check_level2'] = ldf.apply(lambda x: hbxd_house_loan_type_check(x, 2), axis=1)
    ldf['hs_admission'] = ldf.apply(hbxd_house_loan_admission, axis=1)
    ldf.index = ldf['account']
    accounts = set(ldf.index)
    rs_lst = []
    cmp_dct = {}
    counter = 0
    for ii, record in enumerate(get_value('creditDetail,loan', pboc_entity.raw_data, [])):
        if 'X{0}'.format(ii) not in accounts:
            continue
        dct = dict(ldf.ix['X{0}'.format(ii)])
        if not dct['hs_type_check_level1'] or not dct['hs_admission']:
            continue
        states = record.get('latest24State', '')
        dct['repay_months'] = len(re.split('[^/*#]', states, 1)[-1]) + 1 + max(int(dct['open_days']) // 30 - 24, 0)
        if dct['repay_months'] < 6:
            continue
        open_date = dct['openDate']
        hs = 'hs' if '个人住房' in dct['type'] or '公积金' in dct['type'] else 'ot'
        loan_from = dct['loan_from']
        loan_type = dct['loan_type']
        select = None
        if hs == 'hs':
            k1 = '{0},{1}'.format(open_date, hs)
            if k1 not in cmp_dct:
                cmp_dct[k1] = counter
            else:
                select = cmp_dct[k1]
        k2 = '{0},{1},{2}'.format(open_date, loan_from, loan_type)
        if k2 not in cmp_dct:
            cmp_dct[k2] = counter
        else:
            select = cmp_dct[k2]
        if select is not None:
            rs_lst[select]['scheduledPaymentAmount'] += dct['scheduledPaymentAmount']
        else:
            rs_lst.append(dct)
            counter += 1
    ldf = pd.DataFrame(rs_lst)
    if len(ldf) == 0:
        return features
    ldf = ldf[ldf['scheduledPaymentAmount'] >= 2000]
    if len(ldf) == 0:
        return features
    ldf = ldf[['open_days', 'repay_months', 'hs_type_check_level2', 'scheduledPaymentAmount']].copy()
    ldf['repay_amount_monthly_coefficient'] = ldf.apply(_cal_repay_amount_monthly_coefficient, axis=1)
    ldf['credit_limit_coefficient'] = ldf.apply(_cal_amount_coefficient, axis=1)

    idxmax = (ldf['repay_amount_monthly_coefficient'] * ldf['scheduledPaymentAmount']).idxmax()

    features['pboc_hs_coffiecient_level1'] = ldf.loc[idxmax, 'credit_limit_coefficient']
    features['pboc_hs_repay_monthly_coffiecient_level1'] = ldf.loc[idxmax, 'credit_limit_coefficient']
    features['pboc_hs_credit_limit_level1'] = ldf.loc[idxmax, 'credit_limit_coefficient'] * ldf.loc[idxmax, 'scheduledPaymentAmount']

    ldf2 = ldf[ldf['hs_type_check_level2']]
    if len(ldf2) > 0:
        idxmax2 = (ldf2['repay_amount_monthly_coefficient'] * ldf2['scheduledPaymentAmount']).idxmax()
        features['pboc_hs_coffiecient_level2'] = ldf2.loc[idxmax2, 'credit_limit_coefficient']
        features['pboc_hs_repay_monthly_coffiecient_level2'] = ldf2.loc[idxmax2, 'credit_limit_coefficient']
        features['pboc_hs_credit_limit_level2'] = ldf2.loc[idxmax2, 'credit_limit_coefficient'] * ldf2.loc[idxmax2, 'scheduledPaymentAmount']

    return features


def _cal_repay_amount_monthly_coefficient(x):
    """

    :param x:
    :return:
    X≥36个月	房贷月还款额×8	40
12个月≤X<36个月	房贷月还款额×8	25
6个月≤X<12个月	房贷月还款额×6	12
    """
    if x['repay_months'] >= 36:
        return 8
    elif x['repay_months'] >= 12:
        return 8
    elif x['repay_months'] >= 6:
        return 8
    else:
        return 0


def _cal_amount_coefficient(x):
    """

    :param x:
    :return:
    X≥36个月	房贷月还款额×8	40
12个月≤X<36个月	房贷月还款额×8	25
6个月≤X<12个月	房贷月还款额×6	12
    """
    if x['repay_months'] >= 36:
        return 40
    elif x['repay_months'] >= 12:
        return 25
    elif x['repay_months'] >= 6:
        return 12
    else:
        return 0


class Address(object):
    """
    province	string	是	省
    city	string	是	市
    district	string	是	区，可能为空字串
    street	string	是	街道，可能为空字串
    """
    prefix = None
    tail = None
    province = None
    city = None
    district = None
    street = None
    detail = None

    def __init__(self, *args):
        if len(args) != 0:
            self.province, self.city, self.district, self.street, self.detail = args

    def value_count(self) -> List[int]:
        prefix_cnt, tail_cnt, cnt = 0, 0, 0
        if self.province is not None:
            prefix_cnt += 1
            cnt += 1
        if self.city is not None:
            prefix_cnt += 1
            cnt += 1
        if self.district is not None:
            prefix_cnt += 1
            cnt += 1
        if self.street is not None:
            tail_cnt += 1
            cnt += 1
        if self.detail is not None:
            tail_cnt += 1
            cnt += 1
        return [prefix_cnt, tail_cnt, cnt]


class Residence(object):
    """
    pass
    """
    residence_type = None
    get_time = None
    address = None
    address_his = None

    def __init__(self, *args):
        if len(args) > 0:
            self.residence_type, self.get_time, self.address = args
            self.address_his = self.address
        else:
            self.address_his = []

    def union(self, residence):
        if isinstance(self.address_his, list):
            self.address_his.append(residence.address)
        else:
            self.address_his = [self.address, residence.address]
        if self.residence_type is None:
            self.residence_type = residence.residence_type
        if self.get_time is None:
            self.get_time = residence.get_time
        if self.address is None:
            self.address = residence.address
        return self


def address_parse(address) -> Address:
    """
    居住地址解析
    :param address:
    :return:
    """
    #
    address = re.sub('(^中国)|(--)|(待补充)|(UNKNOW)', '', address)
    # model 1
    province_pattern = '(?P<province>.*?省)?'
    city_pattern = '(?P<city>.*?((自治州)|市(?!场)))?'
    district_pattern = '(?P<district>(((.*[^社工业市\d一二三四五六七八九东南])区)|((.*?[^城])市)|(.*?县))?)'
    prefix_pattern = '(?P<prefix>{0}{1}{2})'.format(province_pattern, city_pattern, district_pattern)
    street_pattern = '(?P<street>((.+路(\d+号)?)|(.+[街道](\d+号)?)|(.*?[镇乡].*?村)))?'
    detail_pattern = '(?P<detail>.*)'
    tail_pattern = '(?P<tail>{0}{1})'.format(street_pattern, detail_pattern)
    pattern = prefix_pattern + tail_pattern
    rs = re.search(pattern, address)
    add = Address()
    prefix = rs.group('prefix')
    tail = rs.group('tail')
    province = rs.group('province')
    city = rs.group('city')
    district = rs.group('district')
    street = rs.group('street')
    detail = rs.group('detail')

    address_prefix_detail_parse(prefix, add, **{'province': province, 'city': city, 'district': district})
    address_tail_detail_parse(tail, add)
    add.street = street
    add.detail = detail

    return add


def address_cell_cmp(x1, x2) -> bool:
    if x1 is None or x2 is None:
        return True
    else:
        return x1 in x2 or x2 in x1


def string_similarity(x1: str, x2: str) -> float:
    def add_space(s: str):
        return ' '.join(list(s))

    # 将字中间加入空格
    s1, s2 = add_space(x1), add_space(x2)
    # 转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 计算TF系数
    rs = np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))
    return rs


def address_cls(address: List[dict], sim_score: pd.DataFrame) -> List[dict]:
    """
    score > 8 即可视为一类
    :param address:
    :param sim_score:
    :return:
    """
    cls = defaultdict(list)
    cls_count = 0
    cutoff = 8
    for ii, row in sim_score.iterrows():
        for jj, score in enumerate(row):
            if jj < ii:
                continue
            find = False
            if score >= cutoff:
                for k, v in cls.items():
                    if ii in v and jj not in v:
                        v.append(jj)
                        find = True
                    elif ii not in v and jj in v:
                        v.append(ii)
                        find = True
                    elif ii in v and jj in v:
                        find = True
            else:
                for k, v in cls.items():
                    if jj in v:
                        find = True
            if not find and jj == ii:
                cls[cls_count].append(jj)
                cls_count += 1

    rs = []
    for k, v in cls.items():
        r = Residence()
        for index in v:
            add = address[index]
            r = r.union(Residence(add.get('residenceType'), add.get('getTime'), add.get('address')))
        rs.append(r.__dict__)
    return rs


def address_cls_v1(address: List[str]):
    rs = defaultdict(list)
    adds = address
    cls_counter = 0
    cls = [None] * len(adds)
    cutoff = 8
    for ii, add1 in enumerate(adds):
        for jj, add2 in enumerate(adds):
            if jj < ii:
                continue
            if cls[jj] is not None:
                continue
            if jj == ii:
                cls[jj] = cls_counter
                rs[cls_counter].append(address[jj])
                cls_counter += 1
                continue
            score = int(string_similarity(add1, add2) * 10)
            if score > cutoff:
                cls[jj] = cls[ii]
                rs[cls[ii]].append(address[jj])
    return rs


def address_vague_match(address: str, adds: List[str], num: int = 1) -> List[str]:
    """
    地址模糊匹配
    :param address:
    :param adds:
    :param num: 若num==-1,返回所有匹配成功的结果,否则返回指定数量的结果
    :return:
    """
    cutoff = 7
    matched_address, final_matched_address = [], []
    final_matched_counter = 0
    for ii, add in enumerate(adds):
        score = int(string_similarity(address, add) * 10)
        if score >= cutoff:
            matched_address.append(add)
    print(json.dumps(matched_address, ensure_ascii=False, indent=4))
    matched_address = [address_parse(add) for add in matched_address]
    parsed_add = address_parse(address)
    for ii, add in enumerate(matched_address):
        prefix_score, tail_score = address_match_score(parsed_add, add)
        final_matched_address.append((add.prefix + add.tail, tail_score))
        if prefix_score >= 3 and tail_score >= 0.75:
            # final_matched_address.append((add.prefix + add.tail, tail_score))
            final_matched_counter += 1
    final_matched_address = sorted(final_matched_address, key=lambda x: x[1], reverse=True)
    print(json.dumps(final_matched_address, ensure_ascii=False, indent=4))
    final_matched_address = [v[0] for v in final_matched_address]
    return final_matched_address if num == -1 else final_matched_address[:num]


def address_match_score(address1: Address, address2: Address):
    """
    地址匹配相似的得分
    :param address1:
    :param address2:
    :return:
    """
    score = 0
    if address_cell_cmp(address1.province, address2.province):
        score += 1
    if address_cell_cmp(address1.city, address2.city):
        score += 1
    if address_cell_cmp(address1.district, address2.district):
        score += 1
    return score, string_similarity(address1.tail, address2.tail)


def address_cell_fill(add: Address, p: str, index: int) -> Address:
    if '省' in p or '自治区' in p:
        add.province = p if add.province is None else add.province
    if '市' in p or '自治州' in p:
        if index < 2:
            add.city = p if add.city is None else add.city
    if '区' in p or '县' in p:
        add.district = p if add.district is None else add.district
    return add


def address_tail_detail_parse(tail: str, add: Address) -> Address:
    """
    区县级以下具体地址解构
    :param tail:
    :param add:
    :return:
    """
    add.tail = tail
    return add


def fix_jieba_over_cut(prefix_lst: List[str], **kwargs) -> List[str]:
    lst = []
    province, city, district = kwargs.get('province'), kwargs.get('city'), kwargs.get('district')
    for ii, p in enumerate(prefix_lst):
        match = False
        for c in ['市', '区', '县', '新区']:
            # 过度分词
            if p == c:
                lst[-1] = lst[-1] + p
                match = True
                break
        if match:
            continue
        for c in ['镇', '村', '乡', '街道', '路', '号']:
            # 过度匹配
            if p.endswith(c):
                lst.pop()
                match = True
                break
        if match:
            continue
        for jj, n in enumerate(lst):
            # 词重复
            if n in p:
                lst[jj] = p
                match = True
                break
            elif p in n:
                match = True
                break
        if not match:
            lst.append(p)
    if len(lst) > 3:
        last = ''.join(lst[2:])
        lst = lst[:2] + [last]
    return lst


def address_prefix_detail_parse(prefix: str, add: Address, **kwargs) -> Address:
    """
    区县级以上地址解构
    :param prefix:
    :param add:
    :return:
    """
    if len(prefix) == 0:
        return
    prefix_lst = list(jieba.cut(prefix))
    prefix_lst = fix_jieba_over_cut(prefix_lst, **kwargs)
    if ''.join(prefix_lst) != prefix:
        prefix = ''.join(prefix_lst)
        prefix_lst = list(jieba.cut(prefix))
        prefix_lst = fix_jieba_over_cut(prefix_lst)
    for ii, p in enumerate(prefix_lst):
        address_cell_fill(add, p, ii)
    prefix_value_count = add.value_count()[0]
    if len(prefix_lst) == 3 and prefix_value_count < 3:
        if add.province is None:
            add.province = prefix_lst[0]
        if add.city is None:
            add.city = prefix_lst[1]
        if add.district is None:
            add.district = prefix_lst[2]
    elif len(prefix_lst) == 1 and prefix_value_count < 1:
        if add.district is None:
            add.district = prefix_lst[0]
    elif len(prefix_lst) == 2 and prefix_value_count < 2:
        if add.city is None:
            add.city = prefix_lst[0]
        else:
            if prefix_lst.index(add.city) == 0:
                add.district = prefix_lst[1]
            else:
                add.province = prefix_lst[0]
    elif len(prefix_lst) > 3:
        print('位置情况: {0}'.format(prefix))
        if add.district is not None:
            add.district = prefix
    if add.district is not None and '市' in add.district:
        add.city = add.district
        add.district = None
    add.prefix = prefix
    return add


def transfer_settle_type(x):
    """
    已到期未结清（“结清”文字优先，未结清看逻辑：到期时间早于报告时间，且本金余额>0为未结清）
    :param x:
    :return:
    """
    if pd.isnull(x['settle_type']):
        if x['end_days'] >= 0 and x['balance'] > 0:
            return 'ustl'
        elif x['end_days'] < 0 and x['balance'] > 0:
            return 'ustl1'
        else:
            return
    elif x['settle_type'] == '结清':
        return 'stl'
    else:
        return


def transfer_query_reason(query_reason):
    """ reason"""
    reason = {u'信用卡审批': 'xs,dq', u'贷后管理': 'dg', u'贷款审批': 'dk,dq', u'本人查询': 'bc',
              u'本人查询（互联网个人信用信息服务平台）': 'bcn', u'保前审查': 'ot',
              u'本人查询（临柜）': 'ot', u'担保资格审查': 'ot', u'本人查询（商业银行网上银行）': 'bc', u'保后管理': 'ot'}
    if query_reason is None:
        return 'ot'
    elif query_reason in reason:
        return reason[query_reason]
    else:
        return 'ot'


def get_overdue_info_from_latest24State(latest24Date, latest24State):
    rs = []
    if latest24Date is None:
        return rs
    month = latest24Date.split('-')[0]
    month = transfer_month(month)
    for ii, v in enumerate(latest24State):
        if v.isdigit():
            dct = {}
            dct['month'] = month_adjust(month, ii)
            dct['lastMonths'] = v
            rs.append(dct)
    return rs


def month_adjust(datestr, diff_month=0):
    now = datetime.strptime(datestr, '%Y-%m-%d %H:%M:%S')
    now = now + relativedelta.relativedelta(months=diff_month)
    return now.strftime('%Y-%m-%d %H:%M:%S')


def transfer_query_operator(query_operator):
    if query_operator is None:
        return 'ot'
    elif u'消费金融' in query_operator:
        return 'xj'
    elif u'小额贷款' in query_operator:
        return 'ln'
    elif u'信用卡' in query_operator:
        return 'cd'
    elif (u'银行' in query_operator or u'花旗' in query_operator or u'农村信用' in query_operator) \
            and u'信用卡' not in query_operator:
        if (u'工商' in query_operator or u'农业' in query_operator or u'建设' in query_operator or u'中国银行' in query_operator
                or u'交通' in query_operator):
            return 'sb'
        else:
            return 'tb'
    elif u'本人' in query_operator:
        return 'own'
    elif (u'保险' in query_operator or u'信托' in query_operator or u'担保' in query_operator
          or u'汽车金融' in query_operator or u'租赁' in query_operator or u'证券' in query_operator):
        return 'nf'
    else:
        return 'ot'


def transfer_loan_item(loan_item):
    """  loan_item
    """
    if loan_item is None:
        return 'othr'
    elif u'住房' in loan_item:
        return 'hs'
    elif u'经营' in loan_item:
        return 'mngm'
    elif u'消费' in loan_item:
        return 'cnsm'
    elif u'助学' in loan_item:
        return 'stdt'
    elif u'汽车' in loan_item:
        return 'car'
    elif u'农户' in loan_item:
        return 'frmr'
    elif u'其他' in loan_item:
        return 'othr'
    else:
        return 'othr'


def transfer_loan_item_v1(x):
    """  loan_item
    """
    if x['type'] is None:
        return 'othr'
    elif '银行' in x['loan_from'] and '住房' in x['type']:
        return 'bank,hs'
    elif '住房' in x['type']:
        return 'nbank,hs'
    elif '银行' in x['loan_from']:
        return 'bank'
    else:
        return 'nbank'


def transfer_amount(amount):
    """

    :param amount:
    :return:
    """
    return float(re.sub('[^\d\.]', '', amount))


def transfer_date(datestr):
    """
    transfer date string like '2019.01.01' or '2019年11月11日' or '2019年1月1日'
    :param datestr:
    :return:
    """
    if datestr is None or datestr == '':
        return
    splits = re.split('[^\d]', datestr)
    year = splits[0]
    month = splits[1] if len(splits[1]) >= 2 else '0' + splits[1]
    day = splits[2] if len(splits[2]) >= 2 else '0' + splits[2]
    return '{0}-{1}-{2} 00:00:00'.format(year, month, day)


def transfer_month(datestr):
    """
    transfer date string like '2019.01' or '2019年11月' or '2019年1月'
    :param datestr:
    :return:
    """
    splits = re.split('[^\d]', datestr)
    year = splits[0]
    month = splits[1] if len(splits[1]) >= 2 else '0' + splits[1]
    return '{0}-{1}-01 00:00:00'.format(year, month)


def get_value(key_str, obj, default=None):
    """获取指定key的值,key的格式为`node1,node2`"""

    def g_val(key_, obj_):
        val = None
        if isinstance(obj_, str):
            obj_ = json.loads(obj_)
        elif isinstance(obj_, bytes):
            obj_ = json.loads(str(obj, encoding="utf8"))
        if isinstance(obj_, list):
            val = obj_[key_]
        elif isinstance(obj_, dict):
            val = obj_.get(key_)
        return default if empty_judge(val) else val

    if empty_judge(obj):
        return default
    key_lst = key_str.split(',')
    try:
        key = int(key_lst[0])
    except ValueError:
        key = key_lst[0]
    tp_obj = g_val(key, obj)
    if len(key_lst) <= 1 or empty_judge(tp_obj):
        return tp_obj
    else:
        return get_value(','.join(key_lst[1:]), tp_obj, default)


def empty_judge(obj):
    """ empty data judge"""
    if obj is None:
        return True
    elif isinstance(obj, list):
        return True if obj == [] else all(map(empty_judge, obj))
    elif isinstance(obj, dict):
        return True if obj == {} else False
    elif isinstance(obj, str):
        return True if obj == '' else False
    else:
        return False


def get_time(tm, fmt=START_TIME_FORMAT):
    """ h """
    if tm is None:
        return
    if isinstance(tm, datetime):
        return tm
    elif isinstance(tm, str):
        return datetime.strptime(tm, fmt)
    elif isinstance(tm, date):
        return datetime.strptime(tm.strftime('%Y-%m-%d'), '%Y-%m-%d')
    else:
        raise TypeError("can not recognized the time[%s] type!" % str(tm))


def clean(features):
    rs = {}
    for k in features:
        if pd.isnull(features[k]):
            features[k] = None
        elif not isinstance(features[k], str):
            features[k] = round(float(features[k]), 5)
            if features[k] == 0:
                continue
            rs[k] = features[k]
        else:
            rs[k] = features[k]
    return rs


def summary_bom(pboc: PBOCEntity):
    """

    :param pboc:
    :return:
    """
    features = dict()
    loanSumHighestOverdueAmountPerMon = transfer_amount(
        get_value('summary_info,overdueAndFellBack,overdueSummary,loanSumHighestOverdueAmountPerMon', pboc.raw_data,
                  '0'))
    loanCardSumHighestOverdueAmountPerMon = transfer_amount(
        get_value('summary_info,overdueAndFellBack,overdueSummary,loanCardSumHighestOverdueAmountPerMon', pboc.raw_data,
                  '0'))
    standardLoanCardSumHighestOverdueAmountPerMon = transfer_amount(
        get_value('summary_info,overdueAndFellBack,overdueSummary,standardLoanCardSumHighestOverdueAmountPerMon',
                  pboc.raw_data, '0'))
    features['pboc_ln_due_amt1m_max_lf'] = loanSumHighestOverdueAmountPerMon
    features['pboc_lc_due_amt1m_max_lf'] = loanCardSumHighestOverdueAmountPerMon
    features['pboc_slc_due_amt1m_max_lf'] = standardLoanCardSumHighestOverdueAmountPerMon

    return features


if __name__ == '__main__':
    test_file = 'test.json'
    with open(test_file, encoding='utf-8') as f:
        obj = json.load(f)
        rs = pboc_bom(obj)
    with open('out.json', 'w') as of:
        print(rs)
        json.dump(rs, of)
