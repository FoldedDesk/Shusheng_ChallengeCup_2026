import pytest

from classifier.problem_spec import build_problem_spec
from core.submission_agent import SubmissionAgent
from tools.exact_textbook_tool import ExactTextbookTool
from tools.sympy_tool import SympyTool


FINAL = r"Remember to put your final answer within \boxed{}."

D8 = rf"""设$D_8$是正方形上的二面体群，下列正确的是：
\begin{{itemize}}
    \item[A.] $D_8$中存在$8$阶元.
    \item[B.] $D_8$的四阶子群一定是Abel群.
    \item[C.] $C(D_8)=\{{1\}}$.
    \item[D.] $[D_8,D_8]$是$2$阶群.
{FINAL}"""

LEBESGUE = rf"""关于函数$f$在区间$[a, b]$上的勒贝格可积性，下列哪些陈述是正确的？A. 如果$\int_a^b |f(x)| \, dx < \infty$，则函数$f$在$[a, b]$上勒贝格可积。
B. 如果函数$f$在$[a, b]$上连续，则它在$[a, b]$上勒贝格可积。
C. 如果函数$f$在$[a, b]$上的不连续点集测度为零，则它在$[a, b]$上勒贝格可积。
D. 如果函数$f$在$[a, b]$上有界，则它在$[a, b]$上勒贝格可积。
{FINAL}"""

COMPACT_REAL = rf"""下列关于实数集上紧集的描述，正确的是：A. 实数集 $\mathbb{{R}}$ 的子集是紧集当且仅当它是闭集。
B. 实数集 $\mathbb{{R}}$ 的子集是紧集当且仅当它是开集。
C. 实数集 $\mathbb{{R}}$ 的子集是紧集当且仅当它是闭集且有界。
D. 实数集 $\mathbb{{R}}$ 的子集是紧集当且仅当它是开集且有界。
E. 实数集 $\mathbb{{R}}$ 的子集是紧集当且仅当它的每个开覆盖都有有限子覆盖。
{FINAL}"""

CAUCHY = rf"""设$(a_n)$是一个数列。如果对于任意的$\epsilon > 0$，存在一个正整数$N$，使得对所有$m, n \geq N$，都有$|a_n - a_m| < \epsilon$，则称$(a_n)$是Cauchy数列。在完备度量空间中，下列关于Cauchy收敛准则的陈述正确的是：A. 每个有界数列都是Cauchy数列。
B. 每个收敛数列都是Cauchy数列。
C. 每个Cauchy数列都是有界的。
D. 每个Cauchy数列在完备度量空间中都收敛。
{FINAL}"""

DUALITY = rf"""关于线性规划的对偶问题，下列说法正确的是：A. 原问题的目标函数在对偶问题中变为约束条件。
B. 如果原问题是求最小值问题，对偶问题也必定是求最小值问题。
C. 原问题约束条件的系数在对偶问题中变为目标函数的系数。
D. 对偶问题的对偶问题会返回原问题。
E. 原问题的可行域与对偶问题的约束条件互为对应。
{FINAL}"""

CONDITION_NUMBER = rf"""7. 矩阵A 的条件数定义是:

A. \(\kappa(A)=\sqrt{{|A|_{{1}}|A^{{-1}}|_{{1}}}}\) B. \(\kappa(A)=|A|_{{1}}|A^{{-1}}|_{{1}}\) C. \(\kappa(A)=\sqrt{{|A|_{{2}}|A^{{-1}}|_{{2}}}}\) D. \(\kappa(A)=|A|_{{2}}|A^{{-1}}|_{{2}}\)
{FINAL}"""

PDE_DISCRETIZATION = rf"""4. 对于偏微分方程 \( D e l t a\;u\;=\;f \) 在区域 \(\Omega\) 上,边界条件为 \( u=g\;o n\;\;\partial\;\Omega \),使用(\ )方法进行离散化处理,可以有效逼近解。
{FINAL}"""

TIME_SERIES = rf"""6. 时间序列的构成要素有（）。

A.长期趋势B.季节变动C.循环变动D.不规则变动E.随机变动
{FINAL}"""

SEASONAL_ADJUSTMENT = rf"""10．对于时间序列的季节调整，常用的方法有( )、( )
{FINAL}"""

DISPERSION_MEASURE = rf"""1.在统计学中，用来表示数据分散程度的一个指标是
{FINAL}"""

AGGREGATE_RATIO = rf"""判断：5. 两个总量指标时间数列相比照得到的时间数列一定是相对数时间数列。（
{FINAL}"""

UNKNOWN_FORM = rf"""10、在研究某种疾病的发病率与环境因素的关系时，由于无法确定环境因素对发病率的具体函数形式，应采用哪种回归方法？（）

A. 线性回归B. 非线性回归C. 逻辑回归D. 以上都不对
{FINAL}"""

STEPWISE = rf"""3.在逐步回归法中，若新引入的变量使得（），则该变量应被剔除。

A.某个旧变量的t检验不显著

B.调整的判定系数减小

C.F检验不显著

D.以上都有可能
{FINAL}"""

NONLINEAR = rf"""5.非线性回归模型的参数估计通常采用（）方法。

A.最小二乘法

B.极大似然法

C.牛顿－拉夫森法

D.以上都可以
{FINAL}"""

HETEROSCEDASTIC_OLS = rf"""判断：4.异方差性会导致普通最小二乘估计量的方差增大。（）
{FINAL}"""

HETEROSCEDASTIC_CONSEQUENCE = rf"""10.异方差性会导致参数估计量的方差（ ）
{FINAL}"""

HETEROSCEDASTIC_CONSEQUENCE_ANSWER = (
    "异方差性不会导致参数估计量的偏误，但会使传统方差估计失效，即低估或高估真实方差，"
    "导致OLS估计量不再是有效估计。"
)


TEXTBOOK_INTEGRATION_CASES = (
    (D8, "square_dihedral_facts", "B,D", "closed_world_group_fact_table"),
    (
        LEBESGUE,
        "lebesgue_integrability_facts",
        "A,B",
        "closed_world_integrability_fact_table",
    ),
    (
        COMPACT_REAL,
        "compact_real_facts",
        "C,E",
        "heine_borel_and_open_cover_definition",
    ),
    (
        CAUCHY,
        "cauchy_complete_space_facts",
        "B,C,D",
        "complete_metric_space_cauchy_theorems",
    ),
    (
        DUALITY,
        "linear_programming_duality_facts",
        "D",
        "standard_lp_duality_fact_table",
    ),
    (
        CONDITION_NUMBER,
        "matrix_condition_number_definition",
        "B",
        "matrix_one_norm_condition_definition",
    ),
    (
        PDE_DISCRETIZATION,
        "dirichlet_pde_discretization_methods",
        "有限差分法、有限元法（或有限体积法）",
        "elliptic_pde_discretization_families",
    ),
    (
        TIME_SERIES,
        "time_series_components",
        "A,B,C,D,E",
        "classical_time_series_decomposition",
    ),
    (
        SEASONAL_ADJUSTMENT,
        "seasonal_adjustment_methods",
        "移动平均法、时间序列分解法",
        "classical_seasonal_adjustment_methods",
    ),
    (
        DISPERSION_MEASURE,
        "dispersion_measure_standard_deviation",
        "标准差",
        "descriptive_statistics_definition",
    ),
    (
        AGGREGATE_RATIO,
        "aggregate_series_ratio_truth",
        "正确",
        "statistical_index_definition",
    ),
    (
        UNKNOWN_FORM,
        "unknown_form_regression",
        "D",
        "regression_model_classification",
    ),
    (
        STEPWISE,
        "stepwise_removal",
        "D",
        "stepwise_retesting_criteria",
    ),
    (
        NONLINEAR,
        "nonlinear_regression_estimation",
        "A",
        "nonlinear_least_squares_definition",
    ),
    (
        HETEROSCEDASTIC_OLS,
        "heteroscedastic_ols_variance_truth",
        "错误",
        "heteroscedastic_variance_direction_check",
    ),
    (
        HETEROSCEDASTIC_CONSEQUENCE,
        "heteroscedastic_parameter_variance_consequence",
        HETEROSCEDASTIC_CONSEQUENCE_ANSWER,
        "heteroscedastic_ols_inference_consequences",
    ),
)

TEXTBOOK_OPERATIONS = frozenset(case[1] for case in TEXTBOOK_INTEGRATION_CASES)


class NoCallClient:
    def __init__(self) -> None:
        self.calls = 0

    def chat_result(self, **kwargs):
        self.calls += 1
        raise AssertionError(f"certified textbook route called the model: {kwargs}")


@pytest.mark.parametrize(
    "problem, expected",
    [
        (D8, "本地二面体群选择答案: B,D"),
        (LEBESGUE, "本地勒贝格可积选择答案: A,B"),
        (COMPACT_REAL, "本地实数紧集选择答案: C,E"),
        (CAUCHY, "本地Cauchy准则选择答案: B,C,D"),
        (DUALITY, "本地线性规划对偶选择答案: D"),
        (CONDITION_NUMBER, "本地矩阵条件数选择答案: B"),
        (PDE_DISCRETIZATION, "本地Dirichlet边值离散化方法: 有限差分法、有限元法（或有限体积法）"),
        (TIME_SERIES, "本地时间序列构成选择答案: A,B,C,D,E"),
        (SEASONAL_ADJUSTMENT, "本地时间序列季节调整方法: 移动平均法、时间序列分解法"),
        (DISPERSION_MEASURE, "本地数据分散程度指标: 标准差"),
        (AGGREGATE_RATIO, "本地总量指标时间数列判断答案: 正确"),
        (UNKNOWN_FORM, "本地回归方法选择答案: D"),
        (STEPWISE, "本地逐步回归选择答案: D"),
        (NONLINEAR, "本地非线性回归选择答案: A"),
        (HETEROSCEDASTIC_OLS, "本地异方差OLS判断答案: 错误"),
        (
            HETEROSCEDASTIC_CONSEQUENCE,
            f"本地异方差参数方差后果: {HETEROSCEDASTIC_CONSEQUENCE_ANSWER}",
        ),
    ],
)
def test_closed_world_textbook_families_emit_one_legacy_hint(problem, expected):
    assert ExactTextbookTool().hints_for(problem) == [expected]


def test_option_labels_are_derived_from_the_current_option_texts():
    reordered = rf"""设$D_8$是正方形上的二面体群，下列正确的是：
A. $D_8$的四阶子群一定是Abel群.
B. $D_8$中存在$8$阶元.
C. $[D_8,D_8]$是$2$阶群.
D. $C(D_8)=\{{1\}}$.
{FINAL}"""

    assert ExactTextbookTool().hints_for(reordered) == [
        "本地二面体群选择答案: A,C"
    ]

    reordered_condition = CONDITION_NUMBER.replace(
        r"A. \(\kappa(A)=\sqrt{|A|_{1}|A^{-1}|_{1}}\) B. \(\kappa(A)=|A|_{1}|A^{-1}|_{1}\)",
        r"A. \(\kappa(A)=|A|_{1}|A^{-1}|_{1}\) B. \(\kappa(A)=\sqrt{|A|_{1}|A^{-1}|_{1}}\)",
    )
    assert ExactTextbookTool().hints_for(reordered_condition) == [
        "本地矩阵条件数选择答案: A"
    ]


@pytest.mark.parametrize(
    "problem",
    [
        D8.replace("下列正确的是", "下列错误的是"),
        D8.replace("$[D_8,D_8]$是$2$阶群", "$[D_8,D_8]$是$4$阶群"),
        D8.replace(f"\n{FINAL}", f"\n    \\item[E.] $D_8$是循环群.\n{FINAL}"),
        LEBESGUE.replace(r"< \infty", r"\leq \infty"),
        COMPACT_REAL.replace("闭集且有界", "闭集或有界"),
        CAUCHY.replace(r"|a_n - a_m| < \epsilon", r"|a_n-a_m| \leq \epsilon"),
        DUALITY.replace("会返回原问题", "不会返回原问题"),
        CONDITION_NUMBER.replace(r"|A|_{2}|A^{-1}|_{2}", r"|A|_{F}|A^{-1}|_{F}"),
        PDE_DISCRETIZATION.replace(r"u=g", r"\partial_n u=g"),
        TIME_SERIES.replace("随机变动", "偶然误差"),
        SEASONAL_ADJUSTMENT.replace("季节调整", "季节预测"),
        SEASONAL_ADJUSTMENT.replace("( )、( )", "( )"),
        DISPERSION_MEASURE.replace("分散程度", "集中趋势"),
        AGGREGATE_RATIO.replace("一定是", "不一定是"),
        UNKNOWN_FORM.replace("具体函数形式", "具体线性函数形式"),
        STEPWISE.replace("F检验不显著", "卡方检验不显著"),
        NONLINEAR.replace("通常采用", "不能采用"),
        HETEROSCEDASTIC_OLS.replace("方差增大", "方差减小"),
        HETEROSCEDASTIC_CONSEQUENCE.replace("异方差性", "同方差性"),
        HETEROSCEDASTIC_CONSEQUENCE.replace(
            "会导致参数估计量的方差（ ）", "会导致OLS参数估计量有偏吗（ ）"
        ),
    ],
)
def test_any_changed_contract_or_unknown_statement_rejects_the_whole_hint(problem):
    assert ExactTextbookTool().hints_for(problem) == []


@pytest.mark.parametrize(
    "problem",
    [
        LEBESGUE.replace(
            "C. 如果函数$f$在$[a, b]$上的不连续点集测度为零，则它在$[a, b]$上勒贝格可积。\n",
            "",
        ),
        DUALITY.replace("E. 原问题的可行域与对偶问题的约束条件互为对应。\n", ""),
        "10.异方差性会导致参数估计量（ ）",
        "关于紧集，下列正确的是：A.闭集 B.有界集",
        "",
    ],
)
def test_incomplete_or_nearby_questions_do_not_emit_hints(problem):
    assert ExactTextbookTool().hints_for(problem) == []


@pytest.mark.parametrize(
    "problem, operation, expected, certificate_method",
    TEXTBOOK_INTEGRATION_CASES,
)
def test_textbook_result_has_registered_contract_certificate_and_whole_goal_scope(
    problem,
    operation,
    expected,
    certificate_method,
):
    spec = build_problem_spec(problem)
    matching = [
        result
        for result in SympyTool().results_for(problem)
        if result.operation == operation
    ]

    assert len(matching) == 1
    result = matching[0]
    assert result.result == expected
    assert result.contract is not None
    assert result.contract.operation == operation
    assert result.contract.whole_answer_capable
    assert result.contract.certified
    assert result.verified
    assert result.whole_answer_eligible
    assert result.certificate.passed
    assert result.certificate.method == certificate_method
    assert not result.certificate.issues
    assert {
        "registered_operation",
        "deterministic_handler_matched",
        "nonempty_result",
        *result.contract.required_problem_facts,
    } <= set(result.certificate.checks)

    evidence = SubmissionAgent._tool_evidence(matching, spec)
    assert len(evidence) == 1
    assert evidence[0].operation == operation
    assert evidence[0].result == expected
    assert evidence[0].scope == "whole_goal"
    assert evidence[0].verified
    assert evidence[0].certificate_method == certificate_method
    assert not evidence[0].certificate_issues


@pytest.mark.parametrize(
    "problem, operation, expected, certificate_method",
    TEXTBOOK_INTEGRATION_CASES,
)
def test_exact_textbook_whole_goal_bypasses_the_model_end_to_end(
    problem,
    operation,
    expected,
    certificate_method,
):
    client = NoCallClient()
    result = SubmissionAgent(client).solve(problem, {"idx": "test"})

    assert client.calls == 0
    assert result["final_response"] == rf"\boxed{{{expected}}}"
    call_plan = next(item for item in result["trace"] if item["step"] == "call_plan")
    assert call_plan["content"]["route"] == "certified_tool"
    assert call_plan["content"]["max_model_calls"] == 0
    selection = next(item for item in result["trace"] if item["step"] == "selection")
    assert selection["content"]["source"] == "sympy_verified"
    finalize = next(item for item in result["trace"] if item["step"] == "finalize")
    assert finalize["content"]["model_call_count"] == 0
    tool_trace = next(item for item in result["trace"] if item["step"] == "tool_evidence")
    assert tool_trace["content"]["whole_goal_count"] == 1
    certificate = next(
        item
        for item in tool_trace["content"]["certificates"]
        if item["operation"] == operation
    )
    assert certificate["scope"] == "whole_goal"
    assert certificate["passed"]
    assert certificate["method"] == certificate_method
    assert not certificate["issues"]


@pytest.mark.parametrize(
    "problem",
    [
        D8.replace("下列正确的是", "下列错误的是"),
        LEBESGUE.replace(r"< \infty", r"\leq \infty"),
        COMPACT_REAL.replace("闭集且有界", "闭集或有界"),
        CAUCHY.replace(r"|a_n - a_m| < \epsilon", r"|a_n-a_m| \leq \epsilon"),
        DUALITY.replace("会返回原问题", "不会返回原问题"),
        CONDITION_NUMBER.replace("条件数定义是", "谱半径定义是"),
        PDE_DISCRETIZATION.replace("离散化处理", "解析求解"),
        TIME_SERIES.replace("随机变动", "偶然误差"),
        SEASONAL_ADJUSTMENT.replace("季节调整", "季节预测"),
        DISPERSION_MEASURE.replace("分散程度", "集中趋势"),
        AGGREGATE_RATIO.replace("一定是", "不一定是"),
        UNKNOWN_FORM.replace("具体函数形式", "具体线性函数形式"),
        STEPWISE.replace("F检验不显著", "卡方检验不显著"),
        NONLINEAR.replace("通常采用", "不能采用"),
        HETEROSCEDASTIC_OLS.replace("方差增大", "方差减小"),
        HETEROSCEDASTIC_CONSEQUENCE.replace("异方差性", "同方差性"),
        HETEROSCEDASTIC_CONSEQUENCE.replace(
            "会导致参数估计量的方差（ ）", "会导致OLS参数估计量有偏吗（ ）"
        ),
    ],
)
def test_changed_textbook_contract_never_bypasses_the_model(problem):
    results = SympyTool().results_for(problem)
    assert TEXTBOOK_OPERATIONS.isdisjoint(result.operation for result in results)

    client = NoCallClient()
    result = SubmissionAgent(client).solve(problem, {"idx": "changed-contract"})

    assert client.calls > 0
    call_plan = next(item for item in result["trace"] if item["step"] == "call_plan")
    assert call_plan["content"]["route"] == "model"
    tool_trace = next(item for item in result["trace"] if item["step"] == "tool_evidence")
    assert TEXTBOOK_OPERATIONS.isdisjoint(tool_trace["content"]["operations"])
